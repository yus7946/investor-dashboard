"""予報モデルの改善実験（PDCAの Do/Check）。

本番と同じ build_panel / _validate（直近8四半期・先読みなしのウォークフォワード）で
特徴量セットやモデル設定を比較し、結果を research/results.jsonl に追記する。

使い方:
  python research/experiment.py                # 全実験
  python research/experiment.py base adr       # 指定した実験だけ
  python research/experiment.py --period 10y   # 学習期間を変える
  python research/experiment.py --refresh      # 株価キャッシュを取り直す
"""
import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import forecast.overnight as ov  # noqa: E402
from data.universe import MASTER  # noqa: E402

ov.RESEARCH = True  # 研究用の候補特徴量も作る
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results.jsonl")

BASE = ov.OVERNIGHT + ov.STOCK
P = ov.FEATURES
PRE = ["nkd_pre", "nkd_delta", "pred_delta", "es_delta"]  # 寄付き直前の先物（PDCA#12）
# 「この指標が上がるほど上がりやすい」が理屈で決まる特徴量（単調性制約の対象）
UP = {"nkd", "spx", "pred_gap", "pred_sox", "pred_fx", "pred_all", "adr_imp", "ewj_imp", "pred_ridge", "pred_ridge_adr",
      *ov.EXTRA_SEC}


def lr_pipe(C=0.05):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         FunctionTransformer(lambda a: np.clip(a, -5, 5)), LogisticRegression(C=C, max_iter=2000))


class Blend:
    """勾配ブースティングとロジスティック回帰の確率の平均（性質の違うモデルの誤差を打ち消し合わせる）。"""
    def __init__(self, w=0.5):
        self.w = w

    def fit(self, X, y, sample_weight=None):
        self.g = ov._model().fit(X, y, sample_weight=sample_weight)
        self.l = lr_pipe().fit(X, y)
        return self

    def predict_proba(self, X):
        return self.w * self.g.predict_proba(X) + (1 - self.w) * self.l.predict_proba(X)


def mono(feats):
    cst = {f: 1 for f in feats if f in UP}
    return lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=400,
                                                  l2_regularization=1.0, random_state=0, monotonic_cst=cst)


def gbm(**kw):
    base = dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=1.0, random_state=0)
    return lambda: HistGradientBoostingClassifier(**{**base, **kw})


class Bag:
    """特徴量を毎回ランダムに7割だけ使う勾配ブースティングを5本作って平均（ばらつきを減らす）。"""
    def fit(self, X, y, sample_weight=None):
        self.ms = [gbm(max_features=0.7, random_state=i)().fit(X, y, sample_weight=sample_weight) for i in range(5)]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], axis=0)


def swap(feats, new):
    """本番の pred_ridge_adr を候補に置き換えた特徴量リスト。"""
    return [new if f == "pred_ridge_adr" else f for f in feats]


CAL = ["dow", "gapdays", "monthpos", "monthend", "sq"]


def size_weight(tr, y):
    """値動きの大きい日ほど重く学ぶ（小さな値動きの上げ下げは雑音が多い）。ふだんの値動き比で上限3倍。"""
    return np.clip((tr[y].abs() / tr.vol20).fillna(1).to_numpy(), 0.1, 3.0)


EXPERIMENTS = {
    "base": (BASE, None),
    "mkt": (BASE + ov.EXTRA_MKT, None),
    "sec": (BASE + ov.EXTRA_SEC, None),
    "adr": (BASE + ov.EXTRA_ADR, None),
    "sec_adr": (BASE + ov.EXTRA_SEC + ov.EXTRA_ADR, None),
    "prod": (ov.FEATURES, None),  # 本番で採用中の特徴量
    "v1": (ov.FEATURES_V1, None),  # 2026-10-13〜11-18の本番（リッジ回帰なし）
    "no_stock": (ov.OVERNIGHT + ov.EXTRA_SEC + ov.EXTRA_ADR, None),  # 前日までの値動き（反転傾向など）を除外
    "hl730": (ov.FEATURES, None, 730),  # 新しいデータほど重く（半減期2年）
    "hl365": (ov.FEATURES, None, 365),  # 半減期1年
    "hl180": (ov.FEATURES, None, 180),  # 半減期半年
    # PDCA#7（2026-11）で効果なしだった手法（結果は results.jsonl / PDCA.md）。再検証用に残す
    "mono": (P, mono(P)),  # 単調性制約
    "wt": (P, None, {"weight_fn": size_weight}),  # 値動きの大きさで重み付け
    "ens": (P, Blend),  # 勾配ブースティング＋ロジスティック回帰
    "ridge_noadr": (ov.FEATURES_V1 + ["pred_ridge"], None),  # リッジ回帰のみ（ADRを入れない）
    # PDCA#8（2026-11）
    "pool03": (swap(P, "pred_ridge_pool03"), None),  # リッジ回帰の係数を全銘柄平均へ縮める（経験ベイズ）
    "pool10": (swap(P, "pred_ridge_pool10"), None),  # 同・強く
    "adrb": (swap(P, "pred_ridge_b"), None),  # 日本企業ADRバスケットを20個目の指標に
    "cal": (P + CAL, None),  # 曜日・休日をはさむか・月末月初・SQ
    "rel": (P + ["rel_ridge", "rank_ridge"], None),  # その日の全銘柄の中での想定ギャップの位置
    "gbm_mf": (P, gbm(max_features=0.5)),  # 分岐ごとに特徴量の半分だけ見る
    "gbm_slow": (P, gbm(learning_rate=0.03, max_iter=400, min_samples_leaf=800)),  # ゆっくり・強い正則化
    "bag5": (P, Bag),  # 5本の平均
    # PDCA#12（2026-11）: 寄付き直前の先物（1時間足は直近730日のみ）
    "preopen": (P + PRE, None),
    "preopen_nkd": (P + ["nkd_pre", "nkd_delta", "pred_delta"], None),
    "preopen_swap": ([{"nkd": "nkd_f", "pred_gap": "pred_gap_f", "pred_all": "pred_all_f", "pred_ridge_adr": "pred_ridge_adr_f"}.get(f, f) for f in P], None),
    # PDCA#9（2026-11）: 最近の当たり具合（信頼度）
    "trust_tk": (P + ["hit_tk60", "slope_tk60"], None),
    "trust_mkt": (P + ["skill_mkt20", "follow_mkt20", "hit_mkt20"], None),
    "trust": (P + ["hit_tk60", "slope_tk60", "skill_mkt20", "follow_mkt20", "hit_mkt20"], None),
    "trim3": (P, None, {"weight_fn": lambda tr, y: (tr[y].abs() / tr.vol20 <= 3).fillna(True).to_numpy(float)}),  # 平常の3倍超の日を学習から外す
    "trim4": (P, None, {"weight_fn": lambda tr, y: (tr[y].abs() / tr.vol20 <= 4).fillna(True).to_numpy(float)}),  # 同・4倍
    "all": (BASE + ov.EXTRA_MKT + ov.EXTRA_SEC + ov.EXTRA_ADR, None),
    "all_big": (BASE + ov.EXTRA_MKT + ov.EXTRA_SEC + ov.EXTRA_ADR,
                lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=31,
                                                       min_samples_leaf=300, l2_regularization=1.0, random_state=0)),
}


def load_px(period: str, refresh: bool):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"px_{period}.pkl")
    if os.path.exists(path) and not refresh:
        return pickle.load(open(path, "rb"))
    px = ov.download([t for t, _ in MASTER], period=period)
    pickle.dump(px, open(path, "wb"))
    return px


def paired(base: dict, cand: dict, y: str, last_days: int | None = None, before=None) -> tuple[float, float]:
    """同じ日・同じ銘柄の予測で的中率の差をとり、日単位の標準誤差を返す（日内の銘柄は独立でないため）。
    last_days を指定すると直近その営業日数だけで比べる。"""
    m = base["_preds"][y].merge(cand["_preds"][y], on=["date", "tk", "label"], suffixes=("_b", "_c"))
    m["d"] = ((m.p_c > .5) == m.label).astype(float) - ((m.p_b > .5) == m.label).astype(float)
    daily = m.groupby("date").d.mean() * 100
    if before is not None:
        daily = daily[daily.index < before]
    if last_days:
        daily = daily.tail(last_days)
    return round(float(daily.mean()), 2), round(float(daily.std(ddof=1) / np.sqrt(len(daily))), 2)


DAY_FEATS = ["nkd", "spx", "sox", "fx", "ewj_imp", "ixic", "rut", "vix", "tnx", "oil"]
DAY_MEANS = ["pred_all", "pred_gap", "adr_imp", "pred_ridge_adr", "r1", "x5"] + ov.EXTRA_SEC


def market_eval(panel, base_v):
    """市場全体（全銘柄の平均騰落）の方向を、銘柄予報の平均ではなく日単位の専用モデルで当てる実験。"""
    g = panel.groupby("date")
    day = pd.concat([g[DAY_FEATS].first(), g[DAY_MEANS].mean().add_suffix("_m"), g.y_cc.mean().rename("ret")], axis=1)
    day = day[day.ret.notna() & (day.ret != 0)]
    feats = [c for c in day.columns if c != "ret"]
    lab = (day.ret > 0).astype(int)
    cur = base_v["_preds"]["y_cc"].groupby("date").p.mean()  # 現行: 銘柄の上昇確率の平均
    folds = pd.date_range(end=panel.date.max(), periods=9, freq="QS")
    models = {"lr": lambda: lr_pipe(0.1),
              "gbm": lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=7,
                                                            min_samples_leaf=40, l2_regularization=1.0, random_state=0)}
    out = {k: [] for k in models}
    for i, f0 in enumerate(folds):
        f1 = folds[i + 1] if i + 1 < len(folds) else day.index.max() + pd.Timedelta(days=1)
        tr, te = day[day.index < f0 - pd.Timedelta(days=3)], day[(day.index >= f0) & (day.index < f1)]
        if len(te) == 0 or len(tr) < 250:
            continue
        for k, mf in models.items():
            out[k].append(pd.Series(mf().fit(tr[feats], lab[tr.index]).predict_proba(te[feats])[:, 1], te.index))
    res = pd.DataFrame({k: pd.concat(v) for k, v in out.items()}).join(cur.rename("cur"), how="inner")
    res["blend"] = (res.cur + res.lr) / 2
    y = lab[res.index]
    hit = {k: ((res[k] > .5) == y).astype(float) for k in ["cur", "lr", "gbm", "blend"]}
    for k in ["cur", "lr", "gbm", "blend"]:
        strong = (res[k] >= .6) | (res[k] <= .4)
        d = (hit[k] - hit["cur"]) * 100
        print(f"market:{k:5s} {hit[k].mean() * 100:.1f}% (強{hit[k][strong].mean() * 100:.1f}%/{int(strong.sum())}日) "
              f"vs cur {d.mean():+.1f}pt(±{d.std(ddof=1) / np.sqrt(len(d)):.1f}) n={len(d)}", flush=True)


def add_preopen(panel: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """PDCA#12: 寄付き直前（日本時間8:00〜9:00の1時間足の終値＝9:00直前の値段）の先物の動き。
    Yahooの1時間足は直近730日分だけなので、それより前の行は欠損（勾配ブースティングは欠損を扱える）。
    nkd_pre: 先物の寄付き直前の値段÷前日の日経平均終値（ベーシス調整は既存のnkdと同じ）／nkd_delta: 清算値から寄付き直前までの動き／
    pred_delta: 銘柄の感応度×その動き／es_delta: S&P500先物の清算値から寄付き直前までの動き"""
    import yfinance as yf
    path = os.path.join(CACHE, "fut_60m.pkl")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < 86400:
        h = pickle.load(open(path, "rb"))
    else:
        h = yf.download(["NKD=F", "ES=F"], period="730d", interval="60m", auto_adjust=True, group_by="column", progress=False)["Close"]
        pickle.dump(h, open(path, "wb"))
    jst, cst = h.tz_convert("Asia/Tokyo"), h.tz_convert("America/Chicago")
    pre = jst[jst.index.hour == 8].copy()  # 8:00〜9:00の足。終値は9:00直前の値段
    pre.index = pre.index.tz_localize(None).normalize()
    st = cst[cst.index.hour == 14].copy()  # 米中部14:00〜15:00の足の終値＝清算値（日足の終値と照合済み）
    st.index = st.index.tz_localize(None).normalize()
    days = pd.DatetimeIndex(sorted(panel.date.unique()))
    nxt = days[1:].append(pd.DatetimeIndex([days[-1] + pd.Timedelta(days=1)]))  # 行tの翌営業日
    cal = pd.date_range(min(st.index.min(), days[0]), nxt[-1])
    settle = st.reindex(cal).ffill().shift(1)  # 各日dについて「dより前の暦日に確定した清算値」
    n225 = px["Close"]["^N225"].reindex(days).ffill()
    rows = pd.DataFrame(index=days)
    for k, col in (("NKD=F", "nkd_delta"), ("ES=F", "es_delta")):
        rows[col] = pre[k].reindex(nxt).values / settle[k].reindex(nxt).values - 1
    raw_settle = settle["NKD=F"].reindex(nxt).values / n225.values - 1
    med = pd.Series(raw_settle, index=days).rolling(60, min_periods=20).median()
    rows["nkd_pre"] = pre["NKD=F"].reindex(nxt).values / n225.values - 1 - med.values
    out = panel.merge(rows[["nkd_pre", "nkd_delta", "es_delta"]], left_on="date", right_index=True, how="left")
    out["pred_delta"] = out.gbeta * out.nkd_delta
    # 置き換え版: 先物の値段そのものを寄付き直前の値に差し替える（取れない日は清算値のまま）。リッジ回帰の先物の項は単独の感応度で近似
    out["nkd_f"] = out.nkd_pre.fillna(out.nkd)
    out["pred_gap_f"] = out.gbeta * out.nkd_f
    out["pred_all_f"] = out.pred_gap_f + out.pred_sox + out.pred_fx
    out["pred_ridge_adr_f"] = out.pred_ridge_adr + out.gbeta * (out.nkd_f - out.nkd)
    cov = out.nkd_delta.notna().mean() * 100
    print(f"寄付き直前の先物: 行の{cov:.0f}%で取得（{out.loc[out.nkd_delta.notna(), 'date'].min()}〜）"
          f" 清算値からの動きの大きさ（絶対値の中央値） {out.nkd_delta.abs().median() * 100:.2f}%", flush=True)
    return out


def market_paired(base: dict, cand: dict) -> tuple[float, float]:
    """市場全体（銘柄の上昇確率の平均 vs 全銘柄の平均騰落）の的中の差。日ごとの0/1の差の平均と標準誤差。"""
    def daily(v):
        d = v["_preds"]["y_cc"]
        return d.groupby("date").p.mean()
    m = pd.concat([daily(base).rename("b"), daily(cand).rename("c")], axis=1).dropna()
    m = m.join(base["_mret"], how="inner")
    d = (((m.c > .5) == (m.ret > 0)).astype(float) - ((m.b > .5) == (m.ret > 0)).astype(float)) * 100
    return round(float(d.mean()), 2), round(float(d.std(ddof=1) / np.sqrt(len(d))), 2)


def replay(px, dates):
    """本番と同じ手順（対象日より前のデータだけで全期間学習→対象日を予報）を、確定済みの正しいデータでやり直して答え合わせ。"""
    tickers = [t for t, _ in MASTER]
    full, _ = ov.build_panel(px, tickers, pd.Timestamp(datetime.now(ov.JST).date()) + pd.Timedelta(days=7))
    for d in dates:
        T = pd.Timestamp(d)
        panel, inp = ov.build_panel(px, tickers, T)
        last = panel[panel.date == panel.date.max()]
        last = last[last[ov.OVERNIGHT + ov.STOCK].notna().all(axis=1)]
        ans = full[full.date == panel.date.max()].set_index("tk")
        res = {}
        feats = ov.FEATURES_V1 if d < "2026-11-19" else ov.FEATURES  # その日に本番で使っていたモデル
        for y in ("y_cc", "y_gap"):
            dd = ov._labeled(panel, y)
            p = ov._model().fit(dd[feats], dd.label).predict_proba(last[feats])[:, 1]
            a = ans.loc[last.tk, y].to_numpy()
            ok = ~np.isnan(a) & (a != 0)
            res[y] = round(float(((p > .5) == (a > 0))[ok].mean()) * 100, 1)
            if y == "y_cc":
                res["mkt"] = f"{p.mean() * 100:.0f}%→{'上' if np.nanmean(a) > 0 else '下'}"
        print(f"replay {d}: 終値{res['y_cc']}% 寄付き{res['y_gap']}% 市場{res['mkt']} 先物{inp['nkdPct']}%", flush=True)
        print("replayjson " + json.dumps({"d": d, "cc": res["y_cc"], "gap": res["y_gap"], "nkd": inp["nkdPct"]}), flush=True)


def e2e():
    """本番の予報処理（先物の清算値への置き換えを含む）を通しで実行。運用記録には書き込まない。"""
    import tempfile
    ov.RESEARCH = False
    tmp = os.path.join(tempfile.mkdtemp(), "h.json")
    if os.path.exists(ov.HISTORY_PATH):  # 本物の記録のコピーで、前回予報の答え合わせ・再現チェックまで通す
        with open(ov.HISTORY_PATH, encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as g:
            g.write(f.read())
    ov.HISTORY_PATH = tmp
    fc = ov.build_overnight_forecast(MASTER)
    json.dumps(fc, ensure_ascii=False)
    with open(tmp, encoding="utf-8") as f:
        last = json.load(f)["days"][-1]
    print("e2e last graded:", json.dumps(last, ensure_ascii=False), flush=True)
    print("e2e ok:", fc["targetDate"], "stocks", len(fc["stocks"]), "inputs", fc["inputs"],
          "val", fc["validation"]["y_cc"]["accuracy"], fc["validation"]["y_gap"]["accuracy"], flush=True)
    ov.RESEARCH = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--quarters", type=int, default=8, help="検証する四半期数。8超なら、候補選びに使っていない古い期間の差も表示")
    a = ap.parse_args()
    if "e2e" in a.names:
        e2e()
        a.names.remove("e2e")
    px = load_px(a.period, a.refresh)
    rp = [n for n in a.names if n.startswith("replay=")]
    if rp:
        replay(px, rp[0].split("=", 1)[1].split(","))
        a.names.remove(rp[0])
        if not a.names:
            return
    target = ov._target_date(datetime.now(ov.JST))
    panel, _ = ov.build_panel(px, [t for t, _ in MASTER], target)
    if any(n.startswith("preopen") for n in a.names):
        panel = add_preopen(panel, px)
    print(f"panel rows={len(panel)} stocks={panel.tk.nunique()} last={panel.date.max().date()}", flush=True)
    names = a.names or list(EXPERIMENTS)
    baseline = names[0]  # 先頭の実験を比較の基準にする
    base = None
    for name in names:
        if name == "market":
            market_eval(panel, base)
            continue
        feats, model_fn, *rest = EXPERIMENTS[name]
        kw = rest[0] if rest and isinstance(rest[0], dict) else {"half_life": rest[0] if rest else None}
        t0 = time.time()
        v = ov._validate(panel, quarters=a.quarters, feats=feats, model_fn=model_fn, keep_preds=True, **kw)
        v["_mret"] = panel.groupby("date").y_cc.mean().rename("ret")
        if name == baseline:
            base = v
        row = {
            "at": datetime.now(ov.JST).strftime("%Y-%m-%d %H:%M"), "name": name, "period": a.period,
            "from": v["from"], "to": v["to"], "nFeat": len(feats),
            "cc": v["y_cc"]["accuracy"], "ccHC": v["y_cc"]["highConfAccuracy"], "ccHCn": v["y_cc"]["highConfN"],
            "gap": v["y_gap"]["accuracy"], "gapHC": v["y_gap"]["highConfAccuracy"], "gapHCn": v["y_gap"]["highConfN"],
            "mkt": v["market"]["accuracy"], "mktStrong": v["market"]["strongAccuracy"], "sec": round(time.time() - t0),
            "ccAUC": v["y_cc"]["auc"], "gapAUC": v["y_gap"]["auc"],
        }
        if name != baseline:
            row["baseline"] = baseline
            row["ccDiff"], row["ccDiffSE"] = paired(base, v, "y_cc")
            row["gapDiff"], row["gapDiffSE"] = paired(base, v, "y_gap")
            row["ccDiff60"], row["ccDiff60SE"] = paired(base, v, "y_cc", 60)
            row["gapDiff60"], row["gapDiff60SE"] = paired(base, v, "y_gap", 60)
            row["mktDiff"], row["mktDiffSE"] = market_paired(base, v)
            if a.quarters > 8:  # 候補選び（直近8四半期）に使っていない古い期間だけでの差＝独立な確認
                old = pd.date_range(end=panel.date.max(), periods=9, freq="QS")[0]
                row["quarters"] = a.quarters
                row["ccDiffOld"], row["ccDiffOldSE"] = paired(base, v, "y_cc", before=old)
                row["gapDiffOld"], row["gapDiffOldSE"] = paired(base, v, "y_gap", before=old)
        print(f"{name:8s} 終値{row['cc']}%(高確信{row['ccHC']}%/{row['ccHCn']}件) "
              f"寄付き{row['gap']}%(高確信{row['gapHC']}%/{row['gapHCn']}件) 市場{row['mkt']}%(強{row['mktStrong']}%) "
              f"AUC {row['ccAUC']}/{row['gapAUC']} {row['sec']}s"
              + (f"\n         vs {baseline}: 終値{row['ccDiff']:+}pt(±{row['ccDiffSE']}) 寄付き{row['gapDiff']:+}pt(±{row['gapDiffSE']})"
                 f" ／ 直近60日: 終値{row['ccDiff60']:+}pt(±{row['ccDiff60SE']}) 寄付き{row['gapDiff60']:+}pt(±{row['gapDiff60SE']})"
                 f" ／ 市場{row['mktDiff']:+}pt(±{row['mktDiffSE']})"
                 + (f"\n         古い{a.quarters - 8}四半期だけ: 終値{row['ccDiffOld']:+}pt(±{row['ccDiffOldSE']}) "
                    f"寄付き{row['gapDiffOld']:+}pt(±{row['gapDiffOldSE']})" if "ccDiffOld" in row else "")
                 if "ccDiff" in row else ""), flush=True)
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
