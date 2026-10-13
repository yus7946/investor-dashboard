"""翌営業日の方向予報（寄付き前予報）。

検証の結論（2026-10、166銘柄×約4年のウォークフォワード、先読みなし）:
  - 日足テクニカル指標だけで翌日の上げ下げを当てる精度は約51%（コイン投げと同等）。
  - 東京の引け後に動く「米国株・半導体指数・ドル円・CME日経先物」と、銘柄ごとの
    それらへの感応度を使うと、翌日終値の方向は約60%、確信度上位20%では約73%、
    寄付きの方向は約68%（上位20%で約88%）当たる。予測確率と実際の上昇率もほぼ一致。
  - 寄付き後〜大引けや、5〜20日先の対市場超過リターンはほぼ当たらない（約51〜52%）。
したがって「夜間に確定した海外情報が翌日どう効くか」に絞って予報する。

追加通信は yfinance の一括ダウンロード1回のみ（朝のフル更新時だけ実行）。
"""
import json
import os
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError:
    yf = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

JST = timezone(timedelta(hours=9))
HISTORY_PATH = "output/overnight_history.json"
MKT, N225, SPX, SOX, FX, NKD = "1306.T", "^N225", "^GSPC", "^SOX", "JPY=X", "NKD=F"
OVERNIGHT = ["spx", "sox", "fx", "nkd", "gbeta", "sbeta", "fbeta", "pred_gap", "pred_sox", "pred_fx", "pred_all"]
STOCK = ["r1", "x1", "x5", "x20", "vol20", "vratio", "clv", "gap_today", "rsi", "beta"]
FEATURES = OVERNIGHT + STOCK
BANDS = [0, .2, .3, .4, .45, .55, .6, .7, .8, 1.0001]
HIGH_CONF = 0.7  # p>=0.7 または p<=0.3 を「高確信」とする


def _target_date(now: datetime) -> pd.Timestamp:
    """予報の対象日。大引け(15:30)前ならその日、以降・週末なら次の平日。"""
    d = now.date()
    if d.weekday() >= 5 or (now.hour, now.minute) >= (15, 30):
        d += timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return pd.Timestamp(d)


def _last_before(close: pd.DataFrame, sym: str, idx: pd.DatetimeIndex) -> pd.Series:
    """各日dについて「d より前の暦日に確定した最終値」。米国・先物はd日の東京寄付き前に確定済み。"""
    s = close[sym].dropna()
    cal = s.reindex(pd.date_range(s.index[0], idx[-1])).ffill().shift(1)
    return cal.reindex(idx)


def _roll_beta(y: pd.Series, x: pd.Series, n: int = 250) -> pd.Series:
    return y.rolling(n, min_periods=60).cov(x) / x.rolling(n, min_periods=60).var()


def build_panel(px: pd.DataFrame, tickers: list[str], target: pd.Timestamp):
    """銘柄×日付の特徴量パネル。行tの海外系特徴は「t翌営業日の寄付き前に既知の値」。"""
    C, O, H, L, V = (px[k] for k in ("Close", "Open", "High", "Low", "Volume"))
    jp = C[MKT].dropna().index
    jp = jp[jp < target]
    idx = jp.append(pd.DatetimeIndex([target]))

    lv = {k: _last_before(C, k, idx) for k in (SPX, SOX, FX, NKD)}
    spx, sox, fx = (lv[k] / lv[k].shift(1) - 1 for k in (SPX, SOX, FX))
    n225 = C[N225].reindex(idx).ffill()
    nkd_raw = lv[NKD] / n225.shift(1) - 1  # 先物が示す日経の寄付きギャップ
    nkd = nkd_raw - nkd_raw.rolling(60, min_periods=20).median()  # 金利・配当由来のベーシスを除去
    mkt = C[MKT].reindex(idx)
    mkt_r = mkt / mkt.shift(1) - 1

    frames = []
    for s in tickers:
        if s not in C.columns:
            continue
        c, o, h, l, v = (X[s].reindex(idx) for X in (C, O, H, L, V))
        if c.notna().sum() < 300:
            continue
        r = c / c.shift(1) - 1
        gap = o / c.shift(1) - 1
        f = pd.DataFrame(index=idx)
        f["r1"] = r
        for n in (1, 5, 20):
            f[f"x{n}"] = c / c.shift(n) - mkt / mkt.shift(n)
        f["vol20"] = r.rolling(20).std()
        f["vratio"] = v / v.rolling(20).mean().shift(1)
        f["clv"] = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        f["gap_today"] = gap
        up, dn = r.clip(lower=0).rolling(14).mean(), (-r.clip(upper=0)).rolling(14).mean()
        f["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        f["beta"] = r.rolling(120).cov(mkt_r) / mkt_r.rolling(120).var()
        # 銘柄ごとの寄付きギャップの感応度（当日までの実績のみで推定）
        f["gbeta"], f["sbeta"], f["fbeta"] = _roll_beta(gap, nkd), _roll_beta(gap, sox), _roll_beta(gap, fx)
        # 翌営業日の寄付き前に既知の海外情報を行tに載せる
        f["spx"], f["sox"], f["fx"], f["nkd"] = spx.shift(-1), sox.shift(-1), fx.shift(-1), nkd.shift(-1)
        f["pred_gap"] = f.gbeta * f.nkd
        f["pred_sox"] = f.sbeta * f.sox
        f["pred_fx"] = f.fbeta * f.fx
        f["pred_all"] = f.pred_gap + f.pred_sox + f.pred_fx
        f["y_cc"] = r.shift(-1)
        f["y_gap"] = gap.shift(-1)
        f["close"] = c
        f["tk"] = s
        frames.append(f.iloc[:-1])  # 対象日の行自体は特徴量を持たないので落とす
    panel = pd.concat(frames).rename_axis("date").reset_index()
    inputs = {
        "spxPct": _pct(spx.iloc[-1]), "soxPct": _pct(sox.iloc[-1]), "usdjpyPct": _pct(fx.iloc[-1]),
        "usdjpy": _round(lv[FX].iloc[-1], 2), "nkdPct": _pct(nkd_raw.iloc[-1]), "nkd": _round(lv[NKD].iloc[-1], 0),
        "n225": _round(n225.iloc[-2], 0),
    }
    return panel, inputs


def _pct(x):
    return None if x is None or pd.isna(x) else round(float(x) * 100, 2)


def _round(x, nd):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def _model():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=400, l2_regularization=1.0, random_state=0)


def _labeled(panel: pd.DataFrame, y: str) -> pd.DataFrame:
    d = panel[panel[y].notna() & (panel[y] != 0)]
    return d.assign(label=(d[y] > 0).astype(int))


def _band_table(p: np.ndarray, label: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            rows.append({"lo": round(lo * 100), "hi": round(min(hi, 1) * 100), "n": int(m.sum()),
                         "upRate": round(float(label[m].mean()) * 100, 1)})
    return rows


def _validate(panel: pd.DataFrame, quarters: int = 8) -> dict:
    """直近8四半期を1四半期ずつ「それ以前のデータだけで学習→予測」して答え合わせする。"""
    out = {}
    folds = pd.date_range(end=panel.date.max(), periods=quarters + 1, freq="QS")
    mkt_rows = []
    for y in ("y_cc", "y_gap"):
        d = _labeled(panel, y)
        preds = []
        for i, f0 in enumerate(folds):
            f1 = folds[i + 1] if i + 1 < len(folds) else d.date.max() + pd.Timedelta(days=1)
            tr, te = d[d.date < f0 - pd.Timedelta(days=3)], d[(d.date >= f0) & (d.date < f1)]
            if len(te) == 0 or len(tr) < 20000:
                continue
            m = _model().fit(tr[FEATURES], tr.label)
            preds.append(te.assign(p=m.predict_proba(te[FEATURES])[:, 1]))
        P = pd.concat(preds)
        p, lab = P.p.to_numpy(), P.label.to_numpy()
        hc = (p >= HIGH_CONF) | (p <= 1 - HIGH_CONF)
        out[y] = {
            "n": int(len(P)),
            "accuracy": round(float(((p > .5) == lab).mean()) * 100, 1),
            "baseline": round(float(max(lab.mean(), 1 - lab.mean())) * 100, 1),
            "highConfN": int(hc.sum()),
            "highConfAccuracy": round(float(((p[hc] > .5) == lab[hc]).mean()) * 100, 1) if hc.sum() else None,
            "bands": _band_table(p, lab),
        }
        if y == "y_cc":
            mkt_rows = P.groupby("date").agg(p=("p", "mean"), ret=("y_cc", "mean"))
        out["from"], out["to"] = str(P.date.min().date()), str(P.date.max().date())
    hit = (mkt_rows.p > .5) == (mkt_rows.ret > 0)
    strong = (mkt_rows.p >= .6) | (mkt_rows.p <= .4)
    out["market"] = {
        "n": int(len(mkt_rows)), "accuracy": round(float(hit.mean()) * 100, 1),
        "baseline": round(float(max((mkt_rows.ret > 0).mean(), (mkt_rows.ret <= 0).mean())) * 100, 1),
        "strongN": int(strong.sum()),
        "strongAccuracy": round(float(hit[strong].mean()) * 100, 1) if strong.sum() else None,
    }
    return out


def _band_hit(bands: list[dict], p: float) -> float | None:
    """予測確率pと同じ帯の過去実績から「この向きの予報が当たった率」を返す。"""
    for b in bands:
        if b["lo"] <= p * 100 < b["hi"] or (p == 1 and b["hi"] == 100):
            return b["upRate"] if p >= .5 else round(100 - b["upRate"], 1)
    return None


def _update_history(history: dict, panel: pd.DataFrame, today_pred: dict | None) -> dict:
    """前回の予報を実績と照合して日次サマリーに追加し、今回の予報を保留中として保存する。"""
    pend = history.get("pending")
    last_date = panel.date.max()
    if pend and pd.Timestamp(pend["b"]) < last_date:
        base = pd.Timestamp(pend["b"])
        nxt = panel.loc[panel.date > base, "date"].min()  # 予報後の最初の取引日
        rows = panel[panel.date == base].set_index("tk")
        cc_hit = gap_hit = n = hc_n = hc_hit = 0
        for tk, (pc, pg) in pend["preds"].items():
            if tk not in rows.index:
                continue
            ac, ag = rows.at[tk, "y_cc"], rows.at[tk, "y_gap"]
            if pd.isna(ac) or pd.isna(ag):
                continue
            n += 1
            cc_hit += int((pc >= 50) == (ac > 0))
            gap_hit += int((pg >= 50) == (ag > 0))
            if pc >= HIGH_CONF * 100 or pc <= (1 - HIGH_CONF) * 100:
                hc_n += 1
                hc_hit += int((pc >= 50) == (ac > 0))
        if n:
            mret = float(rows["y_cc"].dropna().mean())
            history.setdefault("days", []).append({
                "d": str(nxt.date()), "n": n, "cc": cc_hit, "gap": gap_hit, "hcN": hc_n, "hc": hc_hit,
                "mp": pend.get("mkt"), "mUp": mret > 0,
            })
            history["days"] = history["days"][-250:]
        history.pop("pending")
    if today_pred:  # 暫定予報は答え合わせ対象にしない
        history["pending"] = today_pred
    return history


def _live_stats(history: dict) -> dict | None:
    days = history.get("days") or []
    if not days:
        return None
    n = sum(d["n"] for d in days)
    hc_n = sum(d["hcN"] for d in days)
    mk = [d for d in days if d.get("mp") is not None]
    return {
        "days": len(days), "since": days[0]["d"], "n": n,
        "ccAccuracy": round(sum(d["cc"] for d in days) / n * 100, 1),
        "gapAccuracy": round(sum(d["gap"] for d in days) / n * 100, 1),
        "highConfN": hc_n,
        "highConfAccuracy": round(sum(d["hc"] for d in days) / hc_n * 100, 1) if hc_n else None,
        "marketN": len(mk),
        "marketAccuracy": round(sum((d["mp"] >= 50) == d["mUp"] for d in mk) / len(mk) * 100, 1) if mk else None,
    }


def is_final_for_today(fc: dict | None, now: datetime | None = None) -> bool:
    """今日の取引日向けの確定予報がすでにあるか（あればライト更新で作り直さない）。"""
    now = now or datetime.now(JST)
    return bool(fc) and not fc.get("provisional") and fc.get("targetDate") == str(_target_date(now).date())


def build_overnight_forecast(universe: list[tuple[str, str]], now: datetime | None = None) -> dict | None:
    if yf is None:
        print("  [警告] yfinance/scikit-learn未導入のため翌日予報をスキップ")
        return None
    now = now or datetime.now(JST)
    names = dict(universe)
    tickers = list(names)
    px = yf.download(tickers + [MKT, N225, SPX, SOX, FX, NKD], period="5y", interval="1d",
                     auto_adjust=True, group_by="column", threads=True, progress=False)
    if px is None or px.empty:
        return None
    target = _target_date(now)
    panel, inputs = build_panel(px, tickers, target)
    if inputs["nkdPct"] is None or inputs["spxPct"] is None:
        print("  [警告] 夜間の海外データが取得できないため翌日予報を出しません")
        return None

    validation = _validate(panel)
    latest = panel[panel.date == panel.date.max()]
    latest = latest[latest[FEATURES].notna().all(axis=1)]
    probs = {}
    for y in ("y_cc", "y_gap"):
        d = _labeled(panel, y)
        m = _model().fit(d[FEATURES], d.label)
        probs[y] = m.predict_proba(latest[FEATURES])[:, 1]

    stocks = []
    for (_, row), pc, pg in zip(latest.iterrows(), probs["y_cc"], probs["y_gap"]):
        stocks.append({
            "ticker": row.tk, "name": names.get(row.tk, row.tk), "close": _round(row.close, 1),
            "pUp": round(pc * 100), "pGapUp": round(pg * 100),
            "expGapPct": _pct(row.pred_gap),
            "hitRate": _band_hit(validation["y_cc"]["bands"], pc),
            "gapHitRate": _band_hit(validation["y_gap"]["bands"], pg),
        })
    stocks.sort(key=lambda s: -abs(s["pUp"] - 50))
    mkt_p = round(float(np.mean(probs["y_cc"])) * 100, 1)

    history = {}
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, encoding="utf-8") as f:
            history = json.load(f)
    base = str(panel.date.max().date())
    provisional = not (target.date() == now.date() and (6, 0) <= (now.hour, now.minute) < (15, 30))
    history = _update_history(history, panel, None if provisional else {
        "b": base, "d": str(target.date()), "mkt": mkt_p,
        "preds": {s["ticker"]: [s["pUp"], s["pGapUp"]] for s in stocks},
    })
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, separators=(",", ":"))

    return {
        "generatedAt": now.strftime("%Y-%m-%d %H:%M"),
        "baseDate": base,
        "targetDate": str(target.date()),
        # 米国の取引終了(JST 5〜6時)〜東京の大引け前に作った予報だけが「確定」。夜間の手動実行などは暫定。
        "provisional": provisional,
        "beforeOpen": not provisional and now.hour < 9,
        "inputs": inputs,
        "market": {"pUp": mkt_p},
        "stocks": stocks,
        "validation": validation,
        "live": _live_stats(history),
        "basis": "夜間に確定した米S&P500・米半導体指数(SOX)・ドル円・CME日経先物の動きと、銘柄ごとの過去の感応度を"
                 "機械学習（勾配ブースティング）で組み合わせた確率。的中率はすべて、その時点より前のデータだけで"
                 "学習した予測の答え合わせ（先読みなし）。寄付き後の値動きや数日先は予測力が確認できないため対象外。",
    }
