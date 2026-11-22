"""売買できる作戦の検証（AIの仮想運用を始める前に、売買コスト込みで勝てるかを調べる）。

予報が出るのは朝6:30。前日の終値ではもう買えず、実際に売買できるのは9:00の寄付き以降。
当たっている「前日終値→寄付き」の値動きはそのままでは取れないので、寄付き以降で取れる作戦を比べる。
  E: 画面の「上がる確率が高い銘柄」（終値の予報の上位）を寄付きで買い、大引けで売る
  A: 寄付き前の情報だけで「寄付き→大引け」に上がる銘柄をAIが選ぶ（寄付き成行で実行できる）
  B: 寄付きの値段（想定ギャップとのズレ）も使う。寄付きの値段で買える前提なので楽観的（効果があれば1時間足で再検証）
  C: ルールのみ。想定より大きく安く始まった銘柄を買う（逆張り）／高く始まった銘柄を買う（順張り）
すべて直近8四半期のウォークフォワード（先読みなし）。毎日 TOP 銘柄を同額で買い、往復の売買コストを差し引く。

使い方: python research/strategy.py
"""
import os
import pickle
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import forecast.overnight as ov  # noqa: E402
from data.universe import MASTER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = 5
COSTS = (0.0005, 0.001, 0.002)  # 往復の売買コスト。手数料0円の証券会社でも気配の幅・滑りで0.05〜0.2%程度


def walk_forward(d: pd.DataFrame, feats: list[str], y: str, quarters: int = 8) -> pd.DataFrame:
    folds = pd.date_range(end=d.date.max(), periods=quarters + 1, freq="QS")
    out = []
    for i, f0 in enumerate(folds):
        f1 = folds[i + 1] if i + 1 < len(folds) else d.date.max() + pd.Timedelta(days=1)
        tr = d[(d.date < f0 - pd.Timedelta(days=3)) & d[y].notna() & (d[y] != 0)]
        te = d[(d.date >= f0) & (d.date < f1)]
        if len(te) == 0 or len(tr) < 20000:
            continue
        m = ov._model().fit(tr[feats], (tr[y] > 0).astype(int))
        out.append(te.assign(p=m.predict_proba(te[feats])[:, 1]))
    return pd.concat(out)


def report(name: str, P: pd.DataFrame, score: str, ascending: bool = False):
    """毎日 score の上位 TOP 銘柄を寄付きで同額買い→大引けで売った場合の成績（全銘柄を同額で買う場合と比較）。"""
    P = P[P.y_oc.notna()]
    picks = P.sort_values(score, ascending=ascending).groupby("date").head(TOP)
    gross = picks.groupby("date").y_oc.mean()
    allw = P.groupby("date").y_oc.mean().reindex(gross.index)
    diff = (gross - allw) * 100
    print(f"{name}  {len(gross)}日  コスト前 平均{gross.mean() * 100:+.3f}%/日（全銘柄{allw.mean() * 100:+.3f}%）"
          f" 全銘柄との差{diff.mean():+.3f}±{diff.std(ddof=1) / np.sqrt(len(diff)):.3f}%", flush=True)
    for c in COSTS:
        net = gross - c
        cum = net.cumsum()
        print(f"     コスト{c * 100:.2f}%: 平均{net.mean() * 100:+.3f}%/日 勝ち日{(net > 0).mean() * 100:.0f}% "
              f"年率シャープ{net.mean() / net.std(ddof=1) * np.sqrt(250):+.2f} 2年累計{net.sum() * 100:+.1f}% "
              f"最大下落{(cum - cum.cummax()).min() * 100:.1f}%", flush=True)


def main():
    path = os.path.join(HERE, "cache", "px_5y.pkl")
    px = pickle.load(open(path, "rb")) if os.path.exists(path) else ov.download([t for t, _ in MASTER])
    panel, _ = ov.build_panel(px, [t for t, _ in MASTER], ov._target_date(datetime.now(ov.JST)))
    panel["y_oc"] = (1 + panel.y_cc) / (1 + panel.y_gap) - 1  # 翌営業日の寄付き→大引け
    panel["gap_act"] = panel.y_gap  # 寄付きで判明する実際のギャップ
    panel["surprise"] = panel.y_gap - panel.pred_ridge_adr  # 想定ギャップとのズレ
    panel["surprise_z"] = panel.surprise / panel.vol20
    print(f"panel {panel.shape} {panel.date.min().date()}〜{panel.date.max().date()}", flush=True)

    E = walk_forward(panel, ov.FEATURES, "y_cc")
    report("E 画面の終値予報の上位を寄付きで買う", E, "p")
    A = walk_forward(panel, ov.FEATURES, "y_oc")
    acc = ((A.p > .5) == (A.y_oc > 0))[A.y_oc.notna() & (A.y_oc != 0)].mean() * 100
    print(f"   （A の寄付き→大引けの方向的中 {acc:.1f}%）", flush=True)
    report("A 寄付き前の情報で寄付き→大引けを予報", A, "p")
    B = walk_forward(panel, ov.FEATURES + ["gap_act", "surprise", "surprise_z"], "y_oc")
    acc = ((B.p > .5) == (B.y_oc > 0))[B.y_oc.notna() & (B.y_oc != 0)].mean() * 100
    print(f"   （B の寄付き→大引けの方向的中 {acc:.1f}%）", flush=True)
    report("B 寄付きの値段も使う（楽観的）", B, "p")
    C = A[A.surprise_z.notna()]
    report("C1 想定より安く始まった銘柄を買う（逆張り）", C, "surprise_z", ascending=True)
    report("C2 想定より高く始まった銘柄を買う（順張り）", C, "surprise_z")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ── 現実的な検証: 寄付きの値段を見てから買う（1時間足で10:00、5分足で9:05〜10:00に買う） ──
def intraday_entry(tickers: list[str], interval: str, period: str) -> pd.DataFrame:
    """各日・各銘柄の、時刻ごとの値段（足の終値＝その足の終わりの時刻の値段）。列は (時刻, 銘柄)。"""
    path = os.path.join(HERE, "cache", f"px_{interval}.pkl")
    if os.path.exists(path) and (datetime.now().timestamp() - os.path.getmtime(path)) < 86400:
        return pickle.load(open(path, "rb"))
    import yfinance as yf
    d = yf.download(tickers, period=period, interval=interval, auto_adjust=True, group_by="column", progress=False,
                    threads=True)["Close"]
    d.index = d.index.tz_convert("Asia/Tokyo")
    pickle.dump(d, open(path, "wb"))
    return d


def realistic(panel: pd.DataFrame):
    """寄付きの値段を見てから買う現実的な場合。買い・売りとも同じ分足の値段（配当の調整の有無がそろう）を使う。
    買う値段 = 時刻tに終わる足の終値、売る値段 = その日の最後の足の終値（大引け）。"""
    tickers = sorted(panel.tk.unique())
    days = pd.Series(sorted(panel.date.unique()))
    nxt = pd.Series(days.shift(-1).values, index=days.values)  # 行tの答えは翌営業日
    C = panel[panel.surprise_z.notna()].copy()
    C["d1"] = C.date.map(nxt)
    for interval, period, times in (("60m", "730d", ["10:00"]), ("5m", "60d", ["09:05", "09:15", "09:30", "10:00"])):
        bars = intraday_entry(tickers, interval, period)
        step = pd.Timedelta(minutes=60 if interval == "60m" else 5)
        tod = bars.index - bars.index.normalize()
        day = bars.index.tz_localize(None).normalize()
        last = bars.groupby(day).last()  # その日の最後の足の終値＝大引け
        for t in times:
            h, m = map(int, t.split(":"))
            sel = bars[tod == pd.Timedelta(hours=h, minutes=m) - step]
            if sel.empty:
                print(f"{interval} {t}: 足が見つからない", flush=True)
                continue
            sel.index = sel.index.tz_localize(None).normalize()
            ret = (last.reindex(sel.index) / sel - 1).stack().rename("y_oc").reset_index()
            ret.columns = ["d1", "tk", "y_oc"]
            D = C.drop(columns=["y_oc"]).merge(ret, on=["d1", "tk"], how="inner")
            D = D[D.y_oc.abs() < 0.3]
            print(f"--- {interval}: {t}に買って大引けで売る（{D.d1.nunique()}日） ---", flush=True)
            report(f"C1@{t} 想定より安く始まった銘柄を買う", D, "surprise_z", ascending=True)
            report(f"C2@{t} 想定より高く始まった銘柄を買う", D, "surprise_z")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "realistic":
    path = os.path.join(HERE, "cache", "px_5y.pkl")
    px = pickle.load(open(path, "rb"))
    panel, _ = ov.build_panel(px, [t for t, _ in MASTER], ov._target_date(datetime.now(ov.JST)))
    panel["y_oc"] = (1 + panel.y_cc) / (1 + panel.y_gap) - 1
    panel["surprise"] = panel.y_gap - panel.pred_ridge_adr
    panel["surprise_z"] = panel.surprise / panel.vol20
    # 比較用: 日足の寄付き（9:00ちょうど）で買えた場合（楽観的）を、分足と同じ期間で
    realistic(panel)


def e_check(quarters: int = 14):
    """作戦E（画面の「上がる確率が高い銘柄」を寄付き成行で買い、引け成行で売る）の頑健性。
    寄付き・大引けとも板寄せ（オークション）で約定するので、検証の値段＝実際に約定する値段。手数料0円の証券会社を想定し、
    コストは0〜0.1%で見る。古い期間（候補選びに使っていない）だけの成績も出す。"""
    global COSTS
    COSTS = (0.0, 0.0005, 0.001)
    path = os.path.join(HERE, "cache", "px_5y.pkl")
    px = pickle.load(open(path, "rb"))
    panel, _ = ov.build_panel(px, [t for t, _ in MASTER], ov._target_date(datetime.now(ov.JST)))
    panel["y_oc"] = (1 + panel.y_cc) / (1 + panel.y_gap) - 1
    cc = walk_forward(panel, ov.FEATURES, "y_cc", quarters)
    gp = walk_forward(panel, ov.FEATURES, "y_gap", quarters)[["date", "tk", "p"]].rename(columns={"p": "pg"})
    P = cc.merge(gp, on=["date", "tk"])
    old_end = pd.date_range(end=panel.date.max(), periods=9, freq="QS")[0]  # 直近8四半期の始まり
    app = P[(P.p >= .6) & (P.pg > .5)]  # 画面の条件（確率60%以上・寄付きと同じ向き）
    for name, D, top in (("E-画面と同じ条件・上位5", app, 5), ("E-上位3（条件なし）", P, 3), ("E-上位5（条件なし）", P, 5),
                         ("E-上位10（条件なし）", P, 10)):
        global TOP
        TOP = top
        for period, sub in (("全期間", D), ("古い期間だけ", D[D.date < old_end]), ("直近8四半期", D[D.date >= old_end])):
            print(f"[{period}] ", end="")
            report(name, sub.assign(y_oc=sub.y_oc), "p")
    traded = app.groupby("date").size()
    print(f"画面の条件を満たす銘柄があった日: {len(traded)}/{P.date.nunique()}日、1日平均{traded.mean():.1f}銘柄", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "e_check":
    e_check()
