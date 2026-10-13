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

import numpy as np  # noqa: E402

import forecast.overnight as ov  # noqa: E402
from data.universe import MASTER  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results.jsonl")

BASE = ov.OVERNIGHT + ov.STOCK
EXPERIMENTS = {
    "base": (BASE, None),
    "mkt": (BASE + ov.EXTRA_MKT, None),
    "sec": (BASE + ov.EXTRA_SEC, None),
    "adr": (BASE + ov.EXTRA_ADR, None),
    "sec_adr": (BASE + ov.EXTRA_SEC + ov.EXTRA_ADR, None),
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


def paired(base: dict, cand: dict, y: str) -> tuple[float, float]:
    """同じ日・同じ銘柄の予測で的中率の差をとり、日単位の標準誤差を返す（日内の銘柄は独立でないため）。"""
    m = base["_preds"][y].merge(cand["_preds"][y], on=["date", "tk", "label"], suffixes=("_b", "_c"))
    m["d"] = ((m.p_c > .5) == m.label).astype(float) - ((m.p_b > .5) == m.label).astype(float)
    daily = m.groupby("date").d.mean() * 100
    return round(float(daily.mean()), 2), round(float(daily.std(ddof=1) / np.sqrt(len(daily))), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--period", default="5y")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    px = load_px(a.period, a.refresh)
    target = ov._target_date(datetime.now(ov.JST))
    panel, _ = ov.build_panel(px, [t for t, _ in MASTER], target)
    print(f"panel rows={len(panel)} stocks={panel.tk.nunique()} last={panel.date.max().date()}")
    names = a.names or list(EXPERIMENTS)
    if "base" not in names:
        names = ["base"] + names
    base = None
    for name in names:
        feats, model_fn = EXPERIMENTS[name]
        t0 = time.time()
        v = ov._validate(panel, feats=feats, model_fn=model_fn, keep_preds=True)
        if name == "base":
            base = v
        row = {
            "at": datetime.now(ov.JST).strftime("%Y-%m-%d %H:%M"), "name": name, "period": a.period,
            "from": v["from"], "to": v["to"], "nFeat": len(feats),
            "cc": v["y_cc"]["accuracy"], "ccHC": v["y_cc"]["highConfAccuracy"], "ccHCn": v["y_cc"]["highConfN"],
            "gap": v["y_gap"]["accuracy"], "gapHC": v["y_gap"]["highConfAccuracy"], "gapHCn": v["y_gap"]["highConfN"],
            "mkt": v["market"]["accuracy"], "mktStrong": v["market"]["strongAccuracy"], "sec": round(time.time() - t0),
            "ccAUC": v["y_cc"]["auc"], "gapAUC": v["y_gap"]["auc"],
        }
        if name != "base":
            row["ccDiff"], row["ccDiffSE"] = paired(base, v, "y_cc")
            row["gapDiff"], row["gapDiffSE"] = paired(base, v, "y_gap")
        print(f"{name:8s} 終値{row['cc']}%(高確信{row['ccHC']}%/{row['ccHCn']}件) "
              f"寄付き{row['gap']}%(高確信{row['gapHC']}%/{row['gapHCn']}件) 市場{row['mkt']}%(強{row['mktStrong']}%) "
              f"AUC {row['ccAUC']}/{row['gapAUC']} {row['sec']}s"
              + (f"\n         vs base: 終値{row['ccDiff']:+}pt(±{row['ccDiffSE']}) 寄付き{row['gapDiff']:+}pt(±{row['gapDiffSE']})"
                 if "ccDiff" in row else ""), flush=True)
        with open(RESULTS, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
