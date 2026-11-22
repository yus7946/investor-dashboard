"""毎週の点検（PDCAのCheck）。運用記録と画面のデータから、点検に必要な集計だけを短く出す。

毎朝の自動更新が output/overnight_history.json に残す記録（的中・作成時刻 at・確定データでの再現 rp・注意 fl）を
1週間分まとめ、修正後（--since 以降）の累計を検証時の成績と比べる。読むのはこの出力だけでよい（トークン節約）。

使い方: python research/weekly_check.py [--since 2026-11-24] [--days 7]
"""
import argparse
import json
import os
import re
from collections import Counter
from datetime import date, timedelta

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pct(a, b):
    return round(a / b * 100, 1) if b else None


def summarize(ds):
    """集計外（x）を除いた日の成績。日ごとの的中率のばらつきから標準誤差も出す（同じ日の銘柄は独立でないため）。"""
    ds = [d for d in ds if not d.get("x")]
    if not ds:
        return None
    daily = np.array([d["cc"] / d["n"] * 100 for d in ds])
    mk = [d for d in ds if d.get("mp") is not None]
    rps = [d["rp"] for d in ds if d.get("rp") and d["rp"].get("cc") is not None]
    return {
        "日数": len(ds),
        "終値": pct(sum(d["cc"] for d in ds), sum(d["n"] for d in ds)),
        "終値の日次SE": round(float(daily.std(ddof=1) / np.sqrt(len(daily))), 1) if len(daily) > 1 else None,
        "寄付き": pct(sum(d["gap"] for d in ds), sum(d.get("gn", d["n"]) for d in ds)),
        "同じ向きの終値": pct(sum(d.get("ag", 0) for d in ds), sum(d.get("agN", 0) for d in ds)),
        "高確信": pct(sum(d["hc"] for d in ds), sum(d["hcN"] for d in ds)),
        "市場全体": f"{sum((d['mp'] >= 50) == d['mUp'] for d in mk)}/{len(mk)}日",
        "寄付き前に作成": f"{sum(1 for d in ds if d.get('at') and d['at'] < '09:00')}/{len(ds)}日",
        "再現(終値/寄付き)": f"{np.mean([r['cc'] for r in rps]):.1f}/{np.mean([r['gap'] for r in rps]):.1f}（{len(rps)}日）" if rps else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-11-24", help="この日以降を「修正後」として累計する（11/21の修正の初日）")
    ap.add_argument("--days", type=int, default=7, help="今週として一覧にする暦日数")
    a = ap.parse_args()
    h = json.load(open(os.path.join(ROOT, "output/overnight_history.json"), encoding="utf-8"))
    dd = json.load(open(os.path.join(ROOT, "output/dashboard_data.json"), encoding="utf-8"))
    o = dd.get("overnight") or {}
    v = o.get("validation") or {}
    days = h.get("days", [])

    print("■ 最新の予報:", o.get("generatedAt"), "対象日", o.get("targetDate"),
          "暫定" if o.get("provisional") else "確定", "寄付き前" if o.get("beforeOpen") else "寄付き後/対象外")
    print("  入力:", json.dumps(o.get("inputs"), ensure_ascii=False))
    print("■ 検証時の成績: 終値", (v.get("y_cc") or {}).get("accuracy"), "寄付き", (v.get("y_gap") or {}).get("accuracy"),
          "市場", (v.get("market") or {}).get("accuracy"))
    hl = (o.get("live") or {}).get("health") or {}
    print("■ 健康状態:", hl.get("status"), hl.get("note"), f"(直近{hl.get('days')}日 {hl.get('recentAccuracy')}% / 誤差幅±{hl.get('margin')})")

    start = str(date.today() - timedelta(days=a.days))
    wk = [d for d in days if d["d"] > start]
    print(f"■ 今週の答え合わせ（{start}より後・{len(wk)}日）")
    for d in wk:
        rp = d.get("rp") or {}
        mk = "—" if d.get("mp") is None else ("○" if (d["mp"] >= 50) == d["mUp"] else "×")
        print(f"  {d['d']} {d.get('at') or '--:--'} {d.get('m', 'v1')} 終値{pct(d['cc'], d['n'])} 寄付き{pct(d['gap'], d.get('gn', d['n']))}"
              f" 市場{mk} 再現{rp.get('cc')}/{rp.get('gap')}" + (" 集計外" if d.get("x") else ""))
        for f in d.get("fl", []):
            if not f.startswith("前日の日経平均を日経225ETFで補完"):  # ほぼ毎朝出る既知の補完は省略
                print("     ⚠", f)
    if not wk:
        print("  （記録なし。毎朝の自動更新が止まっていないか確認）")
    print("■ 今週の集計:", json.dumps(summarize(wk), ensure_ascii=False))
    post = [d for d in days if d["d"] >= a.since]
    s = summarize(post)
    print(f"■ {a.since}以降（修正後）の累計:", json.dumps(s, ensure_ascii=False))
    if s and s["日数"] >= 20:
        print("  → 20営業日に到達。検証時との差を「日次SEの2倍」で判定し、PDCAに記録すること")
    c = Counter(re.sub(r"（.*|:.*|[+-]?\d+pt", "", f) for d in post for f in d.get("fl", []))
    print("■ 修正後の注意の件数:", dict(c) or "なし")
    # AIの仮想運用（寄付き前に出た予報の日だけ数える）
    allp = h.get("paper") or []
    pp = [p for p in allp if not p.get("late")]
    bt = (v.get("paper") or {})
    if pp:
        r = np.array([p["ret"] / 100 if p["picks"] else 0.0 for p in pp])
        traded = [p for p in pp if p["picks"]]
        eq = np.cumprod(1 + r)
        se = r[r != 0].std(ddof=1) / np.sqrt((r != 0).sum()) * 100 if (r != 0).sum() > 1 else None
        print(f"■ AIの仮想運用: {pp[0]['d']}〜 {len(pp)}営業日（売買{len(traded)}日・寄付き後で除外{len(allp) - len(pp)}日）"
              f" 累計{(eq[-1] - 1) * 100:+.2f}%（コスト0.1%なら{(np.cumprod(1 + r - 0.001 * (r != 0))[-1] - 1) * 100:+.2f}%）"
              f" 売買日の平均{r[r != 0].mean() * 100:+.3f}%" + (f"±{se:.3f}" if se else "")
              + f" 勝ち{sum(p['ret'] > 0 for p in traded)}/{len(traded)}日")
        for p in allp[-5:]:
            print(f"  {p['d']} {p.get('at')} {'除外 ' if p.get('late') else ''}{p['ret']:+.2f}% " + " ".join(f"{t.replace('.T', '')}:{x:+.1f}" for t, x in p["picks"]))
    else:
        print(f"■ AIの仮想運用: 実績なし（寄付き後で除外{len(allp)}日）")
    if bt:
        print(f"  検証（同じルール・{bt.get('days')}日）: 1日平均{bt['cost0']['avg']:+.3f}%（コスト0.1%なら{bt['cost10']['avg']:+.3f}%）"
              f" 最大下落{bt['cost0']['maxDD']}%")
    pend = h.get("pending") or {}
    print("■ 答え合わせ待ち:", pend.get("d"), pend.get("at"), pend.get("model"))


if __name__ == "__main__":
    main()
