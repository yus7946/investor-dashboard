"""相場環境レポート: イベント（選挙・中銀会合など）前後の各市場の動き、相場の荒れ具合、
荒れ相場での予報の信頼度、イベント後の予報実績をまとめる。

イベントは data/events.json に1件追加するだけで分析対象になる（showUntil を過ぎたら非表示）。
すべてYahoo Financeの実測値から計算し、推測値では埋めない。
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from forecast.overnight import _clean_spikes

try:
    import yfinance as yf
except ImportError:
    yf = None

JST = timezone(timedelta(hours=9))
EVENTS_PATH = "data/events.json"
HISTORY_PATH = "output/overnight_history.json"
ASSETS = [
    ("^GSPC", "米S&P500", "us"), ("^IXIC", "米ナスダック", "us"), ("^SOX", "米半導体(SOX)", "us"),
    ("^RUT", "米小型株(ラッセル2000)", "us"), ("XLF", "米金融", "sector"), ("XLE", "米エネルギー", "sector"),
    ("XLV", "米ヘルスケア", "sector"), ("XLI", "米資本財", "sector"), ("XLK", "米テクノロジー", "sector"),
    ("ITA", "米防衛・航空", "sector"), ("^VIX", "VIX（恐怖指数）", "level"), ("^TNX", "米10年金利", "level"),
    ("JPY=X", "ドル円", "fx"), ("^N225", "日経平均", "jp"), ("1306.T", "TOPIX連動ETF", "jp"),
]


def _active_event(now: datetime) -> dict | None:
    if not os.path.exists(EVENTS_PATH):
        return None
    with open(EVENTS_PATH, encoding="utf-8") as f:
        events = json.load(f)
    today = now.strftime("%Y-%m-%d")
    live = [e for e in events if e["date"] <= today <= e.get("showUntil", "9999")]
    return max(live, key=lambda e: e["date"]) if live else None


def _move(s: pd.Series, base_before: pd.Timestamp, kind: str) -> dict | None:
    """base_before より前の最終値を基準に、反応日（基準の翌取引日）・累計・直近1日の変化。"""
    s = s.dropna()
    pre = s[s.index < base_before]
    post = s[s.index >= base_before]
    if pre.empty or post.empty or len(s) < 2:
        return None
    b, react, last = pre.iloc[-1], post.iloc[0], s.iloc[-1]
    if kind == "level":  # VIX・金利は水準の差（ポイント）で見る
        return {"base": round(float(b), 2), "last": round(float(last), 2), "react": round(float(react - b), 2),
                "since": round(float(last - b), 2), "day": round(float(s.iloc[-1] - s.iloc[-2]), 2), "unit": "pt",
                "lastDate": str(s.index[-1].date())}
    return {"base": round(float(b), 2), "last": round(float(last), 2), "react": round(float(react / b - 1) * 100, 2),
            "since": round(float(last / b - 1) * 100, 2), "day": round(float(s.iloc[-1] / s.iloc[-2] - 1) * 100, 2),
            "unit": "%", "lastDate": str(s.index[-1].date())}


def _percentile(s: pd.Series, v: float) -> float:
    return round(float((s.dropna() >= v).mean()) * 100)  # 上位何%か


def build_market_report(master: list[tuple[str, str]], overnight: dict | None = None,
                        now: datetime | None = None) -> dict | None:
    if yf is None:
        return None
    now = now or datetime.now(JST)
    event = _active_event(now)
    names = dict(master)
    syms = [a[0] for a in ASSETS]
    px = yf.download(syms + list(names), period="6mo", interval="1d", auto_adjust=True, progress=False)
    if px is None or px.empty:
        return None
    C = _clean_spikes(px["Close"])
    # 当日の東京が引ける前（16:00まで）は、東京の当日分を未確定として使わない
    if (now.hour, now.minute) < (16, 0):
        jp_cols = [c for c in C.columns if str(c).endswith(".T") or c == "^N225"]
        C.loc[C.index >= pd.Timestamp(now.date()), jp_cols] = np.nan

    # 相場の荒れ具合（過去5年の中での位置）
    long = yf.download(["^N225", "^VIX"], period="5y", auto_adjust=True, progress=False)["Close"]
    n225 = _clean_spikes(long[["^N225"]])["^N225"].dropna()
    rv = n225.pct_change().rolling(20).std() * np.sqrt(250) * 100
    vix = long["^VIX"].dropna()
    hi3m = n225.tail(63).max()
    regime = {
        "n225Vol": round(float(rv.iloc[-1]), 1), "n225VolTopPct": _percentile(rv, rv.iloc[-1]),
        "vix": round(float(vix.iloc[-1]), 1), "vixTopPct": _percentile(vix, vix.iloc[-1]), "vixDate": str(vix.index[-1].date()),
        "n225FromHigh": round(float(n225.iloc[-1] / hi3m - 1) * 100, 1),
        "n225Daily10": [round(float(x) * 100, 1) for x in n225.pct_change().tail(10)],
    }

    out = {"generatedAt": now.strftime("%Y-%m-%d %H:%M"), "regime": regime, "event": None}

    # 今の状況に近い過去の日の予報的中率（翌日予報の検証結果から抜き出す）
    if overnight and overnight.get("validation", {}).get("regimes"):
        reg = overnight["validation"]["regimes"]
        nkd = abs((overnight["inputs"].get("nkdPct") or 0) / 100)
        pick = {}
        for key, val in (("nkd", nkd), ("vixlvl", regime["vix"])):
            for r in reg.get(key, {}).get("rows", []):
                if r["lo"] <= val < r["hi"]:
                    pick[key] = {**r, "title": reg[key]["title"], "value": round(val * 100, 2) if key == "nkd" else val}
        out["reliability"] = {"current": pick, "table": reg}

    if event:
        react_day = pd.Timestamp(event["date"]) + pd.Timedelta(days=1)  # 結果が出た後の最初の取引が反応日
        assets = []
        for sym, label, kind in ASSETS:
            if sym in C.columns:
                m = _move(C[sym], react_day, "level" if kind == "level" else "pct")
                if m:
                    assets.append({"symbol": sym, "label": label, "group": kind, **m})
        moves = {}
        for t in names:
            if t in C.columns:
                m = _move(C[t], react_day, "pct")
                if m:
                    moves[t] = [m["react"], m["since"]]
        ranked = sorted(moves.items(), key=lambda kv: kv[1][1])
        track = []
        if os.path.exists(HISTORY_PATH):
            with open(HISTORY_PATH, encoding="utf-8") as f:
                days = json.load(f).get("days", [])
            for d in days:
                if d["d"] > event["date"] and not d.get("x"):
                    track.append({"d": d["d"], "cc": round(d["cc"] / d["n"] * 100, 1),
                                  "gap": round(d["gap"] / max(1, d.get("gn", d["n"])) * 100, 1),
                                  "agree": round(d["ag"] / d["agN"] * 100, 1) if d.get("agN") else None,
                                  "mkt": d.get("mp"), "mktUp": d.get("mUp")})
        out["event"] = {
            **{k: event.get(k) for k in ("id", "name", "date", "summary", "sources", "history")},
            "assets": assets, "moves": moves,
            "topUp": [{"ticker": t, "name": names[t], "since": v[1]} for t, v in ranked[::-1][:5]],
            "topDown": [{"ticker": t, "name": names[t], "since": v[1]} for t, v in ranked[:5]],
            "forecastTrack": track,
        }
    return out
