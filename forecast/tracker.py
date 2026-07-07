"""予測の答え合わせ・的中率の蓄積・レンジ自動補正。

GitHubリポジトリに output/prediction_history.json を蓄積し、実行のたびに:
  1. 過去の未解決予測のうち「翌営業日の確定終値」が出たものを実データで照合
  2. 銘柄別（±1σレンジ的中／方向）と市場全体（米国→東京）の的中率を集計
  3. レンジ的中率が理想の約68%からずれていれば、σ幅の補正係数を実績から算出

原則:
  - JST基準で「今日より前」の確定終値のみ使用（look-aheadと未確定日中値を排除）
  - 8:30/11:30の二重実行に備え baseDate+ticker で重複記録を防止
  - 実データが無ければ記録しない・数値を捏造しない
"""
import io
import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

HISTORY_PATH = "output/prediction_history.json"
TOPIX_ETF = "1306.T"
SPX = "^GSPC"
IDEAL_HIT = 0.68          # ±1σの理想的中率
MIN_FOR_CALIB = 30        # レンジ補正を有効化する最小解決済み件数
CALIB_BOUNDS = (0.8, 1.3) # 補正係数の範囲


def _jst_today() -> "datetime.date":
    return (datetime.now(timezone.utc) + timedelta(hours=9)).date()


def load_history(path: str = HISTORY_PATH) -> dict:
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                h = json.load(f)
            h.setdefault("stockPredictions", [])
            h.setdefault("marketPredictions", [])
            return h
        except Exception:
            pass
    return {"stockPredictions": [], "marketPredictions": []}


def save_history(history: dict, path: str = HISTORY_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def calibration_factor(history: dict) -> float:
    """解決済みレンジ予測の実績的中率から、σ幅の補正係数を算出。

    的中率が理想より低ければ（レンジが狭すぎ）幅を広げ、高すぎれば狭める。
    件数が少ない間は 1.0（無補正）。
    """
    resolved = [p for p in history.get("stockPredictions", []) if p.get("inRange") is not None]
    if len(resolved) < MIN_FOR_CALIB:
        return 1.0
    hit = sum(1 for p in resolved if p["inRange"]) / len(resolved)
    if hit <= 0:
        return CALIB_BOUNDS[1]
    # 正規分布近似で「的中率hitを与える幅」と「理想68%を与える幅」の比。
    # ラフに hit と ideal の比の逆数でスケール（過補正を避け緩やかに）。
    factor = (1 - hit + 0.5) / (1 - IDEAL_HIT + 0.5)
    factor = max(CALIB_BOUNDS[0], min(CALIB_BOUNDS[1], factor))
    return round(factor, 3)


def _download_closes(tickers: list[str]) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, period="1mo", interval="1d",
                           auto_adjust=True, progress=False, threads=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data
    except Exception:
        return pd.DataFrame()


def _completed_closes(series: pd.Series, today) -> pd.Series:
    """JST今日より前の確定終値のみ（未確定の当日日中値を除外）。"""
    s = series.dropna()
    if s.empty:
        return s
    idx_dates = pd.to_datetime(s.index).date
    return s[[d < today for d in idx_dates]]


def update(history: dict, stocks: list[dict], outlook: dict | None) -> dict:
    """未解決予測を照合し、今回の新規予測を記録する。"""
    today = _jst_today()

    # 照合対象＝新規記録対象の銘柄
    tickers = sorted({s["ticker"] for s in stocks if s.get("forecast")})
    need = set(tickers) | {TOPIX_ETF, SPX}
    # 未解決の過去予測の銘柄も照合に必要
    for p in history["stockPredictions"]:
        if p.get("actualClose") is None:
            need.add(p["ticker"])
    data = _download_closes(sorted(need))

    # ── 1) 銘柄別の照合 ──
    for p in history["stockPredictions"]:
        if p.get("actualClose") is not None:
            continue
        if p["ticker"] not in data.columns:
            continue
        closes = _completed_closes(data[p["ticker"]], today)
        base_date = pd.to_datetime(p["baseDate"]).date()
        after = closes[[pd.to_datetime(i).date() > base_date for i in closes.index]]
        if after.empty:
            continue  # まだ翌営業日の終値が出ていない
        actual_date = pd.to_datetime(after.index[0]).date()
        actual_close = float(after.iloc[0])
        p["actualDate"] = str(actual_date)
        p["actualClose"] = round(actual_close, 1)
        p["actualReturn"] = round((actual_close / p["basePrice"] - 1) * 100, 2)
        p["inRange"] = bool(p["rangeLow"] <= actual_close <= p["rangeHigh"])
        p["wentUp"] = bool(actual_close > p["basePrice"])

    # ── 2) 市場全体の照合 ──
    for p in history["marketPredictions"]:
        if p.get("topixReturn") is not None:
            continue
        if TOPIX_ETF not in data.columns:
            continue
        closes = _completed_closes(data[TOPIX_ETF], today)
        base_date = pd.to_datetime(p["baseDate"]).date()
        after = closes[[pd.to_datetime(i).date() > base_date for i in closes.index]]
        prior = closes[[pd.to_datetime(i).date() <= base_date for i in closes.index]]
        if after.empty or prior.empty:
            continue
        base_close = float(prior.iloc[-1])
        nxt = float(after.iloc[0])
        p["actualDate"] = str(pd.to_datetime(after.index[0]).date())
        p["topixReturn"] = round((nxt / base_close - 1) * 100, 2)
        p["wentUp"] = bool(nxt > base_close)

    # ── 3) 今回の新規予測を記録（baseDate+tickerでdedup） ──
    existing_keys = {(p["baseDate"], p["ticker"]) for p in history["stockPredictions"]}
    for s in stocks:
        f = s.get("forecast")
        if not f:
            continue
        # 予測の基準日＝最後の確定終値の日
        if s["ticker"] not in data.columns:
            base_date = str(today - timedelta(days=1))
        else:
            cc = _completed_closes(data[s["ticker"]], today)
            base_date = str(pd.to_datetime(cc.index[-1]).date()) if not cc.empty else str(today - timedelta(days=1))
        key = (base_date, s["ticker"])
        if key in existing_keys:
            continue
        existing_keys.add(key)
        history["stockPredictions"].append({
            "baseDate": base_date,
            "ticker": s["ticker"],
            "name": s["name"],
            "basePrice": f["price"],
            "rangeLow": f["rangeLow"],
            "rangeHigh": f["rangeHigh"],
            "predUpRate": (f.get("cond") or {}).get("upRate"),
            "condN": (f.get("cond") or {}).get("n"),
            "actualDate": None, "actualClose": None,
            "actualReturn": None, "inRange": None, "wentUp": None,
        })

    if outlook:
        m_keys = {p["baseDate"] for p in history["marketPredictions"]}
        # 市場予測の基準日＝TOPIXの最後の確定終値日
        if TOPIX_ETF in data.columns:
            cc = _completed_closes(data[TOPIX_ETF], today)
            m_base = str(pd.to_datetime(cc.index[-1]).date()) if not cc.empty else str(today - timedelta(days=1))
        else:
            m_base = str(today - timedelta(days=1))
        if m_base not in m_keys:
            history["marketPredictions"].append({
                "baseDate": m_base,
                "spxUp": bool(outlook.get("spxChangePct", 0) > 0),
                "predUpRate": outlook.get("topixUpRate"),
                "n": outlook.get("n"),
                "actualDate": None, "topixReturn": None, "wentUp": None,
            })

    # 履歴の肥大化防止（直近2年ぶん程度に制限）
    history["stockPredictions"] = history["stockPredictions"][-4000:]
    history["marketPredictions"] = history["marketPredictions"][-800:]
    return history


def compute_accuracy(history: dict) -> dict:
    """蓄積した答え合わせから的中率を集計（サンプル数付き）。"""
    sp = [p for p in history["stockPredictions"] if p.get("inRange") is not None]
    mp = [p for p in history["marketPredictions"] if p.get("wentUp") is not None]

    result = {
        "resolvedStockCount": len(sp),
        "resolvedMarketCount": len(mp),
        "rangeHitRate": None,
        "leanUpActualUp": None, "leanUpN": 0,
        "leanDownActualUp": None, "leanDownN": 0,
        "marketHitRate": None,
        "sinceDate": None,
        "note": "予測と実際の答え合わせを毎営業日ぶん蓄積した実績です。件数が増えるほど信頼度が上がります。将来を保証するものではありません。",
    }

    all_dates = [p["baseDate"] for p in history["stockPredictions"]] + [p["baseDate"] for p in history["marketPredictions"]]
    if all_dates:
        result["sinceDate"] = min(all_dates)

    if sp:
        result["rangeHitRate"] = round(sum(1 for p in sp if p["inRange"]) / len(sp) * 100, 1)
        lean_up = [p for p in sp if p.get("predUpRate") is not None and p["predUpRate"] >= 55]
        lean_dn = [p for p in sp if p.get("predUpRate") is not None and p["predUpRate"] <= 45]
        if lean_up:
            result["leanUpActualUp"] = round(sum(1 for p in lean_up if p["wentUp"]) / len(lean_up) * 100, 1)
            result["leanUpN"] = len(lean_up)
        if lean_dn:
            result["leanDownActualUp"] = round(sum(1 for p in lean_dn if p["wentUp"]) / len(lean_dn) * 100, 1)
            result["leanDownN"] = len(lean_dn)

    if mp:
        result["marketHitRate"] = round(sum(1 for p in mp if p["wentUp"]) / len(mp) * 100, 1)

    return result
