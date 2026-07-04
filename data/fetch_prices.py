"""yfinanceで株価・財務指標・配当利回りを取得する。"""
import random
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def fetch_stock_data(ticker: str, name: str) -> dict | None:
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="6mo")
        if hist.empty:
            return None

        price = float(hist["Close"].iloc[-1])
        delta = hist["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if not rs.empty else 50.0
        if pd.isna(rsi):
            rsi = 50.0

        momentum = float(hist["Close"].iloc[-1] / hist["Close"].iloc[0] - 1)
        volatility = float(hist["Close"].pct_change().std() * (252 ** 0.5))

        day_change_pct = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-2] - 1) if len(hist) >= 2 else 0.0
        vol_20avg = float(hist["Volume"].iloc[-21:-1].mean()) if len(hist) >= 21 else float(hist["Volume"].mean())
        volume_ratio = float(hist["Volume"].iloc[-1] / vol_20avg) if vol_20avg > 0 else 1.0
        week_change_pct = float(hist["Close"].iloc[-1] / hist["Close"].iloc[-6] - 1) if len(hist) >= 6 else 0.0

        return {
            "ticker": ticker,
            "name": name,
            "price": price,
            "per": info.get("trailingPE"),
            "pbr": info.get("priceToBook"),
            "roe": (info.get("returnOnEquity") or 0) * 100,
            "dividend_yield": (info.get("dividendYield") or 0) * (100 if info.get("dividendYield", 0) < 1 else 1),
            "market_cap": info.get("marketCap"),
            "rsi": round(rsi, 1),
            "momentum_raw": momentum,
            "volatility": volatility,
            "day_change_pct": round(day_change_pct, 4),
            "volume_ratio": round(volume_ratio, 2),
            "week_change_pct": round(week_change_pct, 4),
        }
    except Exception:
        return None


def fetch_universe(universe: list[tuple[str, str]]) -> list[dict]:
    results = []
    for ticker, name in universe:
        d = fetch_stock_data(ticker, name)
        if d:
            results.append(d)
    if len(results) < max(5, len(universe) // 3):
        # 通信失敗が多い場合はフォールバックのサンプルデータで補完
        print(f"  [警告] 取得成功 {len(results)}/{len(universe)} 件 -> 不足分をサンプルデータで補完")
        got = {r["ticker"] for r in results}
        for ticker, name in universe:
            if ticker in got:
                continue
            results.append(_sample_stock_data(ticker, name))
    return results


def _sample_stock_data(ticker: str, name: str) -> dict:
    random.seed(ticker)
    price = round(random.uniform(800, 9000), 1)
    return {
        "ticker": ticker,
        "name": name,
        "price": price,
        "per": round(random.uniform(8, 35), 1),
        "pbr": round(random.uniform(0.6, 5.0), 2),
        "roe": round(random.uniform(4, 20), 1),
        "dividend_yield": round(random.uniform(0, 4), 2),
        "market_cap": int(price * random.uniform(5e7, 2e9)),
        "rsi": round(random.uniform(30, 70), 1),
        "momentum_raw": round(random.uniform(-0.15, 0.2), 3),
        "volatility": round(random.uniform(0.15, 0.45), 3),
        "day_change_pct": round(random.uniform(-0.05, 0.05), 4),
        "volume_ratio": round(random.uniform(0.5, 2.5), 2),
        "week_change_pct": round(random.uniform(-0.15, 0.15), 4),
        "_sample": True,
    }
