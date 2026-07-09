"""yfinanceで株価・財務指標・配当利回りを取得する。

取得に失敗した銘柄は表示対象から除外する（架空データでの補完は行わない）。
"""
from datetime import datetime, timezone

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

        # 配当利回り: yfinanceはバージョンにより比率(0.023)と百分率(2.3)の両方があり得る。
        # 比率表記なら%へ変換し、二重変換で異常値(>20%)になった場合は戻す。
        dy = float(info.get("dividendYield") or 0)
        if 0 < dy < 1:
            dy *= 100
        if dy > 20:
            dy /= 100

        # 一株配当（年間予想）と権利確定日
        div_per_share = info.get("dividendRate")
        if not div_per_share and dy and price:
            div_per_share = round(price * dy / 100, 1)  # 利回りからの推定
        ex_div = info.get("exDividendDate")
        ex_div_date = None
        if ex_div:
            try:
                ex_div_date = datetime.fromtimestamp(int(ex_div), tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                ex_div_date = None

        return {
            "ticker": ticker,
            "name": name,
            "price": price,
            "per": info.get("trailingPE"),
            "pbr": info.get("priceToBook"),
            "roe": (info.get("returnOnEquity") or 0) * 100,
            "dividend_yield": dy,
            "div_per_share": div_per_share,
            "ex_div_date": ex_div_date,
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
    failed = []
    for ticker, name in universe:
        d = fetch_stock_data(ticker, name)
        if d:
            results.append(d)
        else:
            failed.append(ticker)
    if failed:
        # 架空データで補完せず、取得できた銘柄のみを分析対象とする
        print(f"  [情報] {len(failed)}件は取得失敗のため表示対象から除外: {', '.join(failed[:10])}{' ほか' if len(failed) > 10 else ''}")
    return results
