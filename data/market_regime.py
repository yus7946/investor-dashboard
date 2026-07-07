"""市場全体の地合い判定。

TOPIX連動ETF(1306.T)の日足と200日移動平均から、市場環境を bull / neutral / bear に分類する。
下落基調の相場では個別銘柄の相対スコアが高くても「強い買い」を出さないための安全装置。
データが取得できない場合は None を返し、シグナルは無調整のまま「地合い不明」と表示する。
"""

try:
    import yfinance as yf
except ImportError:
    yf = None

BENCHMARK = "1306.T"


def fetch_market_regime() -> dict | None:
    if yf is None:
        return None
    try:
        hist = yf.Ticker(BENCHMARK).history(period="2y", interval="1d", auto_adjust=True)["Close"]
        hist = hist.dropna()
        if len(hist) < 220:
            return None

        ma200 = hist.rolling(200).mean()
        price = float(hist.iloc[-1])
        ma_now = float(ma200.iloc[-1])
        ma_month_ago = float(ma200.iloc[-21])  # 約1ヶ月前の200日線

        above = price > ma_now
        rising = ma_now > ma_month_ago

        if above and rising:
            regime, label = "bull", "上昇基調"
            desc = "市場全体（TOPIX）は200日移動平均の上で推移しており、比較的良好な地合いです。"
        elif not above and not rising:
            regime, label = "bear", "下落基調"
            desc = "市場全体（TOPIX）が200日移動平均を下回り、平均線も下向きです。買いシグナルは通常より慎重に扱ってください。"
        else:
            regime, label = "neutral", "中立"
            desc = "市場全体（TOPIX）は方向感の乏しい局面です。"

        return {
            "regime": regime,
            "label": label,
            "desc": desc,
            "priceVsMa200Pct": round((price / ma_now - 1) * 100, 1),
            "source": "TOPIX連動ETF(1306.T)終値と200日移動平均から判定",
        }
    except Exception:
        return None
