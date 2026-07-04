"""スコア・RSIから売買シグナルを判定する。"""


def judge_signal(stock: dict) -> dict:
    score = stock["score"]
    rsi = stock["rsi"]

    if score >= 0.7 and rsi < 50:
        signal, conf = "🟢 強い買い", "高"
    elif score >= 0.6:
        signal, conf = "🔵 買い", "中"
    elif score >= 0.45:
        signal, conf = "⚪ ホールド", "低"
    elif rsi > 70:
        signal, conf = "🔴 売り", "中"
    else:
        signal, conf = "⚪ 様子見", "低"

    stock["signal"] = signal
    stock["conf"] = conf
    return stock


def judge_all(stocks: list[dict]) -> list[dict]:
    return [judge_signal(s) for s in stocks]
