"""スコア・RSI・市場地合いから売買シグナルを判定する。

スコアはユニバース内の相対評価のため、地合いが悪い時でも高スコア銘柄は必ず存在する。
下落基調(bear)の相場では買い系シグナルを1段階格下げし、初心者に「強い買い」を出さない。
"""


def judge_signal(stock: dict, regime: str = "unknown") -> dict:
    score = stock["score"]
    rsi = stock["rsi"]

    if score >= 0.7 and rsi < 50:
        signal, conf = "強い買い", "高"
    elif score >= 0.6:
        signal, conf = "買い", "中"
    elif score >= 0.45:
        signal, conf = "ホールド", "低"
    elif rsi > 70:
        signal, conf = "売り", "中"
    else:
        signal, conf = "様子見", "低"

    # 地合いが下落基調の場合は買い系シグナルを格下げ（相対スコアの過信防止）
    if regime == "bear":
        if signal == "強い買い":
            signal, conf = "買い", "中"
            stock["regime_adjusted"] = True
        elif signal == "買い":
            signal, conf = "ホールド", "低"
            stock["regime_adjusted"] = True

    stock["signal"] = signal
    stock["conf"] = conf
    return stock


def judge_all(stocks: list[dict], regime: str = "unknown") -> list[dict]:
    return [judge_signal(s, regime) for s in stocks]
