"""売買シグナルを判定する。

スコアはユニバース内の相対評価（0〜1に正規化した合成値）のため、絶対値の閾値では
「買い」がほとんど出ない。そこで「ユニバース内の順位（上位何%か）」と
テクニカル（RSI・勢い）を組み合わせて、意味のある売買サインを出す。

判定の考え方（初心者にも説明できる根拠）:
  - 売り: 買われすぎ（RSIが非常に高い）
  - 様子見: 過熱気味（RSIが高め）→ 高値づかみを避ける
  - 強い買い: 総合評価が上位15%かつ勢いがプラス、過熱していない
  - 買い: 総合評価が上位35%で過熱していない
  - ホールド/様子見: それ以外や下位

さらに市場全体が下落基調(bear)のときは買い系を1段階格下げする。
"""

RSI_SELL = 75      # これ以上は「売り」（買われすぎ）
RSI_HOT = 68       # これ以上は過熱気味 → 買いを見送り「様子見」
RSI_STRONG_MAX = 62  # 強い買いはこのRSI未満（過熱していない）
TOP_STRONG = 0.15  # 上位15%
TOP_BUY = 0.35     # 上位35%
BOTTOM = 0.85      # 下位15%


def judge_signal(stock: dict, pct_rank: float, regime: str = "unknown") -> dict:
    """pct_rank: ユニバース内の相対順位（0=最上位, 1=最下位）。"""
    rsi = stock.get("rsi", 50) or 50
    mom = stock.get("momentum_raw", 0) or 0

    if rsi >= RSI_SELL:
        signal, conf = "売り", "中"
    elif rsi >= RSI_HOT:
        signal, conf = "様子見", "低"      # 過熱気味は追わない
    elif pct_rank <= TOP_STRONG and mom > 0:
        signal, conf = "強い買い", "高"
    elif pct_rank <= TOP_BUY:
        signal, conf = "買い", "中"
    elif pct_rank >= BOTTOM and mom < 0:
        signal, conf = "様子見", "低"
    else:
        signal, conf = "ホールド", "低"

    # 市場全体が下落基調なら買い系を1段階慎重に
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
    n = len(stocks)
    if n == 0:
        return stocks
    for s in stocks:
        rank = s.get("rank", 1)
        pct_rank = (rank - 1) / max(1, n - 1)
        judge_signal(s, pct_rank, regime)
    return stocks
