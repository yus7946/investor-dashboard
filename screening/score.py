"""バリュー・クオリティ・モメンタムの3因子スコアリングとF-Score、戦略タグ付け。"""


def _normalize(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def score_universe(stocks: list[dict]) -> list[dict]:
    if not stocks:
        return []

    pers = [1 / (s["per"] or 999) for s in stocks]
    pbrs = [1 / (s["pbr"] or 999) for s in stocks]
    roes = [s["roe"] or 0 for s in stocks]
    moms = [s["momentum_raw"] or 0 for s in stocks]
    vols = [1 / (1 + (s["volatility"] or 0)) for s in stocks]

    value_n = _normalize([(a + b) / 2 for a, b in zip(pers, pbrs)])
    quality_n = _normalize(roes)
    momentum_n = _normalize(moms)
    lowvol_n = _normalize(vols)

    for i, s in enumerate(stocks):
        s["value"] = round(value_n[i], 3)
        s["quality"] = round(quality_n[i], 3)
        s["momentum"] = round(momentum_n[i], 3)
        # 勢い(momentum)を最大比重に。財務指標(value/quality)は決算ごとにしか変わらず日々ほぼ不変のため、
        # これらを主にするとTOP10が固定化する。日々の値動きを主軸にしつつ、割安・品質も50%残してバランスを取る。
        s["score"] = round(
            value_n[i] * 0.25 + quality_n[i] * 0.25 + momentum_n[i] * 0.35 + lowvol_n[i] * 0.15, 3
        )

        f = 0
        if (s["roe"] or 0) > 8:
            f += 1
        if (s["per"] or 999) < 25:
            f += 1
        if (s["pbr"] or 999) < 3:
            f += 1
        if (s["momentum_raw"] or 0) > 0:
            f += 1
        if (s["dividend_yield"] or 0) > 1:
            f += 1
        f += min(4, round(quality_n[i] * 4))
        s["f"] = min(9, f)

        strategy = []
        if s["momentum_raw"] and s["momentum_raw"] > 0.05 and s["rsi"] < 65:
            strategy.append("swing")
        if (s["dividend_yield"] or 0) >= 2.5:
            strategy.append("dividend")
        if s["price"] and s["price"] * 100 <= 50000:
            strategy.append("low_price")
        if (s["market_cap"] or 0) < 3e11 and (s["momentum_raw"] or 0) > 0.15:
            strategy.append("growth")
        s["strategy"] = strategy or ["swing"]

    stocks.sort(key=lambda s: s["score"], reverse=True)
    for i, s in enumerate(stocks, 1):
        s["rank"] = i

    n = len(stocks)
    cutoff_out = max(1, int(n * 0.15))
    for i, s in enumerate(stocks):
        s["rotation_out"] = i >= n - cutoff_out
        s["rotation_in"] = (s["score"] >= 0.65) and ((s["momentum_raw"] or 0) > 0.08)

    return stocks
