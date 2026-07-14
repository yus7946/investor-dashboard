"""分析結果をHTMLダッシュボード用JSON (output/dashboard_data.json) に出力する。"""
import json
import os
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))


def export_dashboard_json(
    stocks: list[dict],
    alerts: list[dict],
    flow: dict | None,
    theme_trends: list[dict],
    backtest: dict,
    market: dict | None = None,
    market_outlook: dict | None = None,
    accuracy: dict | None = None,
    stock_master: list[dict] | None = None,
    fetched_count: int | None = None,
    universe_total: int | None = None,
    output_path: str = "output/dashboard_data.json",
) -> str:
    out_stocks = []
    for s in stocks[:10]:
        out_stocks.append({
            "rank": s["rank"],
            "ticker": s["ticker"],
            "name": s["name"],
            "score": s["score"],
            "f": s["f"],
            "per": round(s["per"], 1) if s.get("per") else None,
            "pbr": round(s["pbr"], 2) if s.get("pbr") else None,
            "roe": round(s.get("roe", 0), 1),
            "value": s["value"],
            "quality": s["quality"],
            "momentum": s["momentum"],
            "signal": s["signal"],
            "rsi": s["rsi"],
            "conf": s["conf"],
            "strategy": s["strategy"],
            "dividend_yield": round(s.get("dividend_yield", 0), 1),
            "volatility": round(s["volatility"], 4) if s.get("volatility") else None,
            "short_ratio": s.get("short_ratio"),
            "regime_adjusted": s.get("regime_adjusted", False),
            "forecast": s.get("forecast"),
            "yutai": s.get("yutai"),
            "news": s.get("news", []),
        })

    now_jst = datetime.now(JST)
    data = {
        # 表示用文字列は必ずJSTで統一する（GitHub ActionsのランナーはUTCで動くため、
        # タイムゾーン変換せずに表示すると「経過時間」の計算結果と9時間ズレて見える）。
        "updated": now_jst.strftime("%Y年%m月%d日 %H:%M"),
        "updatedAtMs": int(now_jst.timestamp() * 1000),
        "fetched": fetched_count,
        "universeTotal": universe_total,
        "market": market,
        "marketOutlook": market_outlook,
        "accuracy": accuracy,
        "backtest": backtest,
        "stocks": out_stocks,
        "stockMaster": stock_master or [],
        "alerts": alerts,
        "flow": flow,
        "themeTrends": theme_trends,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"dashboard JSON exported: {output_path}")
    return output_path
