"""yfinanceのニュースを取得し、ルールベースでsentimentとthemeを付与する。"""
import random

try:
    import yfinance as yf
except ImportError:
    yf = None

POS_WORDS = ["増益", "上方修正", "好調", "最高益", "増配", "拡大", "回復", "承認", "好決算"]
NEG_WORDS = ["減益", "下方修正", "不振", "減配", "縮小", "悪化", "撤退", "減産"]

THEME_KEYWORDS = {
    "AI": ["AI", "人工知能", "生成AI"],
    "半導体": ["半導体", "チップ", "ファウンドリ"],
    "資源": ["資源", "原油", "鉱山", "金属"],
    "金利": ["金利", "利上げ", "日銀"],
    "円安": ["円安", "為替"],
    "自動車": ["自動車", "EV", "車載"],
}


def _sentiment(title: str) -> str:
    if any(w in title for w in POS_WORDS):
        return "pos"
    if any(w in title for w in NEG_WORDS):
        return "neg"
    return "neu"


def _theme(title: str) -> str:
    for theme, kws in THEME_KEYWORDS.items():
        if any(kw in title for kw in kws):
            return theme
    return "その他"


def fetch_news_for_ticker(ticker: str, name: str, limit: int = 1) -> list[dict]:
    if yf is not None:
        try:
            items = yf.Ticker(ticker).news or []
            out = []
            for it in items[:limit]:
                content = it.get("content", it)
                title = content.get("title") or it.get("title") or f"{name}関連ニュース"
                url = (content.get("canonicalUrl", {}) or {}).get("url") or it.get("link") or f"https://finance.yahoo.co.jp/quote/{ticker}"
                source = (content.get("provider", {}) or {}).get("displayName") or it.get("publisher") or "ニュース"
                out.append({
                    "title": title,
                    "source": source,
                    "url": url,
                    "sentiment": _sentiment(title),
                    "theme": _theme(title),
                })
            if out:
                return out
        except Exception:
            pass
    return _sample_news_for_ticker(ticker, name, limit)


def _sample_news_for_ticker(ticker: str, name: str, limit: int = 1) -> list[dict]:
    """yfinanceニュース取得失敗・0件時のフォールバック。"""
    random.seed(ticker + "_news_fallback")
    templates = [
        ("{name}が好調な業績を発表", "pos", "AI"),
        ("{name}、新製品の引き合い増加と発表", "pos", "半導体"),
        ("{name}の業績見通しに慎重な声", "neg", "金利"),
        ("{name}は据え置きの計画を維持と説明", "neu", "その他"),
    ]
    picked = random.sample(templates, min(limit, len(templates)))
    return [{
        "title": t.format(name=name),
        "source": "サンプルニュース（取得失敗時の代替）",
        "url": f"https://finance.yahoo.co.jp/quote/{ticker}",
        "sentiment": s,
        "theme": th,
    } for t, s, th in picked]


def build_theme_trends(all_news: list[dict], prev_trends: dict[str, int] | None = None) -> list[dict]:
    prev_trends = prev_trends or {}
    counts: dict[str, dict[str, int]] = {}
    for n in all_news:
        t = n.get("theme", "その他")
        c = counts.setdefault(t, {"pos": 0, "neg": 0, "mentions": 0})
        c["mentions"] += 1
        if n.get("sentiment") == "pos":
            c["pos"] += 1
        elif n.get("sentiment") == "neg":
            c["neg"] += 1

    trends = []
    for theme, c in counts.items():
        trends.append({
            "theme": theme,
            "mentions": c["mentions"],
            "pos": c["pos"],
            "neg": c["neg"],
            "prevMentions": prev_trends.get(theme, c["mentions"]),
        })
    trends.sort(key=lambda t: t["mentions"], reverse=True)
    return trends[:6]
