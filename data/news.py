"""銘柄別ニュース取得。Google News RSS（無料・キー不要）を主とし、yfinanceを予備とする。

取得できない場合は空を返す（架空の見出しは生成しない）。
sentimentはキーワードによる簡易判定（画面にも「簡易」と明記する）。
"""
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

try:
    import yfinance as yf
except ImportError:
    yf = None

POS_WORDS = [
    "増益", "上方修正", "好調", "最高益", "増配", "拡大", "回復", "承認", "好決算",
    "上昇", "買い", "最高値", "受注", "提携", "投資", "急騰", "堅調", "黒字",
]
NEG_WORDS = [
    "減益", "下方修正", "不振", "減配", "縮小", "悪化", "撤退", "減産",
    "下落", "売り", "安値", "赤字", "リコール", "訴訟", "急落", "延期", "停止",
]

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


def _fetch_gnews(name: str, limit: int) -> list[dict]:
    """Google News RSS検索（日本語）。"""
    q = urllib.parse.quote(f"{name} 株")
    url = f"https://news.google.com/rss/search?q={q}&hl=ja&gl=JP&ceid=JP:ja"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    xml = urllib.request.urlopen(req, timeout=15).read()
    root = ET.fromstring(xml)
    out = []
    for it in root.findall(".//item")[:limit]:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        # タイトル末尾の「 - 媒体名」を分離
        source = "Google News"
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        if not title:
            continue
        out.append({
            "title": title[:80],
            "source": source[:20],
            "url": link,
            "pubDate": pub,
            "sentiment": _sentiment(title),
            "theme": _theme(title),
        })
    return out


def fetch_news_for_ticker(ticker: str, name: str, limit: int = 1) -> list[dict]:
    # 1) Google News RSS（日本語記事が豊富）
    try:
        items = _fetch_gnews(name, limit)
        if items:
            return items
    except Exception:
        pass
    # 2) yfinance（英語中心・予備）
    if yf is not None:
        try:
            items = yf.Ticker(ticker).news or []
            out = []
            for it in items[:limit]:
                content = it.get("content", it)
                title = content.get("title") or it.get("title")
                if not title:
                    continue
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
    # 3) 取得できなければ空（架空の見出しは生成しない）
    return []


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
