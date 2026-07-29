"""決算日程・決算速報を扱う。無料データ(yfinance)の範囲で、事実のみを扱う。

正直な限界:
  - 「発表の瞬間のリアルタイム速報」は無料データでは取得できない。
  - 本アプリの決算情報は、直近の毎時更新（取引時間中）時点での yfinance の値を反映する。
    決算はふつう15時の引け後に発表されるため、数値の反映は発表当日夜〜翌営業日の更新になる。
  - よって「本日決算発表」フラグ＝注意喚起（窓リスク・またぎ回避）が主目的で、
    実績値（売上/EPSの前年比）は取得できたものだけを出典付きで補助表示する。

  決算日は yfinance の Ticker.calendar から取得（バージョン差異に両対応）。
  取得失敗時は None とし、架空の日付・数値は一切入れない。
"""
from datetime import date, datetime

import pandas as pd


def _to_date(v):
    """yfinanceが返す様々な型(date/datetime/Timestamp/str)を date に正規化。"""
    if v is None:
        return None
    try:
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        if isinstance(v, datetime):
            return v.date()
        if isinstance(v, date):
            return v
        if isinstance(v, pd.Timestamp):
            return v.date()
        return pd.to_datetime(str(v)).date()
    except Exception:
        return None


def extract_earnings_date(ticker_obj) -> str | None:
    """yfinance Ticker から次回（または直近）決算日を 'YYYY-MM-DD' で返す。取れなければ None。"""
    try:
        cal = ticker_obj.calendar
    except Exception:
        return None
    d = None
    try:
        # 新しめ: dict形式 {'Earnings Date': [date, ...], ...}
        if isinstance(cal, dict):
            d = _to_date(cal.get("Earnings Date"))
        # 古い: DataFrame形式（行 'Earnings Date'）
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                d = _to_date(cal.loc["Earnings Date"].iloc[0])
            else:
                d = _to_date(cal.iloc[0, 0])
    except Exception:
        d = None
    return d.isoformat() if d else None


def _business_days_between(a: date, b: date) -> int:
    """a→b の営業日数（土日を除外・祝日は考慮しない簡易版）。b>=aで正、b<aで負。"""
    if a == b:
        return 0
    step = 1 if b > a else -1
    lo, hi = (a, b) if b > a else (b, a)
    n = int(pd.bdate_range(lo, hi).size) - 1  # 端点調整
    return step * max(0, n)


def annotate_earnings(stocks: list[dict], cached_dates: dict[str, str] | None = None) -> dict[str, str]:
    """各stockに earnings 情報を付与し、{ticker: 'YYYY-MM-DD'} の日付マップを返す（キャッシュ用）。

    cached_dates を渡すと、stock自身が決算日を持たない場合の補完に使う（ライト更新時）。
    """
    cached_dates = cached_dates or {}
    out_map = {}
    today = date.today()
    for s in stocks:
        tk = s["ticker"]
        raw = s.get("earnings_date") or cached_dates.get(tk)
        d = _to_date(raw)
        if not d:
            s["earnings"] = None
            continue
        out_map[tk] = d.isoformat()
        days_to = _business_days_between(today, d)
        rev = s.get("revenue_growth")
        eps = s.get("earnings_growth")
        s["earnings"] = {
            "date": d.isoformat(),
            "daysTo": days_to,
            "isToday": days_to == 0,
            "isSoon": -1 <= days_to <= 5,
            "passed": days_to < 0,
            # 直近実績（yfinance info・前年同期比。取得できたものだけ・出典明示）
            "revGrowthPct": round(rev * 100, 1) if isinstance(rev, (int, float)) else None,
            "epsGrowthPct": round(eps * 100, 1) if isinstance(eps, (int, float)) else None,
            "source": "決算日: yfinance calendar / 前年比: yfinance info（無料・遅延あり）",
        }
    return out_map


def build_earnings_board(stocks: list[dict]) -> dict | None:
    """画面の『決算速報』セクション用に、至近の決算銘柄を抽出してまとめる。"""
    today_list, soon_list = [], []
    for s in stocks:
        e = s.get("earnings")
        if not e:
            continue
        row = {
            "ticker": s["ticker"],
            "name": s["name"],
            "date": e["date"],
            "daysTo": e["daysTo"],
            "revGrowthPct": e.get("revGrowthPct"),
            "epsGrowthPct": e.get("epsGrowthPct"),
        }
        if e["isToday"] or (e["passed"] and e["daysTo"] >= -2):
            today_list.append(row)
        elif e["isSoon"] and e["daysTo"] > 0:
            soon_list.append(row)
    soon_list.sort(key=lambda x: x["daysTo"])
    if not today_list and not soon_list:
        return None
    return {
        "today": today_list,     # 本日発表 / 発表直後（〜2営業日）
        "soon": soon_list,       # 5営業日以内に発表予定
        "note": (
            "決算前後は株価が窓を開けて動きやすく、損切り幅を超えて飛ぶことがあります。"
            "決算をまたぐ保有は想定外の損失につながりやすいため、原則さけるのが低リスクです。"
            "実績値は発表後の毎時更新で反映されます（発表の瞬間の速報ではありません）。"
        ),
        "source": "yfinance（決算日・前年比・無料/遅延あり）",
    }
