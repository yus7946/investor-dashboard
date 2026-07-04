"""EDINET API（無料・登録不要）から大量保有報告書を取得しアラート化する。
高頻度アクセス制限があるため、書類一覧取得は日付ごとに1回のみ呼び出す。
"""
import time
from datetime import date, timedelta

import requests

EDINET_LIST_URL = "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
LARGE_HOLDING_DOC_TYPE = "350"  # 大量保有報告書


def fetch_edinet_alerts(target_tickers: set[str], days_back: int = 3) -> list[dict]:
    alerts = []
    today = date.today()
    any_success = False
    for offset in range(days_back):
        d = today - timedelta(days=offset)
        try:
            resp = requests.get(
                EDINET_LIST_URL,
                params={"date": d.isoformat(), "type": 2},
                timeout=10,
            )
            resp.raise_for_status()
            any_success = True
            docs = resp.json().get("results", [])
            for doc in docs:
                if doc.get("docTypeCode") != LARGE_HOLDING_DOC_TYPE:
                    continue
                filer = doc.get("filerName", "機関投資家")
                desc = doc.get("docDescription", "大量保有報告書を提出")
                alerts.append({
                    "type": "warn",
                    "title": f"{filer[:18]}",
                    "desc": desc[:40],
                })
        except Exception:
            continue
        time.sleep(1)  # 節度あるアクセス間隔

    if not any_success or not alerts:
        alerts.extend(_fallback_alerts())
    return alerts[:5]


def _fallback_alerts() -> list[dict]:
    """EDINET接続失敗時や該当データなしの場合のフォールバック（出来高・決算予定ベースのサンプル）。"""
    return [
        {"type": "good", "title": "出来高急増（サンプル）", "desc": "EDINET接続不可のためサンプルアラートを表示中"},
        {"type": "info", "title": "決算発表シーズン", "desc": "保有銘柄の決算スケジュールを各自IRでご確認ください"},
        {"type": "warn", "title": "空売り動向（サンプル）", "desc": "最新の機関投資家動向は次回更新時に反映されます"},
    ]
