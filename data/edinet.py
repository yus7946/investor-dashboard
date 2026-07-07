"""EDINET API（無料・登録不要）から大量保有報告書を取得しアラート化する。

documents.json の secCode（対象企業の証券コード）を使って「提出者 → 対象銘柄」を紐付ける。
分析対象ユニバースに含まれる銘柄への報告を優先表示する。
接続失敗・該当なしの場合は架空のアラートを表示せず、その旨を正直に返す。
高頻度アクセス制限があるため、書類一覧取得は日付ごとに1回のみ呼び出す。
"""
import time
from datetime import date, timedelta

import requests

EDINET_LIST_URL = "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
LARGE_HOLDING_DOC_TYPES = {"350", "360"}  # 大量保有報告書・変更報告書


def fetch_edinet_alerts(universe_map: dict[str, str], days_back: int = 3) -> list[dict]:
    """universe_map: {"7203.T": "トヨタ自動車", ...}"""
    code4_to_name = {t.split(".")[0]: n for t, n in universe_map.items()}

    in_universe = []
    others = []
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
                if doc.get("docTypeCode") not in LARGE_HOLDING_DOC_TYPES:
                    continue
                filer = (doc.get("filerName") or "提出者不明").strip()
                sec_code = (doc.get("secCode") or "").strip()
                code4 = sec_code[:4] if len(sec_code) >= 4 else ""
                subject = code4_to_name.get(code4)
                kind = "変更報告" if doc.get("docTypeCode") == "360" else "大量保有報告"

                if subject:
                    in_universe.append({
                        "type": "warn",
                        "title": f"{kind}: {subject} ({code4})",
                        "desc": f"{filer[:24]} が提出（{d.strftime('%m/%d')}・EDINET）",
                    })
                elif code4:
                    others.append({
                        "type": "info",
                        "title": f"{kind}: コード{code4}",
                        "desc": f"{filer[:24]} が提出（{d.strftime('%m/%d')}・EDINET）",
                    })
        except Exception:
            continue
        time.sleep(1)  # 節度あるアクセス間隔

    # 分析対象銘柄への報告を優先し、残り枠をその他で埋める
    alerts = in_universe[:5] + others[: max(0, 3 - len(in_universe))]

    if not any_success:
        return [{
            "type": "info",
            "title": "EDINET大量保有: 取得失敗",
            "desc": "EDINETに接続できませんでした。次回更新時に再試行します（架空のアラートは表示しません）",
        }]
    if not alerts:
        return [{
            "type": "info",
            "title": "EDINET大量保有: 新着なし",
            "desc": f"直近{days_back}日間に大量保有報告書の提出はありませんでした（出典: EDINET）",
        }]
    return alerts
