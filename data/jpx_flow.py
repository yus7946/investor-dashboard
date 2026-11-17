"""JPX公表の投資部門別売買状況（無料・週次）から実測の資金フローを取得する。

出典: https://www.jpx.co.jp/markets/statistics-equities/investor-type/
東証プライムの週次売買代金（千円）から、主要投資部門の差引き（買越/売越）を億円で返す。
公表は約2週間遅れのため、対象週をラベルで明示する。
取得失敗時は空を返す（架空データは返さない）。
"""
import io
import json
import os
import re

import pandas as pd
import requests

INDEX_URL = "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html"
BASE_URL = "https://www.jpx.co.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 表示する投資部門（表示名, 列見出し（英語部分）に含まれる語）。複数列に一致したら合計する
# （海外投資家=法人＋個人、個人=現金＋信用）。2026-10下旬にJPXのファイルが stock_1_w_*.xlsx 形式に変わった。
CATEGORIES = [
    ("海外投資家", "Foreigners"),
    ("個人", "Brokerage > Individuals"),
    ("信託銀行", "Trust BK"),
    ("投資信託", "Investment Trusts"),
]
FILE_RE = r'href="(/markets/statistics-equities/investor-type/[^"]*stock_1_w_(\d{8})_(\d{8})\.xlsx)"'


def _latest_file() -> tuple[str, str, str] | None:
    """最新週のファイルURL・週ラベル（"MM/DD～MM/DD"）・週の開始日（YYYYMMDD）。"""
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    found = re.findall(FILE_RE, resp.text)
    if not found:
        return None
    path, start, end = max(found, key=lambda m: m[1])
    return BASE_URL + path, f"{start[4:6]}/{start[6:]}～{end[4:6]}/{end[6:]}", start


def _english(v) -> str:
    """「金額 Value」のような日英併記セルから英語部分だけを取り出す。"""
    m = re.search(r"[A-Za-z][A-Za-z .,&\-/()]*$", str(v).replace("\n", " "))
    return m.group(0).strip() if m and str(v) != "nan" else ""


def fetch_investor_flow() -> dict | None:
    """戻り値: {"week": "11/02～11/06", "flows": [{"label","value"(億円)},...], "source": str} または None"""
    try:
        latest = _latest_file()
        if not latest:
            return None
        url, week_label, week_start = latest
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), header=None)

        # 見出し行（結合セル）を横に埋めて、各列を「Brokerage > Individuals > Cash > Balance」のような名前にする
        top = next(i for i in range(len(df)) if any("Proprietary" in str(v) for v in df.iloc[i]))
        sub = next(i for i in range(top, len(df)) if any("Balance" in str(v) for v in df.iloc[i]))
        hdr = df.iloc[top:sub + 1].ffill(axis=1)
        labels = {}
        for c in range(df.shape[1]):
            parts = [_english(hdr.iat[r, c]) for r in range(len(hdr))]
            labels[c] = " > ".join(dict.fromkeys(p for p in parts if p))

        # 東証プライムの「金額」行（市場名の行か、その次の行）
        prime = next(i for i in range(sub + 1, len(df)) if any("TSE Prime" in str(v) for v in df.iloc[i, :3]))
        value_row = next(i for i in (prime, prime + 1) if any("Value" in str(v) for v in df.iloc[i, :4]))

        flows = []
        for disp, key in CATEGORIES:
            cols = [c for c, lab in labels.items() if lab.endswith("Balance") and key in lab]
            vals = [pd.to_numeric(str(df.iat[value_row, c]).replace(",", ""), errors="coerce") for c in cols]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                flows.append({"label": disp, "value": round(float(sum(vals)) / 100000)})  # 千円→億円

        if not flows:
            return None
        return {
            "week": week_label,
            "start": week_start,
            "flows": flows,
            "source": "JPX 投資部門別売買状況（東証プライム・週次・金額ベース）",
        }
    except Exception:
        return None


HISTORY_PATH = "output/flow_history.json"
HISTORY_MAX_WEEKS = 26  # 約6ヶ月分。個人vs海外投資家の綱引きトレンドを見るのに十分な長さ。


def update_flow_history(flow: dict | None, path: str = HISTORY_PATH,
                        max_weeks: int = HISTORY_MAX_WEEKS) -> list[dict]:
    """今週分のJPXフローを履歴に追記（同じ週は上書き、架空データは追加しない）。

    戻り値は「個人投資家 vs 海外投資家」トレンド表示用に、週の新しい順ではなく古い順で返す。
    """
    try:
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        history = []

    if flow and flow.get("week") and flow.get("flows"):
        history = [h for h in history if h.get("week") != flow["week"]]
        history.append({"week": flow["week"], "start": flow.get("start"), "flows": flow["flows"]})
        # 年つきの開始日（YYYYMMDD）で並べる（"MM/DD"の文字列順だと年をまたいだ時に1月が先頭に来て最新週が消える）
        history.sort(key=lambda h: h.get("start") or h["week"])
        history = history[-max_weeks:]
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    return history
