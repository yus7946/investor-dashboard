"""JPX公表の投資部門別売買状況（無料・週次）から実測の資金フローを取得する。

出典: https://www.jpx.co.jp/markets/statistics-equities/investor-type/
東証プライムの週次売買代金（千円）から、主要投資部門の差引き（買越/売越）を億円で返す。
公表は約2週間遅れのため、対象週をラベルで明示する。
取得失敗時は空を返す（架空データは返さない）。
"""
import io
import re

import pandas as pd
import requests

INDEX_URL = "https://www.jpx.co.jp/markets/statistics-equities/investor-type/index.html"
BASE_URL = "https://www.jpx.co.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 表示する投資部門（表中の日本語ラベル → 表示名）
CATEGORIES = [
    ("海外投資家", "海外投資家"),
    ("個　人", "個人"),
    ("信託銀行", "信託銀行"),
    ("投資信託", "投資信託"),
]


def _latest_file_url() -> str | None:
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    m = re.search(
        r'href="(/markets/statistics-equities/investor-type/[^"]*stock_val_1_\d+\.xls)"',
        resp.text,
    )
    return BASE_URL + m.group(1) if m else None


def fetch_investor_flow() -> dict | None:
    """戻り値: {"week": "06/22～06/26", "flows": [{"label","value"(億円)},...], "source": str} または None"""
    try:
        url = _latest_file_url()
        if not url:
            return None
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name="TSE Prime", header=None)

        # 週ラベル行（"MM/DD～MM/DD"）を探す。最右の週ブロックが最新
        week_cells = []  # (row, col, label)
        for i in range(len(df)):
            for j in range(df.shape[1]):
                v = str(df.iat[i, j])
                if re.fullmatch(r"\d{2}/\d{2}～\d{2}/\d{2}", v.strip()):
                    week_cells.append((i, j, v.strip()))
        if not week_cells:
            return None
        # 最新週 = 最も右の列のブロック。差引き列はブロック先頭列+3
        latest = max(week_cells, key=lambda x: x[1])
        balance_col = latest[1] + 3
        week_label = latest[2]
        if balance_col >= df.shape[1]:
            return None

        flows = []
        for row_label, disp in CATEGORIES:
            for i in range(len(df)):
                cell = str(df.iat[i, 0]).strip()
                if cell != row_label:
                    continue
                # 差引きセルは売り/買い/合計3行のどこかに入る（結合セルのため）
                val = None
                for k in range(3):
                    if i + k >= len(df):
                        break
                    raw = df.iat[i + k, balance_col]
                    num = pd.to_numeric(str(raw).replace(",", ""), errors="coerce")
                    if pd.notna(num):
                        val = float(num)
                        break
                if val is not None:
                    flows.append({"label": disp, "value": round(val / 100000)})  # 千円→億円
                break

        if not flows:
            return None
        return {
            "week": week_label,
            "flows": flows,
            "source": "JPX 投資部門別売買状況（東証プライム・週次・金額ベース）",
        }
    except Exception:
        return None
