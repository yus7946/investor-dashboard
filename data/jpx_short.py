"""JPX公表の空売り残高報告（無料・日次）から銘柄別の空売り残高を集計する。

出典: https://www.jpx.co.jp/markets/public/short-selling/
残高割合0.5%以上の開示ポジションが銘柄・提出者ごとに掲載される。
取得失敗時は空を返す（架空データは返さない）。
"""
import io
import re

import pandas as pd
import requests

INDEX_URL = "https://www.jpx.co.jp/markets/public/short-selling/index.html"
BASE_URL = "https://www.jpx.co.jp"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def _latest_file_url() -> str | None:
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    m = re.search(r'href="(/markets/public/short-selling/[^"]*\d{8}_Short_Positions\.xls)"', resp.text)
    return BASE_URL + m.group(1) if m else None


def fetch_short_positions(universe_map: dict[str, str]) -> tuple[list[dict], dict[str, float]]:
    """戻り値: (アラートのリスト, {ticker: 空売り残高割合合計%})"""
    try:
        url = _latest_file_url()
        if not url:
            return [], {}
        date_str = re.search(r"(\d{8})_Short_Positions", url).group(1)
        date_label = f"{date_str[4:6]}/{date_str[6:8]}"

        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=None)

        # ヘッダー行（「計算年月日」を含む行）を特定し、その2行下からデータ
        header_row = None
        for i in range(min(12, len(df))):
            if df.iloc[i].astype(str).str.contains("計算年月日").any():
                header_row = i
                break
        if header_row is None:
            return [], {}
        data = df.iloc[header_row + 2:]

        # 列: 2=銘柄コード, 3=銘柄名, 10=空売り残高割合, 14=直近(前回)空売り残高割合
        code4_to_ticker = {t.split(".")[0]: t for t in universe_map}
        agg: dict[str, dict] = {}
        for _, row in data.iterrows():
            code = str(row.iloc[2]).strip().split(".")[0]
            if code not in code4_to_ticker:
                continue
            ratio = pd.to_numeric(row.iloc[10], errors="coerce")
            prev = pd.to_numeric(row.iloc[14], errors="coerce")
            if pd.isna(ratio):
                continue
            a = agg.setdefault(code, {"ratio": 0.0, "prev": 0.0, "n": 0})
            a["ratio"] += float(ratio)
            a["prev"] += float(prev) if pd.notna(prev) else float(ratio)
            a["n"] += 1

        alerts = []
        ratios: dict[str, float] = {}
        for code, a in sorted(agg.items(), key=lambda kv: kv[1]["ratio"], reverse=True):
            ticker = code4_to_ticker[code]
            name = universe_map[ticker]
            ratios[ticker] = round(a["ratio"] * 100, 2)
            if len(alerts) >= 3:
                continue
            rising = a["ratio"] > a["prev"] * 1.02
            falling = a["ratio"] < a["prev"] * 0.98
            trend = "増加中" if rising else "減少中" if falling else "横ばい"
            alerts.append({
                "type": "warn" if rising else "info",
                "title": f"空売り残高: {name} ({code})",
                "desc": f"開示合計{a['ratio']*100:.2f}%・前回比{trend}（{a['n']}件・出典JPX {date_label}時点）",
            })
        return alerts, ratios
    except Exception:
        return [], {}
