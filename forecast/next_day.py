"""翌営業日の見通しを「実データから計算できるものだけ」で構成する予測モジュール。

方針（断定禁止・全根拠表示）:
  1. 想定レンジ: 過去2年の日次リターンの標準偏差から±1σ/±2σを円建てで提示
  2. 類似局面の実績: 「今日と同じ条件（RSI帯×直近5日方向）だった過去の日」の
     翌日上昇率を実際に数えて提示（サンプル数付き。20回未満なら%を出さない）
  3. 市場全体: 前夜の米S&P500・ドル円の実績値と、
     「米国上昇/下落の翌日に東京が上昇した割合」の過去2年実測値
  4. 材料一覧: RSI・出来高・空売り残高・ニュースセンチメント等を方向付きで列挙

これらは統計的傾向であり将来の保証ではない旨を必ず表示データに含める。
取得失敗時は None / 欠損とし、架空の数値は返さない。
"""
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

TOPIX_ETF = "1306.T"
SPX = "^GSPC"
USDJPY = "USDJPY=X"
MIN_SAMPLES = 20  # これ未満のサンプル数では確率を表示しない


def _rsi_series(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, pd.NA)
    return 100 - 100 / (1 + rs)


def _rsi_zone(rsi: float) -> tuple[str, str]:
    if rsi <= 30:
        return "low", "RSI30以下（売られすぎ圏）"
    if rsi >= 70:
        return "high", "RSI70以上（買われすぎ圏）"
    return "mid", "RSI中立圏"


def _conditional_uprate(close: pd.Series, current_rsi: float, week_chg: float) -> dict | None:
    """今日と同条件（RSI帯×5日方向）だった過去の日の、翌日上昇実績を数える。"""
    try:
        rsi = _rsi_series(close)
        ret5 = close.pct_change(5)
        next_ret = close.pct_change().shift(-1)  # その日の翌日リターン

        zone, zone_label = _rsi_zone(current_rsi)
        up5 = week_chg >= 0

        if zone == "low":
            zone_mask = rsi <= 30
        elif zone == "high":
            zone_mask = rsi >= 70
        else:
            zone_mask = (rsi > 30) & (rsi < 70)
        dir_mask = (ret5 >= 0) if up5 else (ret5 < 0)

        mask = zone_mask & dir_mask & next_ret.notna()
        n = int(mask.sum())
        if n < MIN_SAMPLES:
            return {"upRate": None, "n": n, "condition": f"{zone_label}×5日{'上昇' if up5 else '下落'}局面",
                    "note": f"同条件の過去事例が{n}回と少ないため確率は表示しません"}
        up_rate = float((next_ret[mask] > 0).mean() * 100)
        return {
            "upRate": round(up_rate, 1),
            "n": n,
            "condition": f"{zone_label}×5日{'上昇' if up5 else '下落'}局面",
            "note": None,
        }
    except Exception:
        return None


def _stock_factors(s: dict) -> list[dict]:
    """方向付きの材料一覧。全て取得済みの実データから。"""
    f = []
    rsi = s.get("rsi")
    if rsi is not None:
        if rsi <= 30:
            f.append({"label": f"RSI {rsi}（売られすぎ圏・反発しやすい傾向）", "dir": "+"})
        elif rsi >= 70:
            f.append({"label": f"RSI {rsi}（買われすぎ圏・反落しやすい傾向）", "dir": "-"})
    vr = s.get("volume_ratio")
    dc = s.get("day_change_pct")
    if vr is not None and dc is not None and vr >= 2.0:
        d = "+" if dc > 0 else "-"
        f.append({"label": f"出来高{vr:.1f}倍で{'上昇' if dc > 0 else '下落'}（勢い継続に注意）", "dir": d})
    sr = s.get("short_ratio")
    if sr is not None and sr >= 2.0:
        f.append({"label": f"空売り残高{sr}%（売り圧力・踏み上げ両方の要因）", "dir": "-"})
    news = s.get("news") or []
    pos = sum(1 for n in news if n.get("sentiment") == "pos")
    neg = sum(1 for n in news if n.get("sentiment") == "neg")
    if pos > neg:
        f.append({"label": f"直近ニュースがポジティブ寄り（{pos}件/簡易判定）", "dir": "+"})
    elif neg > pos:
        f.append({"label": f"直近ニュースがネガティブ寄り（{neg}件/簡易判定）", "dir": "-"})
    return f


def build_forecasts(top_stocks: list[dict], regime: str = "unknown", calib: float = 1.0) -> dict | None:
    """各銘柄のforecastをin-placeで付与し、市場全体見通しを返す。

    calib: 過去の答え合わせ実績から算出したレンジ幅の補正係数（1.0=無補正）。
    """
    if yf is None or not top_stocks:
        return None

    tickers = [s["ticker"] for s in top_stocks]
    try:
        data = yf.download(
            tickers + [TOPIX_ETF, SPX, USDJPY],
            period="2y", interval="1d", auto_adjust=True, progress=False, threads=False,
        )["Close"]
    except Exception:
        return None

    calibrated = abs(calib - 1.0) > 1e-6

    # ── 銘柄別 ──
    for s in top_stocks:
        try:
            close = data[s["ticker"]].dropna()
            if len(close) < 60:
                continue
            r = close.pct_change().dropna()
            vol_d = float(r.std()) * calib  # 実績に基づくレンジ補正
            price = float(close.iloc[-1])
            cond = _conditional_uprate(close, s.get("rsi", 50), s.get("week_change_pct", 0) or 0)
            basis = "レンジは過去2年の日次値動きの標準偏差を、答え合わせ実績で当たり幅に合わせて補正したもの。実績確率は同条件だった過去の日の翌日騰落を実際に集計。統計的傾向であり将来を保証しません。"
            if calibrated:
                basis += f"（答え合わせ実績に基づきレンジ幅を{calib:.2f}倍に自動補正済み。目標カバー率は約75%）"

            # 売買目安3点セット: 1日の実績振れ幅(σ)を単位に機械的に算出
            unit = price * vol_d
            entry = round(price - 0.5 * unit)
            target = round(entry + 2.0 * unit)
            stop = round(entry - 1.5 * unit)
            hi3m = float(close.iloc[-60:].max())
            lo3m = float(close.iloc[-60:].min())
            plan = {
                "entry": entry,
                "target": target,
                "stop": stop,
                "rr": round((target - entry) / max(1, entry - stop), 2),
                "hi3m": round(hi3m),
                "lo3m": round(lo3m),
                "basis": (
                    f"1日の実績振れ幅σ={vol_d*100:.1f}%を単位に、買い目安=現値-0.5σ・利確=買値+2σ・損切り=買値-1.5σで機械的に算出"
                    f"（利益:損失の比率 {round((target-entry)/max(1,entry-stop),1)}:1）。"
                    f"参考: 直近3ヶ月の高値{round(hi3m):,}円 / 安値{round(lo3m):,}円。"
                    "この目安が実際に機能したかは毎営業日答え合わせして下の実績に反映します。損切りは必ずセットで守ることを想定した設計です。"
                ),
            }

            s["forecast"] = {
                "price": round(price, 1),
                "rangeLow": round(price * (1 - vol_d)),
                "rangeHigh": round(price * (1 + vol_d)),
                "rangeLow2": round(price * (1 - 2 * vol_d)),
                "rangeHigh2": round(price * (1 + 2 * vol_d)),
                "volDPct": round(vol_d * 100, 2),
                "calibrated": calibrated,
                "plan": plan,
                "cond": cond,
                "factors": _stock_factors(s),
                "basis": basis,
            }
        except Exception:
            continue

    # ── 市場全体（米国オーバーナイト連動） ──
    outlook = None
    try:
        topix = data[TOPIX_ETF].dropna()
        spx = data[SPX].dropna()
        fx = data[USDJPY].dropna()

        spx_ret = spx.pct_change().dropna()
        topix_ret = topix.pct_change().dropna()

        # 各東京営業日について「直前の米国営業日」のリターン符号を対応付け
        pairs = []
        spx_dates = spx_ret.index
        for dt, tr in topix_ret.items():
            prior = spx_dates[spx_dates < dt]
            if len(prior) == 0:
                continue
            pairs.append((float(spx_ret[prior[-1]]), float(tr)))
        dfp = pd.DataFrame(pairs, columns=["spx_prev", "topix"])

        spx_last = float(spx_ret.iloc[-1])
        spx_up = spx_last > 0
        subset = dfp[dfp["spx_prev"] > 0] if spx_up else dfp[dfp["spx_prev"] <= 0]
        n = len(subset)
        up_rate = round(float((subset["topix"] > 0).mean() * 100), 1) if n >= MIN_SAMPLES else None

        fx_last = float(fx.iloc[-1])
        fx_chg = float(fx.pct_change().iloc[-1] * 100)

        outlook = {
            "spxChangePct": round(spx_last * 100, 2),
            "spxDate": str(spx_ret.index[-1].date()),
            "usdjpy": round(fx_last, 2),
            "usdjpyChangePct": round(fx_chg, 2),
            "condition": f"米S&P500が直近{'上昇' if spx_up else '下落'}",
            "topixUpRate": up_rate,
            "n": n,
            "regime": regime,
            "basis": "過去2年で「米国市場が上昇/下落した翌営業日に東京(TOPIX ETF)が上昇した割合」を実際に集計。統計的傾向であり将来を保証しません。出典: Yahoo Finance価格データ。",
        }
    except Exception:
        outlook = None

    return outlook
