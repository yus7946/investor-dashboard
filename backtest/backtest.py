"""先読みバイアスのない月次リバランス戦略のバックテスト。

検証方法:
  各月末に「前月末までの直近6ヶ月騰落率」上位N銘柄を選び、翌月保有する（モメンタム戦略の近似）。
  選定に使うのは常に過去の情報のみ（look-aheadなし）。TOPIX連動ETF(1306.T)を同期間の比較対象とする。

限界（正直な注記）:
  過去時点のPER/ROE等の財務データは無料では取得できないため、
  総合スコアのうち検証できるのは「勢い(モメンタム)」成分のみ。
  割安度・稼ぐ力を含む総合スコア自体の検証ではない。

データ取得に失敗した場合は架空の数値を返さず「計測不能」を返す。
"""
import time

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

FEE_RATE = 0.001      # 月次リバランス1回あたりの概算手数料
BENCHMARK = "1306.T"  # TOPIX連動ETF（比較対象）
LOOKBACK = 6          # モメンタム測定期間（ヶ月）
CHUNK = 30            # レート制限回避のための分割ダウンロード単位


def _clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    """データソース側の異常値を除去する。

    Yahooの月次データには単月だけ価格が桁落ちして翌月戻る異常（例: 435円→37円→394円）が
    実際に混ざるため、前後の月といずれも±50%超乖離する点をNaN化して計算から除外する。
    """
    prev = df.shift(1)
    nxt = df.shift(-1)
    spike = ((df / prev - 1).abs() > 0.5) & ((df / nxt - 1).abs() > 0.5)
    return df.mask(spike)


def _download_monthly(symbols: list[str]) -> pd.DataFrame:
    """月次終値を分割ダウンロードで取得（一括だとレート制限で全滅しやすいため）。"""
    frames = []
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        try:
            df = yf.download(chunk, period="3y", interval="1mo",
                             auto_adjust=True, progress=False, threads=False)["Close"]
            if isinstance(df, pd.Series):
                df = df.to_frame(name=chunk[0])
            frames.append(df.dropna(axis=1, how="all"))
        except Exception:
            pass
        time.sleep(1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


def run_backtest(tickers: list[str], months: int = 24, top_n: int = 10) -> dict:
    if yf is None or not tickers:
        return _unavailable("yfinance未導入または銘柄リストが空")
    try:
        symbols = sorted(set(tickers)) + [BENCHMARK]
        data = _download_monthly(symbols)
        if BENCHMARK not in data.columns:
            # ベンチマークのみ個別リトライ
            try:
                time.sleep(2)
                bh = yf.Ticker(BENCHMARK).history(period="3y", interval="1mo", auto_adjust=True)["Close"]
                bh.index = bh.index.tz_localize(None)
                if not data.empty:
                    data.index = pd.to_datetime(data.index).tz_localize(None)
                data[BENCHMARK] = bh
            except Exception:
                pass
        if data.empty or BENCHMARK not in data.columns:
            return _unavailable("価格データを取得できず（レート制限の可能性・次回更新で再試行）")

        data = _clean_prices(data)

        # 直近 months + LOOKBACK ヶ月分に絞る
        data = data.iloc[-(months + LOOKBACK):]
        bench_monthly = data[BENCHMARK].pct_change()
        prices = data.drop(columns=[BENCHMARK])
        if prices.shape[1] < top_n or len(prices) < LOOKBACK + 6:
            return _unavailable("検証に足る価格データが不足")

        monthly = prices.pct_change()

        port_rets, bench_rets = [], []
        # t=保有月。銘柄選定は t-1 月末までの情報のみを使用（先読みなし）
        for t in range(LOOKBACK + 1, len(prices)):
            past = (prices.iloc[t - 1] / prices.iloc[t - 1 - LOOKBACK] - 1).dropna()
            if len(past) < top_n:
                continue
            picks = past.sort_values(ascending=False).head(top_n).index
            hold = monthly.iloc[t][picks].dropna()
            if hold.empty:
                continue
            port_rets.append(float(hold.mean()) - FEE_RATE)
            b = bench_monthly.iloc[t]
            bench_rets.append(float(b) if pd.notna(b) else 0.0)

        if len(port_rets) < 6:
            return _unavailable("検証可能な月数が6ヶ月未満")

        series = pd.Series(port_rets)
        bench_series = pd.Series(bench_rets)
        annual = (1 + series.mean()) ** 12 - 1
        bench_annual = (1 + bench_series.mean()) ** 12 - 1
        # TOPIX連動ETFの年率が±60%を超えることは通常なく、データ異常が残っている場合は表示しない
        if abs(bench_annual) > 0.6:
            return _unavailable("ベンチマークデータに異常値が残存（次回更新で再試行）")
        sharpe = (series.mean() / series.std()) * (12 ** 0.5) if series.std() else 0.0
        cum = (1 + series).cumprod()
        dd = float(((cum / cum.cummax()) - 1).min())
        win = float((series > 0).mean() * 100)

        return {
            "annual": f"{annual*100:+.1f}%",
            "benchmark": f"{bench_annual*100:+.1f}%",
            "sharpe": f"{sharpe:.2f}",
            "dd": f"{dd*100:.1f}%",
            "winrate": f"{win:.1f}%",
            "months": f"{len(series)}ヶ月",
            "note": (
                f"検証方法: 前月末までの6ヶ月騰落率上位{top_n}銘柄を翌月保有・月次入替（先読みなし・手数料{FEE_RATE*100:.1f}%/月控除）。"
                f"同期間のTOPIX連動ETF(1306)は年率{bench_annual*100:+.1f}%。"
                "これは総合スコアのうち勢い(モメンタム)成分のみの近似検証で、割安度・稼ぐ力を含む総合スコア自体の検証ではありません。"
                "過去の実績であり、将来の成果を保証するものではありません。"
            ),
        }
    except Exception as e:
        return _unavailable(str(e)[:80])


def _unavailable(reason: str = "") -> dict:
    """取得失敗時。架空の数値は一切返さない。"""
    return {
        "annual": "—", "benchmark": "—", "sharpe": "—", "dd": "—",
        "winrate": "—", "months": "—",
        "note": "データ取得に失敗したため計測できませんでした（架空の数値は表示しません）"
                + (f"。理由: {reason}" if reason else ""),
        "unavailable": True,
    }
