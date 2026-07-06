"""分析パイプラインの統合実行エントリーポイント。
各ステップはtry/exceptで保護し、エラーが起きても可能な限り処理を継続する。
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.universe import UNIVERSE
from data.fetch_prices import fetch_universe
from data.edinet import fetch_edinet_alerts
from data.news import fetch_news_for_ticker, build_theme_trends
from screening.score import score_universe
from signals.signal import judge_all
from backtest.backtest import run_backtest
from reports.json_export import export_dashboard_json

# 一部銘柄の株主優待情報（手動メンテナンス。yfinanceには優待情報が無いため）
YUTAI_MAP = {
    "7203.T": "なし（株主優待は廃止）",
    "8801.T": "保有株数に応じ宿泊優待割引券（要IR確認・手動更新）",
    "9432.T": "なし（過去はdポイント優待）",
}


def _estimate_flow(stocks: list[dict]) -> list[dict]:
    """無料の投資主体別売買代金データは公開APIが無いため、モメンタム平均からの簡易推定値。"""
    if not stocks:
        raise ValueError("no stocks")
    moms = [s.get("momentum_raw", 0) or 0 for s in stocks]
    avg_mom = sum(moms) / len(moms)
    foreign = round(avg_mom * 20000)
    trust = round(-avg_mom * 6000)
    individual = round(-avg_mom * 12000)
    return [
        {"label": "外国人", "value": foreign, "max": 8000},
        {"label": "信託銀行", "value": trust, "max": 3000},
        {"label": "個人", "value": individual, "max": 5000},
    ]


def main():
    print("=== 機関投資家型AIエージェント バックエンド処理 開始 ===")

    print("\n1/7 株価・財務データ取得中...")
    stocks = []
    try:
        stocks = fetch_universe(UNIVERSE)
        print(f"  取得件数: {len(stocks)} / {len(UNIVERSE)}")
    except Exception as e:
        print(f"  [エラー] 株価取得に失敗しました: {e}")

    print("\n2/7 スコアリング中...")
    try:
        stocks = score_universe(stocks)
        print(f"  スコアリング完了: {len(stocks)} 銘柄")
    except Exception as e:
        print(f"  [エラー] スコアリングに失敗しました: {e}")

    print("\n3/7 シグナル判定中...")
    try:
        stocks = judge_all(stocks)
        print("  シグナル判定完了")
    except Exception as e:
        print(f"  [エラー] シグナル判定に失敗しました: {e}")

    top10 = stocks[:10] if stocks else []

    print("\n4/7 銘柄ニュース取得中...")
    all_news = []
    try:
        for s in top10:
            news = fetch_news_for_ticker(s["ticker"], s["name"], limit=1)
            s["news"] = news
            all_news.extend(news)
        for s in top10:
            s["yutai"] = YUTAI_MAP.get(s["ticker"])
        print(f"  ニュース取得完了（{len(all_news)}件）")
    except Exception as e:
        print(f"  [エラー] ニュース取得に失敗しました: {e}")

    print("\n5/7 EDINETアラート・テーマトレンド集計中...")
    edinet_alerts = []
    theme_trends = []
    try:
        tickers = {s["ticker"] for s in top10}
        edinet_alerts = fetch_edinet_alerts(tickers)
        print(f"  アラート件数: {len(edinet_alerts)}")
    except Exception as e:
        print(f"  [エラー] EDINETアラート取得に失敗しました: {e}")
    try:
        theme_trends = build_theme_trends(all_news)
        if not theme_trends:
            raise ValueError("テーマ集計結果が0件")
        print(f"  テーマ件数: {len(theme_trends)}")
    except Exception as e:
        # 架空のテーマ数値は表示しない。取得できなければ空のままUI側で「データなし」を表示する。
        print(f"  [警告] テーマ集計に失敗（表示なしになります）: {e}")
        theme_trends = []

    print("\n6/7 投資主体別フロー推定・バックテスト実行中...")
    flow = []
    try:
        flow = _estimate_flow(stocks)
        print("  フロー推定完了")
    except Exception as e:
        # 架空のフロー数値は表示しない
        print(f"  [警告] フロー推定に失敗（表示なしになります）: {e}")
        flow = []

    bt = {}
    try:
        # 現在のスコア上位だけに絞ると選択バイアスが入るため、取得できた全銘柄で検証する
        bt = run_backtest([s["ticker"] for s in stocks])
        print(f"  バックテスト完了: 年率{bt.get('annual')} シャープ{bt.get('sharpe')}")
    except Exception as e:
        print(f"  [エラー] バックテストに失敗しました: {e}")
        bt = {
            "annual": "—", "benchmark": "—", "sharpe": "—", "dd": "—",
            "winrate": "—", "months": "—",
            "note": "データ取得に失敗したため計測できませんでした（架空の数値は表示しません）",
            "unavailable": True,
        }

    # 急騰・落ちナイフ・ローテーションアラート生成
    extra_alerts = []
    for s in stocks:
        dc = s.get("day_change_pct", 0) or 0
        vr = s.get("volume_ratio", 1) or 1
        wc = s.get("week_change_pct", 0) or 0
        rsi = s.get("rsi", 50) or 50
        name = s.get("name", s.get("ticker", ""))
        if dc >= 0.05 and vr >= 2.0:
            extra_alerts.append({
                "type": "good",
                "title": f"急騰: {name}",
                "desc": f"+{dc:.1%}・出来高{vr:.1f}倍",
            })
        if rsi <= 28 and wc <= -0.12 and vr >= 1.5:
            extra_alerts.append({
                "type": "warn",
                "title": f"逆張り候補: {name}",
                "desc": f"RSI{rsi}・週次{wc:.1%}・要リスク管理",
            })
    for s in stocks:
        name = s.get("name", s.get("ticker", ""))
        if s.get("rotation_out"):
            extra_alerts.append({
                "type": "warn",
                "title": f"入替候補OUT: {name}",
                "desc": f"スコア下位15% (score={s.get('score', 0)})",
            })
        if s.get("rotation_in"):
            extra_alerts.append({
                "type": "good",
                "title": f"入替候補IN: {name}",
                "desc": f"高スコア+モメンタム (score={s.get('score', 0)})",
            })
    combined_alerts = extra_alerts + edinet_alerts

    print("\n7/7 JSON出力中...")
    try:
        path = export_dashboard_json(
            stocks=top10 if top10 else stocks[:10],
            alerts=combined_alerts,
            flow=flow,
            theme_trends=theme_trends,
            backtest=bt,
            fetched_count=len(stocks),
            universe_total=len(UNIVERSE),
        )
        print(f"  出力完了: {path}")
    except Exception as e:
        print(f"  [致命的エラー] JSON出力に失敗しました: {e}")
        sys.exit(1)

    print("\n=== 処理完了 ===")


if __name__ == "__main__":
    main()
