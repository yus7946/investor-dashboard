"""分析パイプラインの統合実行エントリーポイント。
各ステップはtry/exceptで保護し、エラーが起きても可能な限り処理を継続する。

実行モード（環境変数 DASHBOARD_LIGHT=1 でライト更新）:
  - フル: 全データ取得（寄付き前の1日1回。EDINET/JPX/ニュース/決算日/バックテスト含む）
  - ライト: 株価・スコア・シグナル・リスク・見通しのみ毎時更新。
    日中に更新されない重いデータ（EDINET/JPX/ニュース等）は朝のフル更新の
    キャッシュを再利用し、各データの出典・対象期間ラベルはそのまま保つ。
    （高頻度アクセスによるIPブロック回避と実行時間短縮のため）
"""
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from data.universe import UNIVERSE, MASTER, EXTRA_HOLDINGS
from data.fetch_prices import fetch_universe, fetch_stock_data
from data.edinet import fetch_edinet_alerts
from data.earnings import annotate_earnings, build_earnings_board, fetch_past_earnings_dates
from data.news import fetch_news_for_ticker, build_theme_trends
from data.market_regime import fetch_market_regime
from data.jpx_flow import fetch_investor_flow
from data.jpx_short import fetch_short_positions
from screening.score import score_universe
from screening.risk import annotate_risk, build_low_risk_plan
from signals.signal import judge_all
from backtest.backtest import run_backtest
from forecast.next_day import build_forecasts
from forecast.tracker import (
    load_history, save_history, calibration_factor, update, compute_accuracy,
)
from reports.json_export import export_dashboard_json

HEAVY_CACHE_PATH = "output/heavy_cache.json"


def _load_heavy_cache() -> dict:
    try:
        with open(HEAVY_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_heavy_cache(cache: dict) -> None:
    try:
        os.makedirs("output", exist_ok=True)
        with open(HEAVY_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [警告] 重データキャッシュの保存に失敗: {e}")

# 一部銘柄の株主優待情報（手動メンテナンス。yfinanceには優待情報が無いため。要IR確認）
YUTAI_MAP = {
    "7203.T": "なし（株主優待は廃止）",
    "8801.T": "保有株数に応じ宿泊優待割引券（要IR確認・手動更新）",
    "9432.T": "なし（過去はdポイント優待）",
    "2181.T": "なし（パーソルHDは株主優待制度なし・配当のみ）",
    "2914.T": "100株以上でJT関連商品の詰め合わせ（年1回・要IR確認）",
    "8591.T": "2024年に株主優待は廃止済み",
    "9433.T": "100株以上でカタログギフト等（長期保有優遇・要IR確認）",
    "7974.T": "なし（任天堂は株主優待制度なし）",
    "9101.T": "なし（日本郵船は株主優待制度なし）",
}


def _build_stock_master(fetched_stocks: list[dict]) -> list[dict]:
    """持ち株ページ用の銘柄マスタを作る。

    UNIVERSEで取得済みの銘柄はその実データを再利用し、EXTRA_HOLDINGSの未取得分だけ個別取得。
    各エントリ: code, name, price, divPerShare, exDivDate, dividendYield, yutai。
    取得できなかったフィールドはnull（架空値は入れない）。
    """
    by_code = {s["ticker"]: s for s in fetched_stocks}
    master = []
    seen = set()
    for code, name in MASTER:
        if code in seen:
            continue
        seen.add(code)
        s = by_code.get(code)
        if s is None:
            # ユニバース外（EXTRA）で未取得のものだけ個別に取りに行く
            try:
                s = fetch_stock_data(code, name)
            except Exception:
                s = None
        entry = {
            "code": code,
            "name": name,
            "price": round(s["price"], 1) if s and s.get("price") else None,
            "divPerShare": s.get("div_per_share") if s else None,
            "exDivDate": s.get("ex_div_date") if s else None,
            "dividendYield": round(s.get("dividend_yield", 0), 2) if s and s.get("dividend_yield") else None,
            "per": round(s["per"], 1) if s and s.get("per") else None,
            "pbr": round(s["pbr"], 2) if s and s.get("pbr") else None,
            "roe": round(s.get("roe", 0), 1) if s and s.get("roe") else None,
            "rsi": s.get("rsi") if s else None,
            "vol": round(s["volatility"], 4) if s and s.get("volatility") else None,
            "inTop": code in {t["ticker"] for t in fetched_stocks[:10]},
            "yutai": YUTAI_MAP.get(code),
        }
        master.append(entry)
    return master


def main():
    light = os.environ.get("DASHBOARD_LIGHT") == "1"
    mode_label = "ライト（毎時・値動き系のみ）" if light else "フル（全データ取得）"
    print(f"=== 機関投資家型AIエージェント バックエンド処理 開始 [{mode_label}] ===")
    cache = _load_heavy_cache() if light else {}

    print("\n1/8 株価・財務データ取得中...")
    stocks = []
    try:
        # ライト時は決算日カレンダー取得(銘柄ごと+1リクエスト)を省略しキャッシュを使う
        stocks = fetch_universe(UNIVERSE, with_earnings=not light)
        print(f"  取得件数: {len(stocks)} / {len(UNIVERSE)}")
    except Exception as e:
        print(f"  [エラー] 株価取得に失敗しました: {e}")

    # 決算日の紐付け（フル: 新規取得しキャッシュ保存 / ライト: 朝のキャッシュで補完）
    try:
        date_map = annotate_earnings(stocks, cached_dates=cache.get("earnings_dates"))
        if not light:
            cache["earnings_dates"] = date_map
        print(f"  決算日ひもづけ: {sum(1 for s in stocks if s.get('earnings'))}銘柄")
    except Exception as e:
        print(f"  [警告] 決算日ひもづけに失敗しました: {e}")

    print("\n2/8 スコアリング中...")
    try:
        stocks = score_universe(stocks)
        print(f"  スコアリング完了: {len(stocks)} 銘柄")
    except Exception as e:
        print(f"  [エラー] スコアリングに失敗しました: {e}")

    print("\n3/8 地合い判定・シグナル判定中...")
    market = None
    try:
        market = fetch_market_regime()
        if market:
            print(f"  地合い: {market['label']}（TOPIX 200日線比 {market['priceVsMa200Pct']:+.1f}%）")
        else:
            print("  [警告] 地合い判定不能（シグナルは無調整・画面に地合い不明と表示）")
    except Exception as e:
        print(f"  [警告] 地合い判定に失敗しました: {e}")
    try:
        regime = market["regime"] if market else "unknown"
        stocks = judge_all(stocks, regime)
        print("  シグナル判定完了")
    except Exception as e:
        print(f"  [エラー] シグナル判定に失敗しました: {e}")

    top10 = stocks[:10] if stocks else []

    print("\n4/8 銘柄ニュース取得中...")
    all_news = []
    if light:
        # ライト時: 朝のフル更新時のニュースを再利用（Google Newsへの毎時アクセスを避ける）
        cached_news = cache.get("news_by_ticker") or {}
        for s in top10:
            s["news"] = cached_news.get(s["ticker"], [])
            all_news.extend(s["news"])
            s["yutai"] = YUTAI_MAP.get(s["ticker"])
        print(f"  ニュースはキャッシュ再利用（{len(all_news)}件）")
    else:
        try:
            for s in top10:
                news = fetch_news_for_ticker(s["ticker"], s["name"], limit=1)
                s["news"] = news
                all_news.extend(news)
            for s in top10:
                s["yutai"] = YUTAI_MAP.get(s["ticker"])
            cache["news_by_ticker"] = {s["ticker"]: s.get("news", []) for s in top10}
            print(f"  ニュース取得完了（{len(all_news)}件）")
        except Exception as e:
            print(f"  [エラー] ニュース取得に失敗しました: {e}")

    print("\n5/8 EDINET・空売り残高・テーマトレンド集計中...")
    edinet_alerts = []
    short_alerts = []
    short_ratios = {}
    theme_trends = []
    universe_map = dict(UNIVERSE)
    if light:
        # ライト時: EDINET/JPXは日中更新されないため朝のキャッシュを再利用
        edinet_alerts = cache.get("edinet_alerts") or []
        short_alerts = cache.get("short_alerts") or []
        short_ratios = cache.get("short_ratios") or {}
        theme_trends = cache.get("theme_trends") or []
        for s in top10:
            if s["ticker"] in short_ratios:
                s["short_ratio"] = short_ratios[s["ticker"]]
        print(f"  EDINET/空売り/テーマはキャッシュ再利用（EDINET{len(edinet_alerts)}件・空売り{len(short_ratios)}銘柄）")
    else:
        try:
            edinet_alerts = fetch_edinet_alerts(universe_map)
            print(f"  EDINETアラート件数: {len(edinet_alerts)}")
        except Exception as e:
            print(f"  [エラー] EDINETアラート取得に失敗しました: {e}")
        try:
            short_alerts, short_ratios = fetch_short_positions(universe_map)
            print(f"  空売り残高: {len(short_ratios)}銘柄で開示あり")
            for s in top10:
                if s["ticker"] in short_ratios:
                    s["short_ratio"] = short_ratios[s["ticker"]]
        except Exception as e:
            print(f"  [警告] 空売り残高取得に失敗しました: {e}")
        try:
            theme_trends = build_theme_trends(all_news)
            if not theme_trends:
                raise ValueError("テーマ集計結果が0件")
            print(f"  テーマ件数: {len(theme_trends)}")
        except Exception as e:
            # 架空のテーマ数値は表示しない。取得できなければ空のままUI側で「データなし」を表示する。
            print(f"  [警告] テーマ集計に失敗（表示なしになります）: {e}")
            theme_trends = []
        cache["edinet_alerts"] = edinet_alerts
        cache["short_alerts"] = short_alerts
        cache["short_ratios"] = short_ratios
        cache["theme_trends"] = theme_trends

    print("\n6/8 投資部門別フロー取得・バックテスト実行中...")
    flow = None
    if light:
        flow = cache.get("flow")
        print("  JPXフローはキャッシュ再利用" if flow else "  JPXフローのキャッシュなし（表示なし）")
    else:
        try:
            flow = fetch_investor_flow()
            if flow:
                print(f"  フロー取得完了（{flow['week']}週・JPX実データ）")
            else:
                print("  [警告] JPXフロー取得不能（表示なしになります・架空値は出しません）")
        except Exception as e:
            print(f"  [警告] フロー取得に失敗（表示なしになります）: {e}")
            flow = None
        cache["flow"] = flow

    bt = {}
    if light and cache.get("backtest"):
        bt = cache["backtest"]
        print("  バックテストはキャッシュ再利用（当日中は不変のため）")
    else:
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
        if not light and not bt.get("unavailable"):
            cache["backtest"] = bt

    # 急騰・落ちナイフ・ローテーションアラート生成。
    # 動き（急騰・逆張り）を最優先、次に入替IN、入替OUTは最後（多くて埋め尽くすため件数制限）
    move_alerts = []
    rotation_in_alerts = []
    rotation_out_alerts = []
    for s in stocks:
        dc = s.get("day_change_pct", 0) or 0
        vr = s.get("volume_ratio", 1) or 1
        wc = s.get("week_change_pct", 0) or 0
        rsi = s.get("rsi", 50) or 50
        name = s.get("name", s.get("ticker", ""))
        if dc >= 0.05 and vr >= 2.0:
            move_alerts.append({
                "type": "good",
                "title": f"急騰: {name}",
                "desc": f"+{dc:.1%}・出来高{vr:.1f}倍",
            })
        if rsi <= 28 and wc <= -0.12 and vr >= 1.5:
            move_alerts.append({
                "type": "warn",
                "title": f"逆張り候補: {name}",
                "desc": f"RSI{rsi}・週次{wc:.1%}・要リスク管理",
            })
    for s in stocks:
        name = s.get("name", s.get("ticker", ""))
        if s.get("rotation_in"):
            rotation_in_alerts.append({
                "type": "good",
                "title": f"入替候補IN: {name}",
                "desc": f"高スコア+モメンタム (score={s.get('score', 0)})",
            })
        if s.get("rotation_out"):
            rotation_out_alerts.append({
                "type": "warn",
                "title": f"入替候補OUT: {name}",
                "desc": f"スコア下位15% (score={s.get('score', 0)})",
            })
    # 優先度順: 実データ系（EDINET大量保有・空売り）→ 値動き → 入替IN → 入替OUT（各上限あり）
    combined_alerts = (
        edinet_alerts
        + short_alerts
        + move_alerts
        + rotation_in_alerts[:3]
        + rotation_out_alerts[:3]
    )[:14]

    print("\n7/8 翌営業日見通し計算・答え合わせ中...")
    # 過去の決算発表日（材料インパクト実績用）。フル時のみ取得しキャッシュ（top10で+10リクエスト）
    past_earnings = {}
    if light:
        past_earnings = cache.get("past_earnings") or {}
        if past_earnings:
            print(f"  過去決算日はキャッシュ再利用（{len(past_earnings)}銘柄）")
    else:
        try:
            for s in top10:
                dates = fetch_past_earnings_dates(s["ticker"])
                if dates:
                    past_earnings[s["ticker"]] = dates
            cache["past_earnings"] = past_earnings
            print(f"  過去決算日取得: {len(past_earnings)}/{len(top10)}銘柄")
        except Exception as e:
            print(f"  [警告] 過去決算日の取得に失敗: {e}")
    outlook = None
    accuracy = None
    try:
        regime = market["regime"] if market else "unknown"
        # 過去の答え合わせ実績からレンジ補正係数を算出
        history = load_history()
        calib = calibration_factor(history)
        if abs(calib - 1.0) > 1e-6:
            print(f"  実績ベースのレンジ補正係数: {calib}")
        outlook = build_forecasts(top10, regime, calib, past_earnings=past_earnings)
        done = sum(1 for s in top10 if s.get("forecast"))
        if outlook:
            print(f"  市場見通し: {outlook['condition']}→東京上昇率{outlook.get('topixUpRate')}%（過去{outlook['n']}回）")
        print(f"  銘柄別見通し: {done}/{len(top10)}銘柄で計算完了")
        # 答え合わせ（過去予測の照合）＋今回予測の記録
        history = update(history, top10, outlook)
        accuracy = compute_accuracy(history)
        save_history(history)
        print(f"  答え合わせ蓄積: 銘柄{accuracy['resolvedStockCount']}件・市場{accuracy['resolvedMarketCount']}件 解決済み"
              + (f"（レンジ的中率{accuracy['rangeHitRate']}%）" if accuracy.get("rangeHitRate") is not None else ""))
    except Exception as e:
        print(f"  [警告] 見通し・答え合わせに失敗しました（表示なしになります）: {e}")

    # リスク管理レイヤー（安全度・低リスク運用プラン・決算速報ボード）
    print("\nリスク評価・低リスクプラン生成中...")
    low_risk_plan = None
    earnings_board = None
    try:
        regime = market["regime"] if market else "unknown"
        annotate_risk(top10)
        low_risk_plan = build_low_risk_plan(top10, regime)
        print(f"  安全度評価: {sum(1 for s in top10 if s.get('risk'))}銘柄 / 低リスク候補: {len(low_risk_plan['basket'])}銘柄")
    except Exception as e:
        print(f"  [警告] リスク評価に失敗しました（表示なしになります）: {e}")
    try:
        earnings_board = build_earnings_board(stocks)
        if earnings_board:
            print(f"  決算速報: 本日/直後{len(earnings_board['today'])}件・5営業日以内{len(earnings_board['soon'])}件")
        else:
            print("  決算速報: 至近の決算予定なし（取得できた範囲で）")
    except Exception as e:
        print(f"  [警告] 決算速報の生成に失敗しました: {e}")

    # 持ち株ページ用の銘柄マスタ（オートコンプリート・価格/配当/優待の自動反映用）
    print("\n銘柄マスタ生成中（持ち株ページ用）...")
    stock_master = _build_stock_master(stocks)
    print(f"  マスタ登録: {len(stock_master)}銘柄")

    print("\n8/8 JSON出力中...")
    try:
        path = export_dashboard_json(
            stocks=top10 if top10 else stocks[:10],
            alerts=combined_alerts,
            flow=flow,
            theme_trends=theme_trends,
            backtest=bt,
            market=market,
            market_outlook=outlook,
            accuracy=accuracy,
            stock_master=stock_master,
            fetched_count=len(stocks),
            universe_total=len(UNIVERSE),
            low_risk_plan=low_risk_plan,
            earnings_board=earnings_board,
            update_mode="light" if light else "full",
        )
        print(f"  出力完了: {path}")
    except Exception as e:
        print(f"  [致命的エラー] JSON出力に失敗しました: {e}")
        sys.exit(1)

    if not light:
        _save_heavy_cache(cache)
        print(f"  重データキャッシュ保存: {HEAVY_CACHE_PATH}（ライト更新時に再利用）")

    print("\n=== 処理完了 ===")


if __name__ == "__main__":
    main()
