# 引き継ぎメモ（2026-07-15時点）

## プロジェクト概要
- 名称: **株コンパス**（旧: 機関投資家型AIエージェント）
- 公開URL: https://yus7946.github.io/investor-dashboard/
- リポジトリ: https://github.com/yus7946/investor-dashboard
- ローカル作業ディレクトリ: `C:\Users\yano5\OneDrive\デスクトップ\機関投資家の動き分析アプリ`
- 完全無料構成: yfinance / JPX公開データ / EDINET / GitHub Actions / GitHub Pages
- 自動更新: 平日 JST 8:30・11:30（`.github/workflows/update_dashboard.yml`）

## 絶対に守るべき原則（過去に一度信頼を損ねた経緯あり）
1. **架空データ・断定的な物言いを絶対に使わない**。取得失敗時は「取得できず」と正直に表示。サンプル値・乱数埋めは禁止。
2. **投資助言に見える断定表現を避ける**（「必ず」「絶対」等）。根拠・出典を必ず併記。
3. コード変更後は **必ず `python3 update_dashboard.py` で `docs/` を再生成 → JS構文チェック(`node --check`) → 可能ならブラウザ実機検証 → git push** の順で確認してから完了とする。
4. push前に `git rev-parse HEAD` と `git rev-parse origin/main` を突き合わせて一致を確認する（OneDrive経由の通信でpushが不安定になることがある）。

## 実行環境の癖
- ローカルにPython/Node直叩き不可（PowerShellにパスが通っていない）。**WSL経由のbashスクリプトファイル**を`C:\Users\yano5\AppData\Local\Temp\`に書き、`powershell.exe -Command "wsl bash '<path>'"`で実行するのが確実。
- PowerShellの直接インライン`-Command`はクォート処理が壊れるため使わない。
- ブラウザ検証は `preview_start` → WSLでローカルサーバー起動 → `javascript_tool`でDOM/localStorageを直接検査するのが安定（`computer{screenshot}`はGSAPアニメーションでよくタイムアウトする）。

## アーキテクチャ
```
main.py                    # 全体オーケストレーション（8ステップ）
data/
  universe.py               # 分析対象166銘柄 + EXTRA_HOLDINGS（持株ページ補完用マスタ）
  fetch_prices.py            # yfinance取得 + smart_money計算呼び出し
  edinet.py                  # EDINET大量保有報告
  jpx_short.py                # JPX空売り残高（日次Excel）
  jpx_flow.py                  # JPX投資部門別売買状況（週次Excel）
  market_regime.py              # TOPIX200日線で地合い判定
  news.py                        # Google News RSS + yfinanceニュース
screening/
  score.py                   # 総合スコア合成（機関フロー30%/勢い25%/割安20%/品質15%/低ボラ10%）
  smart_money.py               # ★最新実装。CMF/OBV/MFI/VWAP乖離/出来高方向で機関の買い集め・売り抜けを推測
signals/signal.py             # 売買シグナル判定（相対順位＋RSI＋機関フロー補正＋地合い補正）
backtest/backtest.py          # 先読みなしバックテスト（TOPIX比較付き）
forecast/
  next_day.py                 # 翌営業日レンジ予測・売買目安3点セット
  tracker.py                    # 予測の答え合わせ・的中率蓄積・レンジ自動補正
reports/json_export.py       # output/dashboard_data.json 生成
update_dashboard.py          # HTML内DATA差し替え + docs/へコピー + PWA関連ファイル同期
investor_dashboard.html      # フロントエンド全部（単一ファイル、CSS/JS inline）
```

## 直近の実装履歴（新しい順）
1. **機関の資金フロー分析**（本セッション最終実装・公開済み）— ユーザーから「RSIだけのテクニカル分析に見える、機関投資家の実態分析ならCMF/OBV/VWAP/出来高/空売り残高等を組み合わせるべき」と指摘を受け実装。`screening/smart_money.py`で日足OHLCVから追加通信ゼロで計算。スコア最大比重に採用。銘柄カードに「買い集め/売り抜けの兆候」を根拠付きで表示。
2. 分析対象銘柄を150→166に拡大（ユーザー保有銘柄IGポート/GENDA/S&J/イーソル/スカパーJSAT追加）
3. 更新時刻のタイムゾーンバグ修正（UTC/JST変換漏れで「経過時間」と矛盾して見えた）
4. 持株フォームで対象外銘柄を手動登録できる機能追加
5. Figmaデザイン（ライトテーマ×ネイビー×アンティークゴールド）への全面リブランド、アプリ名を「株コンパス」に変更
6. 利益目標機能（月20万/年240万を軸に、元本から必要年利を逆算し現実性を正直・厳しめに評価）
7. データバックアップ機能（iOS Safariのstorage分離・自動削除対策としてJSON書き出し/読み込み）

## 既知の未対応・今後の候補
- **大口クロス取引・日中の板情報**: 無料では取得不可と検証済み（諦めるしかない）
- 決算/IR前後の売買パターン分析はまだ未実装（決算日はyfinance calendarで取得可能と検証済み、`smart_money.py`に「決算接近フラグ」を追加する余地あり）
- GitHub Actions環境（Ubuntu runner）でJPX/EDINET/Google News取得が安定するか、まだ継続監視が必要（IPブロック等のリスクは理論上残る）
- ユーザーは「必ず勝てる」を求めがちだが、その都度「保証はできない、規律で負けを減らす方向」と正直に説明してきた経緯あり。今後も同じスタンスを継続すること。

## ユーザーについて
- 保有銘柄: パーソルHD, 三菱UFJ, トヨタ自動車, IGポート, GENDA, S&J, イーソル, スカパーJSAT, 伊藤忠商事, 日本製鉄, NTT 等（持株会含む）
- 利益目標: 月20万円/年240万円（資金制約を自覚しており「現実的な目標」を重視）
- サービス化・他者展開の構想あり
- フィードバックは音声入力由来で長文・口語的。要点を汲み取って実装→検証→正直な結果報告のサイクルを好む
