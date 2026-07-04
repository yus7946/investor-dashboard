# 機関投資家型 投資ダッシュボード

完全無料・自動更新の個人投資判断支援ダッシュボード。

## 構成
- `main.py` : 株価取得→スコアリング→シグナル判定→ニュース→EDINET→バックテスト→JSON出力
- `update_dashboard.py` : `output/dashboard_data.json` を `investor_dashboard.html` に焼き込み `output/investor_dashboard_latest.html` を生成
- `.github/workflows/update_dashboard.yml` : 平日朝(JST)に自動実行・自動push

## ローカル実行
```
pip install -r requirements.txt
python main.py
python update_dashboard.py
```

## 無料でスマホからリアルタイムに見られるようにする手順
1. このフォルダをGitHubリポジトリ（public推奨。privateでも月2000分の無料Actions枠で足ります）として作成しpush。
2. リポジトリの Settings → Pages で、公開ソースを `output/` ディレクトリ（または `output/investor_dashboard_latest.html` を `docs/index.html` にコピーする運用に変更）に設定。
3. Actions タブで `update_dashboard.yml` が有効になっていることを確認（追加のSecrets設定は不要、`GITHUB_TOKEN` は自動付与）。
4. 公開されたURLをスマホのブラウザで開き、「ホーム画面に追加」でPWAとしてインストール。
5. ウォッチタブでntfy.shのトピック名を設定し、スマホにntfyアプリを入れて同じトピックを購読すると、強い買いシグナル検出時に通知が届きます（ページを開いている間のみ）。

## 注意
- EDINET APIは過度な高頻度アクセスを避ける設計にしてあります。
- yfinance等の外部通信が失敗した場合はサンプルデータで自動補完され、処理は止まりません。
- 「必ず勝てる」ことを保証するものではなく、判断材料を提供する意思決定支援ツールです。
