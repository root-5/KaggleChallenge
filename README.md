# KaggleChallenge

Kaggle に参加して Expert を目指すことを目的としたリポジトリです。
具体的には機械学習の基本フローを学びつつ、モデルの作成や MLOps の実践、上位入賞に向けたパイプライン作成などを行います。

**関連リンク**

- [nlp-test Notebook](https://www.kaggle.com/code/root5a/nlp-test)
- [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview)

**ドキュメント**

- [ロードマップ](doc/ロードマップ.md)
- [作業ログ](doc/作業ログ.md)
- [調査結果](doc/調査結果.md)

## 環境構築

1. uv の設定
   1. uv のインストール `curl -LsSf https://astral.sh/uv/install.sh | sh`
   2. uv 同期 `uv sync`
   3. コマンドパレットから「Python: Select Interpreter」を選択
   4. uv が管理している Python インタープリタを選択
2. Kaggle API の設定
   1. [API キーの発行](https://www.kaggle.com/settings)、json ファイルをダウンロード
   2. json ファイルを ~/.kaggle に配置し、権限を設定 `chmod 600 ~/.kaggle/kaggle.json`
3. Kaggle Notebook の設定
   1. ブラウザの Kaggle の左メニューの「Create」を押下し、「Notebook」を選択、エディタ等が開く
   2. 右メニューの「Add Input」から参加したいコンペ名を検索して追加
   3. 任意の Notebook 名を設定する
4. Kaggle ローカル環境の設定
   1. Kaggle API を使ってカーネルをメタデータ付きでダウンロード `uv run kaggle kernels pull [user名]/[Notebook名] -p kaggle/src/ -m`
   2. Kaggle API を使ってメタデータからコンペ情報を取得、一時的な環境変数に保存 `export COMPETITION_NAME=$(grep -ozP '"competition_sources"\s*:\s*\[\s*\K"[^"]+' kaggle/src/kernel-metadata.json | tr -d '"\0')`
   3. Kaggle API を使ってメタデータからコンペ情報を取得、入力データをダウンロード `uv run kaggle competitions download $COMPETITION_NAME -p kaggle/input/`
   4. unzip して展開、zip の削除 `unzip kaggle/input/$COMPETITION_NAME.zip -d kaggle/input/$COMPETITION_NAME && rm kaggle/input/$COMPETITION_NAME.zip`
5. VSCode 拡張機能のインストール
   1. black (Python コードフォーマッタ)
   2. prettier (markdown コードフォーマッタ)
   3. Jupyter (Jupyter Notebook サポート)

## 普段の作業イメージ

1. `git pull` で最新コードを取得
2. ～コード修正～
3. Kaggle API を使ってコンペの Notebook のコードをアップロード `uv run kaggle kernels push -p kaggle/src/`
4. Kaggle のブラウザ上で Notebook を実行結果を確認
5. コミットを作成、`git push` でコードをリモートリポジトリに反映

## 実装内容

### プロジェクト構成

```
KaggleChallenge/
├──.vscode/               # VSCodeの設定ファイル
├── doc/                  # ドキュメント関連
├── kaggle/               # Kaggle 関連ファイル (Kaggle サーバーと同期させる)
│   ├── input/            # コンペのデータ
│   └── src/              # Kaggle Notebook のソースコード
├── .python-version       # pyenv の Python バージョン設定ファイル
├── pyproject.toml        # プロジェクト設定ファイル
├── uv.lock               # uv のロックファイル (編集禁止)
├── AGENTS.md             # エージェントに対する指示書
└── README.md             # プロジェクトの概要
```

### 機能一覧
