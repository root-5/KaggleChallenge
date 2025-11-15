# KaggleChallenge

Kaggle に参加して Expert を目指すことを目的としたリポジトリです。
具体的には機械学習の基本フローを学びつつ、モデルの作成や MLOps の実践、上位入賞に向けたパイプライン作成などを行います。

## 目標と課題

### 現在の目標

- [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview) をやってみる
- [メダルが報酬のコンペ](https://www.kaggle.com/competitions?listOption=active&prestigeFilter=medals&participationFilter=open)に参加する

### 現在の課題

- 知らないことが多すぎるので、適宜調べたことを[調査結果](doc/調査結果.md)をまとめる

## アイディア

### 機能追加関係

### リファクタリング関係

## タスク

- [x] アカウント作成
- [x] コンペに参加
- [ ] チーム作成
- [ ] WEB 版 Kaggle Notebook の使ってみる
- [ ] Kaggle API を使った連携を確立
- [ ] Kaggle Notebook の開発環境構築

### 機能追加関係

### リファクタリング関係

## 実装内容

### 開発環境でやったこと

**VSCode 拡張**

- black (Python コードフォーマッタ)
- prettier (markdown コードフォーマッタ)

**Python 関係**

1. uv のインストール `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. uv 初期化 `uv init`
3. uv でのコード実行 `uv run src/main.py`

**Kaggle API**

1. Kaggle API のインストール `uv add kaggle`
2. インストール確認 `uv run kaggle --help`
3. API キーの発行（https://www.kaggle.com/settings）、json ファイルをダウンロード
4. ~/.kaggle ディレクトリを作成し、権限を設定 `mkdir -p ~/.kaggle && chmod 700 ~/.kaggle`
5. ダウンロードした json ファイルを ~/.kaggle に配置し、権限を設定 `chmod 600 ~/.kaggle/kaggle.json`
6. Kaggle API を使ってコンペのカーネルをローカルにダウンロード `kaggle kernels pull root5a/nlp-test`

### プロジェクト構成

```
KaggleChallenge/
├──.vscode/               # VSCodeの設定ファイル
├── doc/                  # ドキュメント関連
├── src/                  # ソースコード
├── requirements.txt      # 依存パッケージ一覧
├── AGENTS.md             # エージェントに対する基本情報
└── README.md             # プロジェクトの概要
```

### 機能一覧
