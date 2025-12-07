# ライブラリのインポート
import time
import numpy as np  # 線形代数
import pandas as pd  # データ処理、CSVファイルのI/O（例：pd.read_csv）
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Any

# 独自モジュールのインポート
from modules.preprocessor import Preprocessor

# 定数定義
TRAIN_DATA_PATH = "kaggle/input/nlp-getting-started/train.csv"
TEST_DATA_PATH = "kaggle/input/nlp-getting-started/test.csv"
SUBMISSION_PATH = "kaggle/output/submission.csv"
PREPROCESSED_PATH = "kaggle/output/preprocessed.csv"


class Mode:
    TEST_ONCE = "簡易テスト"
    TEST_CROSS_VALIDATION = "交差検証"
    SUBMISSION = "提出用"
    TUNING = "チューニング"


CURRENT_MODE = Mode.TEST_CROSS_VALIDATION


def load_data(path: str) -> pd.DataFrame:
    """データの読み込みと基本的な欠損値処理"""
    df = pd.read_csv(path)
    return df.fillna("")


def compress_rare_locations(X: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """location カラムの低頻度カテゴリを "__RARE__" にまとめる関数"""
    X = X.copy()
    if "location" not in X.columns:
        return X

    location_counts = X["location"].value_counts()
    rare_locations = location_counts[location_counts < min_count].index
    X.loc[X["location"].isin(rare_locations), "location"] = "__RARE__"
    return X


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """共通の前処理パイプライン"""
    X = df.copy()
    if "text" in X.columns:
        X["text"] = Preprocessor.normalize_text(X["text"])
    if "keyword" in X.columns:
        X["keyword"] = Preprocessor.normalize_keyword(X["keyword"])
    if "location" in X.columns:
        X["location"] = Preprocessor.normalize_location(X["location"])
        # location のレアカテゴリをまとめる
        X = compress_rare_locations(X, min_count=10)
    return X


class FeatureExtractor:
    def __init__(self):
        self.keyword_vectorizer = CountVectorizer(
            binary=True,  # 出現有無のみを 0/1 で扱う
            tokenizer=lambda x: x.split(),
            token_pattern=None,
            min_df=20,  # レアカテゴリの足切りライン
        )
        self.location_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500)

    def fit(self, X: pd.DataFrame) -> None:
        self.keyword_vectorizer.fit(X["keyword"])
        self.location_encoder.fit(X[["location"]])
        self.tfidf_vectorizer.fit(X["text"])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_keyword = self.keyword_vectorizer.transform(X["keyword"]).toarray()
        X_location = self.location_encoder.transform(X[["location"]])
        X_text = self.tfidf_vectorizer.transform(X["text"]).toarray()
        return np.hstack([X_text, X_keyword, X_location])


def train_and_evaluate(X, y, mode: str) -> RandomForestClassifier:
    """モデルの学習と評価"""

    # ハイパーパラメータチューニングモード
    if mode == Mode.TUNING:
        print("ハイパーパラメータチューニングを開始します...")
        # ベースとなるモデル
        base_clf = RandomForestClassifier(random_state=42, n_jobs=1)

        # 探索するパラメータの範囲
        param_grid = {
            "n_estimators": [300, 500],
            "max_depth": [None, 20, 50],  # None（無制限）も含めて比較する
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt"],
        }

        # グリッドサーチの設定
        grid_search = GridSearchCV(
            estimator=base_clf,
            param_grid=param_grid,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,  # 並列処理
            verbose=1,  # 進捗表示
        )
        grid_search.fit(X, y)

        print(f"最良パラメータ: {grid_search.best_params_}")
        print(f"最良スコア: {grid_search.best_score_:.8f}")
        return grid_search.best_estimator_

    # 通常の学習・評価モード
    # n_jobs=-1 ですべてのCPUコアを使用
    clf = RandomForestClassifier(
        max_depth=20,  # 木の深さ、真っ先に変更を試みる対象
        min_samples_leaf=1,  # 葉ノードの最小サンプル数、真っ先に変更を試みる対象
        n_estimators=500,  # 決定木の数、増やせば大抵精度向上するが、計算コストと相談
        max_features="sqrt",  # 各決定木で使用する特徴量の数、基本デフォルトで十分だが特徴量多い（数千～）場合は調整を検討
        class_weight=None,  # クラス（モデルの予測対象）が不均衡な場合に有効
        random_state=42,
        n_jobs=-1,
    )

    if mode == Mode.TEST_ONCE:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"精度 (Test Once): {accuracy:.8f}")

    elif mode == Mode.TEST_CROSS_VALIDATION or mode == Mode.SUBMISSION:
        # cross_val_score 自体にも n_jobs を指定して並列評価する
        scores = cross_val_score(
            clf,
            X,
            y,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,  # 全コア使用
        )
        print(f"交差検証スコア: {scores}")
        print(f"精度平均: {scores.mean():.8f}")

        if mode == Mode.SUBMISSION:
            # 全データで再学習
            clf.fit(X, y)

    return clf


def main():
    # 時間計測開始
    start_time = time.time()

    # 1. データの読み込み
    print("データを読み込んでいます...")
    df_train = load_data(TRAIN_DATA_PATH)

    df_test = None
    if CURRENT_MODE == Mode.SUBMISSION:
        df_test = load_data(TEST_DATA_PATH)

    # 2. 前処理
    print("データの前処理を実行中...")
    X_train_processed = preprocess_data(df_train)

    X_test_processed = None
    if CURRENT_MODE == Mode.SUBMISSION:
        X_test_processed = preprocess_data(df_test)

    y_train = df_train["target"]

    # 前処理結果の保存 (任意)
    X_train_processed.to_csv(PREPROCESSED_PATH, index=False)

    # 3. 特徴量生成
    print("特徴量を生成中...")
    extractor = FeatureExtractor()
    extractor.fit(X_train_processed)
    X_train_vec = extractor.transform(X_train_processed)

    X_test_vec = None
    if CURRENT_MODE == Mode.SUBMISSION:
        X_test_vec = extractor.transform(X_test_processed)

    # 4. モデルの学習と評価
    print(f"モデルを学習中 ({CURRENT_MODE})...")
    clf = train_and_evaluate(X_train_vec, y_train, CURRENT_MODE)

    # 5. 予測と提出ファイルの作成
    if CURRENT_MODE == Mode.SUBMISSION:
        print("予測と提出ファイルの作成中...")
        y_pred_prod = clf.predict(X_test_vec)
        submission = pd.DataFrame({"id": df_test["id"], "target": y_pred_prod})
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"提出ファイルを保存しました: {SUBMISSION_PATH}")

    # 時間計測終了
    elapsed_time = time.time() - start_time
    print(f"処理時間: {elapsed_time:.2f} 秒")


if __name__ == "__main__":
    main()
