# %%
# ライブラリのインポート
import numpy as np  # 線形代数
import pandas as pd  # データ処理、CSVファイルのI/O（例：pd.read_csv）
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, Any

# 独自モジュールのインポート
from modules.preprocessor import Preprocessor

# 定数定義
TRAIN_DATA_PATH = "../input/nlp-getting-started/train.csv"
TEST_DATA_PATH = "../input/nlp-getting-started/test.csv"
SUBMISSION_PATH = "../output/submission.csv"
PREPROCESSED_PATH = "../output/preprocessed.csv"


class Mode:
    TEST_ONCE = "test_once"
    TEST_CROSS_VALIDATION = "test_cross_validation"


CURRENT_MODE = Mode.TEST_CROSS_VALIDATION


def load_data(path: str) -> pd.DataFrame:
    """データの読み込みと基本的な欠損値処理"""
    df = pd.read_csv(path)
    return df.fillna("")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """共通の前処理パイプライン"""
    X = df.copy()
    if "text" in X.columns:
        X["text"] = Preprocessor.normalize_text(X["text"])
    if "keyword" in X.columns:
        X["keyword"] = Preprocessor.normalize_keyword(X["keyword"])
    if "location" in X.columns:
        X["location"] = Preprocessor.normalize_location(X["location"])
    return X


def create_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """特徴量生成（学習データでfitし、学習/テストデータをtransform）"""

    # 1. keyword: CountVectorizer (Bag of Words)
    keyword_vectorizer = CountVectorizer(
        binary=True, tokenizer=lambda x: x.split(), token_pattern=None
    )
    keyword_vectorizer.fit(X_train["keyword"])
    X_train_keyword = keyword_vectorizer.transform(X_train["keyword"]).toarray()
    X_test_keyword = keyword_vectorizer.transform(X_test["keyword"]).toarray()

    # 2. location: OneHotEncoder
    location_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    location_encoder.fit(X_train[["location"]])
    X_train_location = location_encoder.transform(X_train[["location"]])
    X_test_location = location_encoder.transform(X_test[["location"]])

    # 3. text: TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_vectorizer.fit(X_train["text"])
    X_train_text = tfidf_vectorizer.transform(X_train["text"]).toarray()
    X_test_text = tfidf_vectorizer.transform(X_test["text"]).toarray()

    # 結合
    X_train_vec = np.hstack([X_train_text, X_train_keyword, X_train_location])
    X_test_vec = np.hstack([X_test_text, X_test_keyword, X_test_location])

    return X_train_vec, X_test_vec, None


def train_and_evaluate(X, y, mode: str) -> RandomForestClassifier:
    """モデルの学習と評価"""
    clf = RandomForestClassifier(random_state=42)

    if mode == Mode.TEST_ONCE:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"精度 (Test Once): {accuracy:.2f}")

    elif mode == Mode.TEST_CROSS_VALIDATION:
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        print(f"交差検証スコア: {scores}")
        print(f"精度平均: {scores.mean():.2f}")
        # 全データで再学習
        clf.fit(X, y)

    return clf


def main():
    # 1. データの読み込み
    print("データを読み込んでいます...")
    df_train = load_data(TRAIN_DATA_PATH)
    df_test = load_data(TEST_DATA_PATH)

    # 2. 前処理
    print("データの前処理を実行中...")
    X_train_processed = preprocess_data(df_train)
    X_test_processed = preprocess_data(df_test)
    y_train = df_train["target"]

    # 前処理結果の保存 (任意)
    X_train_processed.to_csv(PREPROCESSED_PATH, index=False)

    # 3. 特徴量生成
    print("特徴量を生成中...")
    X_train_vec, X_test_vec, _ = create_features(X_train_processed, X_test_processed)

    # 4. モデルの学習と評価
    print(f"モデルを学習中 ({CURRENT_MODE})...")
    clf = train_and_evaluate(X_train_vec, y_train, CURRENT_MODE)

    # 5. 予測と提出ファイルの作成
    print("予測と提出ファイルの作成中...")
    y_pred_prod = clf.predict(X_test_vec)
    submission = pd.DataFrame({"id": df_test["id"], "target": y_pred_prod})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"提出ファイルを保存しました: {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
