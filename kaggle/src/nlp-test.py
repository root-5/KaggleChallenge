# %%
# ==========================================
# 各種ライブラリと CSV データの読み込み
# ==========================================
# 自動リロードを有効にする設定
# %load_ext autoreload
# %autoreload 2

# 必要なライブラリのインストール
# %pip install pycountry

# ライブラリのインポート
import numpy as np  # 線形代数
import pandas as pd  # データ処理、CSVファイルのI/O（例：pd.read_csv）
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

# 独自モジュールのインポート
from modules.preprocessor import Preprocessor

# CSVデータを pandas データフレームオブジェクトとして読み込み
df = pd.read_csv("../input/nlp-getting-started/train.csv")
df = df.fillna("")  # 空のカラムを空文字に置換


# %%
# ==========================================
# 前処理 > データクリーニング（Data Cleaning）
# ==========================================
# 元データをコピーして、特徴量とターゲットに分割
X_raw = df[["text", "keyword", "location"]].copy()
y = df["target"].copy()  # ターゲット変数

# "text" カラムと "location" カラムのデータクリーニング
X_processed = X_raw.copy()
X_processed["text"] = Preprocessor.normalize_text(X_raw["text"])
X_processed["location"] = Preprocessor.normalize_location(X_raw["location"])

# 前処理後のデータをもと csv の各カラムを置換する形で csv 出力
csv_df = df.copy()
csv_df["text"] = X_processed["text"]
csv_df["location"] = X_processed["location"]
csv_df.to_csv("../output/preprocessed.csv", index=False)


# %%
# ==========================================
# 前処理 > データ変換（Data Transformation）
# ==========================================
# カテゴリデータのワンホットベクトル変換を定義
# handle_unknown='ignore' により未知のカテゴリは無視（オール0）に、sparse_output=False により密行列 (numpy array) を返させる
categorical_cols = ["keyword", "location"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(X_processed[categorical_cols])

# "text" カラムのテキストデータの TF-IDF ベクトル変換を定義
unstructured_col = "text"
vectorizer = TfidfVectorizer(max_features=1000)
vectorizer.fit(X_processed[unstructured_col])

# ベクトル変換の実行
X_cat_encoded = encoder.transform(X_processed[categorical_cols])
X_text_vectorized = vectorizer.transform(X_processed[unstructured_col]).toarray()


# %%
# ==========================================
# 前処理 > 構造演算（Structure Operations）
# ==========================================
# 結合して最終的な特徴量行列を作成、pandas は表形式 (二次元) までしか扱えないため numpy で変換しなおす
X = np.hstack([X_text_vectorized, X_cat_encoded])


# %%
# ==========================================
# モデルの学習と評価
# ==========================================
# テストモードの切替
MODES = ["test_once", "test_cross_validation"]
MODE = MODES[1]  # 使用するモードを選択

# モデルの訓練、予測、精度の計算
clf = RandomForestClassifier(random_state=42)
if MODE == "test_once":
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

elif MODE == "test_cross_validation":
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(scores)
    accuracy = scores.mean()

print(f"Accuracy: {accuracy:.2f}")


# %%
# ==========================================
# 本番データに対する予測
# ==========================================
# test.csv に対して予測を行う
df_prod = pd.read_csv("../input/nlp-getting-started/test.csv")
df_prod = df_prod.fillna("")
X_prod = df_prod[["text", "keyword", "location"]].copy()

# 前処理
X_prod_processed = X_prod.copy()
X_prod_processed["text"] = Preprocessor.normalize_text(X_prod["text"])
X_prod_processed["location"] = Preprocessor.normalize_location(X_prod["location"])

# ベクトル変換の実行
X_prod_cat_encoded = encoder.transform(X_prod_processed[categorical_cols])
X_prod_text_vectorized = vectorizer.transform(X_prod_processed[unstructured_col]).toarray()

# 結合して最終的な特徴量行列を作成
X_prod = np.hstack((X_prod_text_vectorized, X_prod_cat_encoded))

# 本番予測用に全データで再学習を行ってから予測
clf.fit(X, y)
y_pred_prod = clf.predict(X_prod)

# 提出用ファイルの作成
submission = pd.DataFrame({"id": df_prod["id"], "target": y_pred_prod})
submission.to_csv("../output/submission.csv", index=False)
