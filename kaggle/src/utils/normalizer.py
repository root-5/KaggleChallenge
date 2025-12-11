import re
import pycountry  # https://github.com/pycountry/pycountry
from nltk.stem import PorterStemmer
from utils.dictionary import Dictionary
import functools


class TextNormalizer:
    @staticmethod
    def normalize_case(text: str) -> str:
        """テキストを小文字に正規化する関数"""
        text = text.lower()  # 小文字に変換
        return text

    @staticmethod
    def remove_newlines(text: str) -> str:
        """改行を除去する関数"""
        text = text.replace("\n", " ").replace("\r", " ")  # 改行の削除
        text = re.sub(r"\s+", " ", text)  # 連続空白を1つに
        text = text.strip()  # 先頭・末尾の空白削除
        return text

    @staticmethod
    def remove_numbers_and_symbols(text: str) -> str:
        """数字と記号を除去する関数"""
        text = re.sub(r"\d", "", text)  # 数字を削除（0-9）
        text = re.sub(r"[^\w\s]", "", text)  # 記号を削除（句読点、特殊文字など）
        text = re.sub(r"_", "", text)  # アンダースコアを削除（\w に含まれるため）
        text = re.sub(r"\s+", " ", text)  # 連続する空白を1つに統合
        text = text.strip()  # 先頭・末尾の空白を削除
        return text

    @staticmethod
    def remove_stopwords(text: str) -> str:
        """ストップワードを除去する関数"""
        stopwords = Dictionary.load_stopwords_set()
        words = text.split()
        filtered_words = [word for word in words if word not in stopwords]
        text = " ".join(filtered_words)
        text = re.sub(r"\s+", " ", text)  # 連続空白を1つに
        text = text.strip()  # 先頭・末尾の空白を削除
        return text

    @staticmethod
    def replace_links(text: str) -> str:
        """URL (http://~, https://~) を置換する関数"""
        text = re.sub(
            r"http[s]?://\S+", "__URLTEXT__", text
        )  # TF-IDF ベクトル化の妨げになるため削除
        return text


class AdditionalNormalizer:
    @staticmethod
    def remove_unreadable_characters(text: str) -> str:
        """ のような読み取り不可能文字を含む単語を削除する関数"""
        text = re.sub(r"\S*\S*", "", text)  # 読み取り不可能文字を含む単語を削除
        text = re.sub(r"\s+", " ", text)  # 連続する空白を1つに統合
        text = text.strip()  # 先頭・末尾の空白を削除
        return text

    @staticmethod
    def remove_mentions(text: str) -> str:
        """メンション (@username) を一つにまとめつつ置換する関数"""
        # すべての @username をプレースホルダーに置換、"__MENTION__" が連続している場合は 1 つにまとめる
        text = re.sub(r"@\w+", "__MENTION__", text)
        text = re.sub(r"(?:__MENTION__\s*)+", "__MENTION__ ", text)
        text = re.sub(r"\s+", " ", text)  # 連続する空白を1つに統合
        text = text.strip()  # 先頭・末尾の空白を削除
        return text

    @staticmethod
    def replace_html_escape(text: str) -> str:
        """HTML エスケープ文字 (&amp;, &lt;, &gt; など) を通常の文字に置換する関数"""
        html_escape_dict = Dictionary.load_html_escapes_dict()
        for escape_seq, char in html_escape_dict.items():
            text = text.replace(escape_seq, char)
        return text

    @staticmethod
    def replace_percent_encording_space(text: str) -> str:
        """パーセントエンコーディングされた空白 (%20) を通常の空白に置換する関数"""
        return text.replace("%20", " ")

    @staticmethod
    def normalize_by_nltk(text: str) -> str:
        """nltk を使って統一的な単語表記で正規化する関数"""
        stemmer = PorterStemmer()
        parts = text.split()
        text = " ".join([stemmer.stem(part) for part in parts])
        return text

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _search_country_fuzzy_cached(part: str) -> str | None:
        """pycountry の fuzzy 検索結果をキャッシュする内部関数"""
        try:
            countries = pycountry.countries.search_fuzzy(part)
        except LookupError:
            return None

        return countries[0].name

    @staticmethod
    def normalize_country_name(text: str) -> str:
        """pycountry を使って統一的な国名表記で正規化する関数"""
        for part in text.split():
            country_name = AdditionalNormalizer._search_country_fuzzy_cached(part)
            if country_name:
                return country_name
        return ""
