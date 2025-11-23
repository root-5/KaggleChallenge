import re
import pycountry  # https://github.com/pycountry/pycountry


class TextNormalizer:
    @staticmethod
    def remove_newlines(text: str) -> str:
        """改行を除去する関数"""
        text = text.replace("\n", " ").replace("\r", " ")  # 改行の削除
        text = re.sub(r"\s+", " ", text)  # 連続空白を1つに
        text = text.strip()  # 先頭・末尾の空白削除
        return text

    @staticmethod
    def replace_links(text: str) -> str:
        """URL (http://~, https://~) を [LINK_URL] に置換する関数"""
        text = re.sub(r"http[s]?://\S+", "[LINK_URL]", text)
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


class AdditionalNormalizer:
    @staticmethod
    def normalize_country_name(text: str) -> str:
        """pycountry を使って統一てきな国名表記で正規化する関数"""
        # 一旦、州・市名までの判定はあきらめ国名までの実装としている

        # 文字列を検索用に分割
        parts = text.split()

        # 純粋に国名を検索
        for part in parts:
            try:
                countries = pycountry.countries.search_fuzzy(part)
                if countries:
                    country = next(iter(countries), None)
                    return country.name
            except LookupError:
                continue

        # 国名でマッチしなかった場合は州・市名でマッチを試みる
        for part in parts:
            try:
                subdivisions = pycountry.subdivisions.partial_match(part)
                if subdivisions:
                    subdivision = next(iter(subdivisions), None)
                    country = pycountry.countries.search_fuzzy(
                        alpha_2=subdivision.country_code
                    )
                    return country.name
            except LookupError:
                continue

        return ""  # マッチしなかった場合
