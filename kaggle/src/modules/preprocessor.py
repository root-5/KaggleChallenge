import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.normalizer import TextNormalizer, AdditionalNormalizer


class Preprocessor:
    @staticmethod
    def normalize_text(X_raw_series: pd.Series) -> pd.Series:
        """text カラムの前処理を行うメソッド"""
        X_processed_series = X_raw_series.copy()
        X_processed_series = (
            X_raw_series.apply(TextNormalizer.normalize_case)
            .apply(TextNormalizer.remove_newlines)
            .apply(TextNormalizer.remove_stopwords)
            .apply(TextNormalizer.remove_links)
            .apply(AdditionalNormalizer.remove_unreadable_characters)
            .apply(AdditionalNormalizer.remove_mentions)
            .apply(AdditionalNormalizer.replace_html_escape)
            .apply(
                TextNormalizer.remove_numbers_and_symbols
            )  # @mention や &amp; などに変換・削除してから記号削除をする必要があるので最後に実行
        )
        return X_processed_series

    @staticmethod
    def normalize_location(X_raw_series: pd.Series) -> pd.Series:
        """location カラムの前処理を行うメソッド"""
        X_processed_series = X_raw_series.copy()
        X_processed_series = (
            X_raw_series.apply(TextNormalizer.remove_numbers_and_symbols)
            .apply(AdditionalNormalizer.replace_percent_encording_space)
            .apply(
                AdditionalNormalizer.normalize_country_name
            )  # すべての正規化後に実行
        )
        return X_processed_series
