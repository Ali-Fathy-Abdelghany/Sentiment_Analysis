from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from clean_social.preprocessing.cleaners import (
    build_stopword_set,
    build_symspell,
    correct_spelling_symspell,
    lemmatize_textblob,
    lowercase_text,
    remove_emojis,
    remove_punctuation,
    remove_stopwords_keep_negation,
    remove_urls,
)
from clean_social.preprocessing.tagging import CategoryTagger
from clean_social.utils.io_utils import ensure_required_columns


@dataclass
class PreprocessingConfig:
    text_column: str = "content"
    category_column: str = "category"

    lowercase: bool = False
    remove_urls: bool = False
    remove_emojis: bool = False
    remove_punctuation: bool = False
    remove_stopwords: bool = False
    lemmatize: bool = False
    fix_spelling: bool = False
    extract_tags: bool = False


def preprocess_dataframe(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """Apply optional cleaning layers according to a configuration object."""
    output_df = df.copy()
    ensure_required_columns(output_df, [config.text_column])

    if config.lowercase:
        output_df[config.text_column] = output_df[config.text_column].apply(lowercase_text)

    if config.remove_urls:
        output_df[config.text_column] = output_df[config.text_column].apply(remove_urls)

    if config.remove_emojis:
        output_df[config.text_column] = output_df[config.text_column].apply(remove_emojis)

    if config.remove_punctuation:
        output_df[config.text_column] = output_df[config.text_column].apply(remove_punctuation)

    if config.remove_stopwords:
        stopword_set = build_stopword_set(preserve_negations=True)
        output_df[config.text_column] = output_df[config.text_column].apply(
            lambda value: remove_stopwords_keep_negation(value, stopword_set)
        )

    if config.fix_spelling:
        symspell = build_symspell()
        output_df[config.text_column] = output_df[config.text_column].apply(
            lambda value: correct_spelling_symspell(value, symspell)
        )

    if config.lemmatize:
        output_df[config.text_column] = output_df[config.text_column].apply(lemmatize_textblob)

    if config.extract_tags:
        tagger = CategoryTagger()
        output_df[config.category_column] = output_df[config.text_column].apply(tagger.predict)

    return output_df
