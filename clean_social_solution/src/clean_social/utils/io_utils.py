from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_TEXT_COLUMN = "content"


def load_reviews_csv(input_path: str | Path, text_column: str = DEFAULT_TEXT_COLUMN) -> pd.DataFrame:
    """Load a CSV file and validate the required text column exists."""
    df = pd.read_csv(input_path)
    ensure_required_columns(df, [text_column])
    return df


def ensure_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Raise a clear error if one or more required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a dataframe while creating parent directories when needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
