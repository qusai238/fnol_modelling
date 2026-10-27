"""Data loading utilities."""

from pathlib import Path

import pandas as pd


def load_excel(file_path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load an Excel sheet into a pandas DataFrame.

    Args:
        file_path: Path to `.xlsx`/`.xls` file.
        sheet_name: Sheet name or index. Use `None` for all sheets.

    Returns:
        Loaded dataframe (or dict of dataframes when `sheet_name=None`).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_excel(file_path, sheet_name=sheet_name)
