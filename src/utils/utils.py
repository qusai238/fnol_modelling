"""Utilities for cleaning FNOL data."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import src.config as config


_ABNORMAL_ENTRY = re.compile(r"^\s*(?:N/K|#)?\s*$", re.IGNORECASE)


def replace_abnormal_entries(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""Return a copy with N/K, #, and whitespace-only entries replaced by NaN."""
	cleaned = dataframe.copy()
	for column in cleaned.columns:
		cleaned[column] = cleaned[column].replace(
			_ABNORMAL_ENTRY, np.nan, regex=True
		)
	return cleaned


def save_preprocessed_data(
	dataframe: pd.DataFrame, filename: str = "preprocessed.csv"
) -> Path:
	"""Save a DataFrame as CSV in data/preporcessed and return its path."""
	output_path = config.DATA_DIR / "preporcessed" / filename
	output_path.parent.mkdir(parents=True, exist_ok=True)
	dataframe.to_csv(output_path, index=False)
	return output_path
