import pandas as pd 
import numpy as np
import src.config as config


def add_split_column(df: pd.DataFrame) -> pd.DataFrame:
    """Label each row as train_valid (loss date <= DEV_END) or oos (later)."""
    df = df.copy()
    is_dev = df[config.DATE_COL] <= pd.Timestamp(config.DEV_END)
    df[config.SPLIT_COL] = np.where(is_dev, config.SPLIT_TRAIN_VALID, config.SPLIT_OOS)
    return df


def get_cv_folds(df: pd.DataFrame):
    """Expanding-window folds on the train_valid rows only.

    Fold for year Y: train on loss years before Y, validate on year Y.
    Returns a list of (train_index, valid_index) pairs. oos rows are never used.
    """
    is_dev = df[config.SPLIT_COL] == config.SPLIT_TRAIN_VALID
    loss_year = df[config.DATE_COL].dt.year

    folds = []
    for year in config.CV_VALID_YEARS:
        train_idx = df.index[is_dev & (loss_year < year)]
        valid_idx = df.index[is_dev & (loss_year == year)]
        folds.append((train_idx, valid_idx))
    return folds


def print_split_summary(df: pd.DataFrame) -> None:
    """Row counts and date range per split, plus the size of each CV fold."""
    print(df.groupby(config.SPLIT_COL)[config.DATE_COL].agg(rows="size", first="min", last="max"))
    for i, (train_idx, valid_idx) in enumerate(get_cv_folds(df), start=1):
        print(f"fold {i}: train {len(train_idx):,} rows, valid {len(valid_idx):,} rows")