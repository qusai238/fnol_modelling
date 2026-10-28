"""Feature engineering functions."""

import pandas as pd
import numpy as np
import src.config as config 


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean baseline feature frame.

    This is a placeholder pass-through transform to be extended as
    case-study features are defined.
    """
    return df.copy()


def build_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar, cyclical and UK season features from datetime date_of_loss and time_hour."""
    df = df.copy()
    date = df["date_of_loss"]
    df["loss_month"] = date.dt.month
    df["loss_day_of_week"] = date.dt.day_of_week
    df["loss_day_of_year"] = date.dt.day_of_year
    df["loss_is_weekend"] = (date.dt.day_of_week >= 5).astype("int8")

    cycles = {
        "loss_hour": (df["time_hour"], 24),
        "loss_month": (df["loss_month"], 12),
        "loss_day_of_week": (df["loss_day_of_week"], 7),
        "loss_day_of_year": (df["loss_day_of_year"], 365),
    }
    for feature, (values, period) in cycles.items():
        df[f"{feature}_sin"] = np.sin(2 * np.pi * values / period)
        df[f"{feature}_cos"] = np.cos(2 * np.pi * values / period)

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    df["loss_season"] = df["loss_month"].map(season_map)
    return df


############################ aggregate features functions ############################
def add_tp_n_total(df: pd.DataFrame) -> pd.DataFrame:
    """Total number of third parties = known types + 'not known' type."""
    df = df.copy()
    known = df[config.TP_TYPE_KNOWN_COLS].sum(axis=1)
    df["tp_n_total"] = known + df[config.TP_TYPE_UNKNOWN_COL]
    return df

def add_inj_flagged_n(df: pd.DataFrame) -> pd.DataFrame:
    """Number of injuries flagged as whiplash, traumatic or fatality."""
    df = df.copy()
    df["inj_flagged_n"] = df[config.INJURY_FLAGGED_COLS].sum(axis=1)
    return df


def add_inj_severity_ord(df: pd.DataFrame) -> pd.DataFrame:
    """Worst injury on the claim as an ordinal: 0 none, 1 whiplash, 2 traumatic, 3 fatality."""
    df = df.copy()
    conditions = [
        df[config.INJURY_FATALITY_COL] > 0,
        df[config.INJURY_TRAUMATIC_COL] > 0,
        df[config.INJURY_WHIPLASH_COL] > 0,
    ]
    values = [
        config.INJURY_SEVERITY_ORDER["fatality"],
        config.INJURY_SEVERITY_ORDER["traumatic"],
        config.INJURY_SEVERITY_ORDER["whiplash"],
    ]
    df["inj_severity_ord"] = np.select(conditions, values, default=config.INJURY_SEVERITY_ORDER["none"])
    return df


def add_inj_per_tp(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged injuries per third party. Needs tp_n_total and inj_flagged_n first."""
    df = df.copy()
    safe_total = df["tp_n_total"].replace(0, np.nan)
    df["inj_per_tp"] = (df["inj_flagged_n"] / safe_total).fillna(0)
    return df


def add_grouped_regions(df: pd.DataFrame) -> pd.DataFrame:
    """Sum the 12 TP region counts into 7 broader groups (tp_reg_london, tp_reg_south, ...)."""
    df = df.copy()
    for group in sorted(set(config.TP_REGION_GROUP_MAP.values())):
        group_cols = [
            config.TP_REGION_PREFIX + region
            for region, grp in config.TP_REGION_GROUP_MAP.items()
            if grp == group
        ]
        df[f"tp_reg_{group}"] = df[group_cols].sum(axis=1)
    return df


def build_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all aggregate steps in the right order (inj_per_tp needs the two before it)."""
    df = add_tp_n_total(df)
    df = add_inj_flagged_n(df)
    df = add_inj_severity_ord(df)
    df = add_inj_per_tp(df)
    df = add_grouped_regions(df)
    return df
############################ end of aggregate features ############################