"""Feature engineering functions."""

import pandas as pd
import numpy as np
import src.config as config 


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


############################ aggregate features functions(tp, inj, region) ############################
def add_tp_n_total(df: pd.DataFrame) -> pd.DataFrame:
    """Total number of third parties = known types + 'not known' type."""
    df = df.copy()
    known = df[config.TP_TYPE_KNOWN_COLS].sum(axis=1)
    df["tp_n_total"] = known + df[config.TP_TYPE_UNKNOWN_COL]
    return df

def add_inj_flagged_n(df: pd.DataFrame) -> pd.DataFrame:
    """Number of injuries flagged as whiplash, traumatic or fatality."""
    df = df.copy()
    df[config.INJURY_FLAGGED_COL] = df[config.INJURY_FLAGGED_COLS].sum(axis=1)
    return df


def add_inj_known_share(df: pd.DataFrame) -> pd.DataFrame:
    """Share of injury records that are flagged (whiplash/traumatic/fatality).

    flagged / (flagged + unclear + not known). Needs inj_flagged_n first.
    A low value means most injuries on the claim are still unclear or unknown.
    """
    df = df.copy()
    total_injury_records = (
        df[config.INJURY_FLAGGED_COL] + df[config.INJURY_UNCLEAR_COL] + df[config.INJURY_UNKNOWN_COL]
    )
    # Avoid dividing by zero: where there are no injury records the share is 0
    safe_total = total_injury_records.replace(0, np.nan)
    df["inj_known_share"] = (df[config.INJURY_FLAGGED_COL] / safe_total).fillna(0)
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
    df["inj_per_tp"] = (df[config.INJURY_FLAGGED_COL] / safe_total).fillna(0)
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
    df = add_inj_known_share(df)
    df = add_inj_severity_ord(df)
    df = add_inj_per_tp(df)
    df = add_grouped_regions(df)
    return df
############################ end of aggregate features ############################

############################ Target-related functions ############################
def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ids + features + targets, set categorical dtypes and sort by date."""
    df = df.copy()
    df = df.rename(columns={"claim number": "claim_number", "vechile_registration_present": "vehicle_registration_present"}) 

    missing = [c for c in config.FINAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[config.FINAL_COLS].copy()
    for col in config.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")                    # NaN stays NaN

    # sorted by date, with a clean unique index for the CV folds
    return df.sort_values(["date_of_loss", "claim_number"]).reset_index(drop=True)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create the three target columns from incurred.

    Negative incurred = recoveries, so it is floored at 0.
    """
    df = df.copy()
    floored = df[config.INCURRED_COL].clip(lower=0)

    df[config.TARGET_INCURRED] = floored                                  # no cap, for total-cost checks
    df[config.TARGET_CAPPED] = floored.clip(upper=config.CAP)                    # severity target (in pounds)
    df[config.TARGET_LARGE] = (df[config.INCURRED_COL] > config.CAP).astype(int)        # large-loss flag
    return df


def split_dev_test(df: pd.DataFrame):
    """Out-of-time split: development years vs the later untouched test years."""
    is_dev = df[config.DATE_COL] <= pd.Timestamp(config.DEV_END)
    return df[is_dev].copy(), df[~is_dev].copy()


def get_mean_excess_over_cap(dev_df: pd.DataFrame) -> float:
    """Average amount above the cap among large claims (flat large-loss loading).

    Use the development data only, so no test information leaks in.
    """
    large_claims = dev_df.loc[dev_df[config.TARGET_LARGE] == 1, config.TARGET_INCURRED]
    return float((large_claims - config.CAP).mean())


def combine_predictions(pred_capped, prob_large, mean_excess: float):
    """Expected claim cost = capped severity + P(large) x mean excess above the cap."""
    return pred_capped + prob_large * mean_excess


def print_target_summary(df: pd.DataFrame) -> None:
    """Quick sanity check on the targets."""
    n_negative = int((df[config.INCURRED_COL] < 0).sum())
    zero_share = (df[config.TARGET_INCURRED] == 0).mean()
    n_large = int(df[config.TARGET_LARGE].sum())
    large_cost_share = df.loc[df[config.TARGET_LARGE] == 1, config.TARGET_INCURRED].sum() / df[config.TARGET_INCURRED].sum()

    print(f"Negative incurred floored to 0: {n_negative}")
    print(f"Share of claims with zero cost: {zero_share:.1%}")
    print(f"Large claims (> {config.CAP:,}): {n_large} ({n_large / len(df):.1%})")
    print(f"Large claims' share of total cost: {large_cost_share:.1%}")


def set_categorical_dtypes(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """Cast text columns to the category dtype so XGBoost can use them. NaN stays NaN."""
    df = df.copy()
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    return df


def check_feature_dtypes(df: pd.DataFrame, feature_cols: list) -> None:
    """Print any feature column XGBoost cannot read (anything other than int, float, bool or category)."""
    allowed = ["int", "float", "bool", "category"]
    bad = [c for c in feature_cols if not any(df[c].dtype.name.startswith(a) for a in allowed)]
    print(f"Columns with unsupported dtypes: {bad}")