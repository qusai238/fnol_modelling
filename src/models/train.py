"""model training helpers."""

import json
import src.config as config
import numpy as np
import pandas as pd
import xgboost as xgb


def get_monotone_constraints(feature_cols: list, use_monotone: bool):
    """Tuple of 1 (increasing) / 0 (free), in the same order as feature_cols."""
    if not use_monotone:
        return None
    return tuple(1 if col in config.MONOTONE_INCREASING_COLS else 0 for col in feature_cols)


def split_best_params(best: dict):
    """Separate the tuned settings: (xgboost params, tweedie power, use_monotone).
    """
    params = dict(best["params"])
    power = params.pop("tweedie_variance_power", None)   # only the severity model has it
    use_monotone = params.pop("monotone")
    return params, power, use_monotone


def make_severity_model(best: dict, feature_cols: list, seed: int) -> xgb.XGBRegressor:
    params, power, use_monotone = split_best_params(best)
    return xgb.XGBRegressor(
        objective="reg:tweedie",
        tweedie_variance_power=power,
        n_estimators=best["n_estimators"],
        enable_categorical=True,
        tree_method="hist",
        monotone_constraints=get_monotone_constraints(feature_cols, use_monotone),
        random_state=seed,
        **params,
    )


def make_large_model(best: dict, feature_cols: list, seed: int) -> xgb.XGBClassifier:
    params, _, use_monotone = split_best_params(best)
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=best["n_estimators"],
        enable_categorical=True,
        tree_method="hist",
        monotone_constraints=get_monotone_constraints(feature_cols, use_monotone),
        random_state=seed,
        **params,
    )


def get_mean_excess_over_cap(df: pd.DataFrame) -> float:
    """Average amount above the cap among large claims (flat large-loss excess above the cap).

    Pass only the rows from dev set which the models is fit on, so no oos information leaks in.
    """
    large = df.loc[df[config.TARGET_LARGE] == 1, config.TARGET_INCURRED]
    return float((large - config.CAP).mean())


def fit_models(df: pd.DataFrame, feature_cols: list, best_severity: dict, best_large: dict):
    """Train one severity model and one large-loss model on the rows in df."""
    severity_model = make_severity_model(best_severity, feature_cols, seed=config.SEED)
    severity_model.fit(df[feature_cols], df[config.TARGET_CAPPED])

    large_model = make_large_model(best_large, feature_cols, seed=config.SEED)
    large_model.fit(df[feature_cols], df[config.TARGET_LARGE])
    return severity_model, large_model


def predict_models(severity_model, large_model, df: pd.DataFrame,
                   feature_cols: list, mean_excess: float) -> pd.DataFrame:
    """Predicted claims cost = capped severity prediction + P(large loss model) x mean excess above the cap."""
    out = pd.DataFrame(index=df.index)
    out[config.PRED_CAPPED_COL] = severity_model.predict(df[feature_cols])
    out[config.PROB_LARGE_COL] = large_model.predict_proba(df[feature_cols])[:, 1]
    out[config.PRED_TOTAL_COL] = out[config.PRED_CAPPED_COL] + out[config.PROB_LARGE_COL] * mean_excess
    return out


def train_and_score_oos(df: pd.DataFrame, feature_cols: list, best_severity: dict, best_large: dict):
    """Train on the train_valid rows only, then predict the oos rows."""
    is_dev = df[config.SPLIT_COL] == config.SPLIT_TRAIN_VALID
    dev, oos = df[is_dev], df[~is_dev]

    severity_model, large_model = fit_models(dev, feature_cols, best_severity, best_large)
    mean_excess = get_mean_excess_over_cap(dev)
    oos_preds = predict_models(severity_model, large_model, oos, feature_cols, mean_excess)
    return severity_model, large_model, mean_excess, oos_preds


def save_models(severity_model, large_model, folder: str) -> None:
    import os
    os.makedirs(folder, exist_ok=True)
    severity_model.save_model(f"{folder}/severity.json")
    large_model.save_model(f"{folder}/large_loss.json")



