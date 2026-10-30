"""model training helpers."""

import json
import src.config as config
from src.models.train_test_split import split_best_params
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

    Works on a copy, so the dictionary you pass in is left unchanged.
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
        n_estimators=best["n_estimators"],            # no early stopping in the final fit
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
    """Average amount above the cap among large claims (flat large-loss loading).

    Pass only the rows you fit on, so no oos information leaks in.
    """
    large = df.loc[df[config.TARGET_LARGE] == 1, config.TARGET_INCURRED]
    return float((large - config.CAP).mean())


def fit_bagged_models(df: pd.DataFrame, feature_cols: list, best_severity: dict, best_large: dict):
    """Fit N_SEEDS severity models and N_SEEDS large-loss models on the rows in df."""
    severity_models, large_models = [], []
    for k in range(config.N_SEEDS):
        seed = config.SEED + k

        severity_model = make_severity_model(best_severity, feature_cols, seed)
        severity_model.fit(df[feature_cols], df[config.TARGET_CAPPED])
        severity_models.append(severity_model)

        large_model = make_large_model(best_large, feature_cols, seed)
        large_model.fit(df[feature_cols], df[config.TARGET_LARGE])
        large_models.append(large_model)
    return severity_models, large_models


def predict_bagged(severity_models: list, large_models: list, df: pd.DataFrame,
                   feature_cols: list, mean_excess: float) -> pd.DataFrame:
    """Expected cost = capped severity + P(large) x mean excess above the cap."""
    out = pd.DataFrame(index=df.index)
    out[config.PRED_CAPPED_COL] = np.mean([m.predict(df[feature_cols]) for m in severity_models], axis=0)
    out[config.PROB_LARGE_COL] = np.mean([m.predict_proba(df[feature_cols])[:, 1] for m in large_models], axis=0)
    out[config.PRED_TOTAL_COL] = out[config.PRED_CAPPED_COL] + out[config.PROB_LARGE_COL] * mean_excess
    return out


def train_and_score_oos(df: pd.DataFrame, feature_cols: list, best_severity: dict, best_large: dict):
    """Fit on the train_valid rows only, then predict the oos rows."""
    is_dev = df[config.SPLIT_COL] == config.SPLIT_TRAIN_VALID
    dev, oos = df[is_dev], df[~is_dev]

    severity_models, large_models = fit_bagged_models(dev, feature_cols, best_severity, best_large)
    mean_excess = get_mean_excess_over_cap(dev)
    oos_preds = predict_bagged(severity_models, large_models, oos, feature_cols, mean_excess)
    return severity_models, large_models, mean_excess, oos_preds


def get_feature_importance(severity_models: list) -> pd.Series:
    """Share of total gain per feature, averaged over the seeds."""
    gains = [pd.Series(m.get_booster().get_score(importance_type="total_gain")) for m in severity_models]
    gain = pd.concat(gains, axis=1).fillna(0).mean(axis=1)
    return (gain / gain.sum()).sort_values(ascending=False).rename("gain_share")


def save_models(severity_models: list, large_models: list, folder: str) -> None:
    import os
    os.makedirs(folder, exist_ok=True)
    for k, (sev, clf) in enumerate(zip(severity_models, large_models)):
        sev.save_model(f"{folder}/severity_seed{k}.json")
        clf.save_model(f"{folder}/large_seed{k}.json")



