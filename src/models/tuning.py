""" tuning helpers"""

import src.config as config
from src.models.train_test_split import get_cv_folds, get_baseline_deviance
import json

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, mean_tweedie_deviance, roc_auc_score

def get_monotone_constraints(feature_cols: list, use_monotone: bool):
    """Tuple of 1 (increasing) / 0 (free), in the same order as feature_cols."""
    if not use_monotone:
        return None
    return tuple(1 if col in config.MONOTONE_INCREASING_COLS else 0 for col in feature_cols)


def suggest_common_params(trial: optuna.Trial, min_child_weight_range: tuple, depth_range: tuple) -> dict:
    """Search space shared by both models."""
    return dict(
        max_depth=trial.suggest_int("max_depth", *depth_range),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", *min_child_weight_range, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("gamma", 0.0, 5.0),
    )


def get_all_valid_index(folds: list) -> np.ndarray:
    """All validation row labels, pooled across the folds."""
    return np.concatenate([valid_idx for _, valid_idx in folds])


def get_baseline_logloss(df: pd.DataFrame, folds: list) -> float:
    """Log-loss of predicting the expanding-window training prevalence (benchmark for the classifier)."""
    valid_index = get_all_valid_index(folds)
    prior = pd.Series(index=valid_index, dtype=float)
    for train_idx, valid_idx in folds:
        prior[valid_idx] = df.loc[train_idx, config.TARGET_COLS[2]].mean()
    return float(log_loss(df.loc[valid_index, config.TARGET_COLS[2]], prior))


def make_severity_objective(df: pd.DataFrame, feature_cols: list, folds: list):
    """Optuna objective for the severity model (pooled out-of-fold Tweedie deviance, power 1.5)."""
    y = df[config.TARGET_CAPPED]
    valid_index = get_all_valid_index(folds)

    def objective(trial: optuna.Trial) -> float:
        power = trial.suggest_float("tweedie_variance_power", *config.TWEEDIE_POWER_RANGE)
        use_monotone = trial.suggest_categorical("monotone", [True, False])
        params = suggest_common_params(trial, config.SEVERITY_MIN_CHILD_WEIGHT_RANGE, config.SEVERITY_DEPTH_RANGE)

        oof = pd.Series(index=valid_index, dtype=float)    # out-of-fold predictions
        best_iters = []
        for train_idx, valid_idx in folds:
            model = xgb.XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=power,
                eval_metric=f"tweedie-nloglik@{power:.3f}",
                n_estimators=config.MAX_TREES,
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                enable_categorical=True,
                tree_method="hist",
                monotone_constraints=get_monotone_constraints(feature_cols, use_monotone),
                random_state=config.SEED,
                **params,
            )
            model.fit(
                df.loc[train_idx, feature_cols], y[train_idx],
                eval_set=[(df.loc[valid_idx, feature_cols], y[valid_idx])],
                verbose=False,
            )
            oof[valid_idx] = model.predict(df.loc[valid_idx, feature_cols])
            best_iters.append(int(model.best_iteration) + 1)

        trial.set_user_attr("best_iters", best_iters)
        # fixed power for scoring, so trials with different Tweedie powers are comparable
        return mean_tweedie_deviance(
            y[valid_index], oof.clip(lower=1e-3), power=config.TWEEDIE_EVAL_POWER
        )

    return objective


def make_large_objective(df: pd.DataFrame, feature_cols: list, folds: list):
    """Optuna objective for the large-loss classifier (pooled out-of-fold log-loss)."""
    y = df[config.TARGET_LARGE]
    valid_index = get_all_valid_index(folds)

    def objective(trial: optuna.Trial) -> float:
        use_monotone = trial.suggest_categorical("monotone", [True, False])
        params = suggest_common_params(trial, config.LARGE_MIN_CHILD_WEIGHT_RANGE, config.LARGE_DEPTH_RANGE)

        oof = pd.Series(index=valid_index, dtype=float)
        best_iters = []
        for train_idx, valid_idx in folds:
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_estimators=config.MAX_TREES,
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
                enable_categorical=True,
                tree_method="hist",
                monotone_constraints=get_monotone_constraints(feature_cols, use_monotone),
                random_state=config.SEED,
                **params,
            )
            model.fit(
                df.loc[train_idx, feature_cols], y[train_idx],
                eval_set=[(df.loc[valid_idx, feature_cols], y[valid_idx])],
                verbose=False,
            )
            oof[valid_idx] = model.predict_proba(df.loc[valid_idx, feature_cols])[:, 1]
            best_iters.append(int(model.best_iteration) + 1)

        trial.set_user_attr("best_iters", best_iters)
        trial.set_user_attr("oof_auc", float(roc_auc_score(y[valid_index], oof)))
        return float(log_loss(y[valid_index], oof.clip(1e-6, 1 - 1e-6)))

    return objective


def run_study(objective, n_trials: int, baseline: float, metric_name: str) -> dict:
    """Run Optuna and return the best settings in one dictionary."""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    mean_iters = float(np.mean(best.user_attrs["best_iters"]))
    result = {
        "metric": metric_name,
        "cv_score": best.value,
        "cv_baseline_score": baseline,
        "params": dict(best.params),
        "n_estimators": int(round(mean_iters * config.FINAL_TREES_MULTIPLIER)),
        "cv_best_iters": best.user_attrs["best_iters"],
    }
    if "oof_auc" in best.user_attrs:
        result["cv_oof_auc"] = best.user_attrs["oof_auc"]

    print(f"{metric_name}: baseline {baseline:.4f} -> best {best.value:.4f} "
          f"(n_estimators = {result['n_estimators']}, fold iterations = {result['cv_best_iters']})")
    return result


def tune_severity(df: pd.DataFrame, feature_cols: list, n_trials: int = config.N_TRIALS) -> dict:
    folds = get_cv_folds(df)
    baseline = get_baseline_deviance(df)
    objective = make_severity_objective(df, feature_cols, folds)
    return run_study(objective, n_trials, baseline, "tweedie_deviance_p1.5")


def tune_large(df: pd.DataFrame, feature_cols: list, n_trials: int = config.N_TRIALS) -> dict:
    folds = get_cv_folds(df)
    baseline = get_baseline_logloss(df, folds)
    objective = make_large_objective(df, feature_cols, folds)
    return run_study(objective, n_trials, baseline, "logloss")


def save_best_params(result: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=2)