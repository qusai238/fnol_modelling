import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, mean_tweedie_deviance,
                             r2_score, roc_auc_score)

import src.config as config
from src.models.train import predict_models, make_severity_model, make_large_model, get_mean_excess_over_cap
from src.models.train_test_split import get_cv_folds


# severity metrics 
SEVERITY_TABLE_METRICS = [
    ("Tweedie deviance (power 1.5)", "tweedie_deviance"),
    ("Baseline deviance", "baseline_tweedie_deviance"),
    ("Deviance improvement vs baseline", "deviance_improvement"),
    ("MAE (£)", "mae"),
    ("R2", "r2"),
    ("Normalised Gini", "gini"),
    ("Actual / expected", "actual_over_expected"),
]

# large loss classifier metrics
LARGE_TABLE_METRICS = [
    ("AUC", "auc"),
    ("Brier skill score", "brier_skill_score"),
    ("Log-loss", "logloss"),
    ("Actual large claims", "n_large_actual"),
    ("Predicted large claims", "n_large_predicted"),
    ("ECE", "ece"),
]


def build_eval_frame(rows: pd.DataFrame, preds: pd.DataFrame, base_capped: float,
                     base_total: float, base_prevalence: float) -> pd.DataFrame:
    """Build an evaluation frame and add predictions to it."""
    out = rows.join(preds)
    pred_cols = [config.PRED_CAPPED_COL, config.PROB_LARGE_COL, config.PRED_TOTAL_COL]
    out[pred_cols] = out[pred_cols].astype("float64")
    out[config.PRED_BASELINE_COL] = base_capped
    out[config.PRED_BASELINE_TOTAL_COL] = base_total
    out[config.PREV_BASELINE_COL] = base_prevalence
    out["injury_flagged"] = np.where(out["inj_severity_ord"] > 0, "injury_flagged", "no_injury_flagged")
    out["loss_year"] = out[config.DATE_COL].dt.year
    return out


def get_train_frame(df: pd.DataFrame, feature_cols: list, severity_model, large_model,
                    mean_excess: float) -> pd.DataFrame:
    """Final models scored on the rows they were fitted on (in-sample, so optimistic)."""
    dev = df[df[config.SPLIT_COL] == config.SPLIT_TRAIN_VALID]
    preds = predict_models(severity_model, large_model, dev, feature_cols, mean_excess)
    return build_eval_frame(dev, preds, dev[config.TARGET_CAPPED].mean(),
                            dev[config.TARGET_INCURRED].mean(), dev[config.TARGET_LARGE].mean())


def get_validation_frames(df: pd.DataFrame, feature_cols: list, best_severity: dict, best_large: dict) -> dict:
    """Out-of-fold predictions for each expanding-window validation year.

    Each fold refits the tuned settings on the earlier years only, using the number of trees
    that early stopping chose for that fold during tuning (best['cv_best_iters']).
    Returns {'valid_2011': frame, ...}.
    """
    frames = {}
    for k, (train_idx, valid_idx) in enumerate(get_cv_folds(df)):
        train, valid = df.loc[train_idx], df.loc[valid_idx]
        year = config.CV_VALID_YEARS[k]

        severity_model = make_severity_model(
            {**best_severity, "n_estimators": best_severity["cv_best_iters"][k]}, feature_cols, seed=config.SEED)
        severity_model.fit(train[feature_cols], train[config.TARGET_CAPPED])
        large_model = make_large_model(
            {**best_large, "n_estimators": best_large["cv_best_iters"][k]}, feature_cols, seed=config.SEED)
        large_model.fit(train[feature_cols], train[config.TARGET_LARGE])

        mean_excess = get_mean_excess_over_cap(train)
        preds = predict_models(severity_model, large_model, valid, feature_cols, mean_excess)
        frames[f"valid_{year}"] = build_eval_frame(valid, preds, train[config.TARGET_CAPPED].mean(),
                                                   train[config.TARGET_INCURRED].mean(), train[config.TARGET_LARGE].mean())
    return frames


def get_split_comparison_tables(frames: dict):
    """Severity and large-loss metrics side by side for every frame (columns = frame names)."""
    severity, large = {}, {}
    for name, frame in frames.items():
        m = get_severity_metrics(frame)
        m = m.iloc[0] if isinstance(m, pd.DataFrame) else m      # one-row DataFrame -> plain values
        severity[name] = {label: m[key] for label, key in SEVERITY_TABLE_METRICS}

        # the baseline prevalence differs by fold, so use the average baseline in the frame
        m = get_large_metrics(frame, dev_prevalence=frame[config.PREV_BASELINE_COL].mean())
        m = m.iloc[0] if isinstance(m, pd.DataFrame) else m      # one-row DataFrame -> plain values
        large[name] = {label: m[key] for label, key in LARGE_TABLE_METRICS}
    return pd.DataFrame(severity), pd.DataFrame(large)


def build_oos_frame(df: pd.DataFrame, oos_preds: pd.DataFrame) -> pd.DataFrame:
    """oos rows + predictions + baselines + segment helpers."""
    is_dev = df[config.SPLIT_COL] == config.SPLIT_TRAIN_VALID
    oos = df[~is_dev].join(oos_preds)
    pred_cols = [config.PRED_CAPPED_COL, config.PROB_LARGE_COL, config.PRED_TOTAL_COL]
    oos[pred_cols] = oos[pred_cols].astype("float64")      # XGBoost returns float32; sums need float64
    oos[config.PRED_BASELINE_COL] = df.loc[is_dev, config.TARGET_CAPPED].mean()
    oos[config.PRED_BASELINE_TOTAL_COL] = df.loc[is_dev, config.TARGET_INCURRED].mean()
    oos["injury_flagged"] = np.where(oos["inj_severity_ord"] > 0, "injury_flagged", "no_injury_flagged")
    oos["loss_year"] = oos[config.DATE_COL].dt.year
    return oos

# severity metrics
def get_gini(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Normalised Gini of the ranking on £ actuals (0 = random, 1 = perfect)."""
    def area(order_by):
        order = np.argsort(order_by, kind="stable")
        lorenz = np.cumsum(actual[order]) / actual.sum()
        x = np.arange(1, len(actual) + 1) / len(actual)
        return float(np.trapezoid(np.r_[0, lorenz], np.r_[0, x]))
    return (0.5 - area(predicted)) / (0.5 - area(actual))


def get_severity_metrics(oos: pd.DataFrame) -> dict:
    y = oos[config.TARGET_CAPPED]
    pred = oos[config.PRED_CAPPED_COL].clip(lower=1e-3)
    base = oos[config.PRED_BASELINE_COL]
    power = config.TWEEDIE_EVAL_POWER
    ya, pa, ba = y.to_numpy(), pred.to_numpy(), base.to_numpy()

    dev, dev_base = mean_tweedie_deviance(y, pred, power=power), mean_tweedie_deviance(y, base, power=power)
    dict = {
        "tweedie_deviance": dev,
        "baseline_tweedie_deviance": dev_base,
        "deviance_improvement": 1 - dev / dev_base,
        "mae": mean_absolute_error(y, pred),
        "baseline_mae": mean_absolute_error(y, base),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y, base))),
        "r2": r2_score(y, pred),
        "gini": get_gini(ya, pa),
        "actual_over_expected": float(y.sum() / pred.sum()),
        "baseline_actual_over_expected": float(y.sum() / base.sum()),
    }
    return pd.DataFrame([dict])


# large-loss classifier metrics
def get_calibration_table(y_true, y_prob, n_bins: int = config.N_CALIBRATION_BINS) -> pd.DataFrame:
    """Quantile-bin table: mean predicted vs observed rate per bin."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False) + 1
    t = d.groupby("bin").agg(n=("y", "size"), predicted_rate=("p", "mean"),
                             observed_rate=("y", "mean"), n_large=("y", "sum"),
                             predicted_n_large=("p", "sum"))
    return t.reset_index()


def get_total_metrics(oos: pd.DataFrame) -> dict:
    """Metrics for the combined prediction against the uncapped incurred cost."""
    y = oos[config.TARGET_INCURRED].to_numpy()
    pred = oos[config.PRED_TOTAL_COL].to_numpy()
    base = oos[config.PRED_BASELINE_TOTAL_COL].to_numpy()
    dict = {
        "actual_total": float(y.sum()),
        "predicted_total": float(pred.sum()),
        "baseline_predicted_total": float(base.sum()),
        "actual_over_expected": float(y.sum() / pred.sum()),
        "baseline_actual_over_expected": float(y.sum() / base.sum()),
        "mae": float(np.abs(y - pred).mean()),
        "baseline_mae": float(np.abs(y - base).mean()),
        "rmse": float(np.sqrt(((y - pred) ** 2).mean())),
        "baseline_rmse": float(np.sqrt(((y - base) ** 2).mean())),
        "r2": r2_score(y, pred),
        "gini": get_gini(y, pred),
    }
    return pd.DataFrame([dict])


def get_total_components_table(oos: pd.DataFrame) -> pd.DataFrame:
    """Where does the total come from? Capped part vs the large-loss excess part, actual vs predicted."""
    actual_excess = (oos[config.TARGET_INCURRED] - config.CAP).clip(lower=0).sum()
    predicted_excess = (oos[config.PRED_TOTAL_COL] - oos[config.PRED_CAPPED_COL]).sum()      # sum of P(large) x excess
    rows = [
        ("Capped part (up to the cap)", oos[config.TARGET_CAPPED].sum(), oos[config.PRED_CAPPED_COL].sum()),
        ("Excess part (above the cap)", actual_excess, predicted_excess),
        ("Total", oos[config.TARGET_INCURRED].sum(), oos[config.PRED_TOTAL_COL].sum()),
    ]
    t = pd.DataFrame(rows, columns=["component", "actual", "predicted"])
    t["actual_over_expected"] = t["actual"] / t["predicted"]
    return t

def get_ece(y_true, y_prob, n_bins: int = config.N_CALIBRATION_BINS) -> float:
    """Expected calibration error with quantile bins (weighted mean gap between predicted and observed rate)."""
    t = get_calibration_table(y_true, y_prob, n_bins)
    return float((t["n"] / t["n"].sum() * (t["predicted_rate"] - t["observed_rate"]).abs()).sum())


def get_bootstrap_ece(y_true, y_prob, n_boot: int = config.N_BOOTSTRAP) -> list:
    rng = np.random.default_rng(config.RANDOM_SEED)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    values = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        values.append(get_ece(y_true[i], y_prob[i]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def get_large_metrics(oos: pd.DataFrame, dev_prevalence: float) -> dict:
    """Discrimination and calibration. dev_prevalence = large-claim rate in the development rows."""
    y, p = oos[config.TARGET_LARGE], oos[config.PROB_LARGE_COL]
    p_clip = p.clip(1e-6, 1 - 1e-6)
    brier = brier_score_loss(y, p)
    brier_ref = brier_score_loss(y, np.full(len(y), dev_prevalence))
    dict = {
        "n_oos": int(len(y)),
        "n_large_actual": int(y.sum()),
        "n_large_predicted": float(p.sum()),
        "calibration_in_the_large": float(y.sum() / p.sum()),     # actual / predicted count (1 = perfect)
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "average_precision_baseline": float(y.mean()),
        "brier": brier,
        "brier_baseline": brier_ref,
        "brier_skill_score": 1 - brier / brier_ref,
        "logloss": log_loss(y, p_clip),
        "logloss_baseline": log_loss(y, np.full(len(y), dev_prevalence)),
        "ece": get_ece(y, p),
    }
    return pd.DataFrame([dict])


# total cost (capped prediction + large-loss excess)
def get_excess(dev: pd.DataFrame, method: str) -> float:
    """Flat excess above the cap per large claim, estimated on development rows only."""
    excess = dev.loc[dev[config.TARGET_LARGE] == 1, config.TARGET_INCURRED] - config.CAP
    if method == "mean":
        return float(excess.mean())
    if method == "median":
        return float(excess.median())
    if method == "trimmed_mean_10pct":
        lo, hi = excess.quantile(0.10), excess.quantile(0.90)
        return float(excess[(excess >= lo) & (excess <= hi)].mean())
    raise ValueError(method)


def get_total_cost_table(oos: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    """Actual vs predicted total cost under different excess loadings."""
    actual_total = oos[config.TARGET_INCURRED].sum()
    rows = []
    for method in config.EXCESS_METHODS:
        excess = get_excess(dev, method)
        pred_total = (oos[config.PRED_CAPPED_COL] + oos[config.PROB_LARGE_COL] * excess).sum()
        rows.append({"excess_method": method, "excess_per_large_claim": excess,
                     "actual_total": actual_total, "predicted_total": pred_total,
                     "actual_over_expected": actual_total / pred_total})
    return pd.DataFrame(rows)


# tables
def get_decile_table(oos: pd.DataFrame, pred_col: str, actual_col: str, n: int = config.N_DECILES) -> pd.DataFrame:
    d = oos.assign(decile=pd.qcut(oos[pred_col].rank(method="first"), n, labels=False) + 1)
    t = d.groupby("decile").agg(n=(actual_col, "size"), actual_mean=(actual_col, "mean"),
                                predicted_mean=(pred_col, "mean"))
    t["actual_over_expected"] = t["actual_mean"] / t["predicted_mean"]
    return t.reset_index()


def get_segment_ae(oos: pd.DataFrame, segment_cols: list = config.SEGMENT_COLS) -> pd.DataFrame:
    """Actual / expected of capped severity by segment level (NaN levels shown as 'UNKNOWN')."""
    tables = []
    for col in segment_cols:
        level = oos[col].astype("object").fillna("UNKNOWN")
        t = oos.groupby(level).agg(n=(config.TARGET_CAPPED, "size"), actual=(config.TARGET_CAPPED, "sum"),
                                   predicted=(config.PRED_CAPPED_COL, "sum"))
        t["actual_over_expected"] = t["actual"] / t["predicted"]
        t.index.name = "level"
        tables.append(t.reset_index().assign(segment=col))
    return pd.concat(tables)[["segment", "level", "n", "actual", "predicted", "actual_over_expected"]]


def get_year_ae(oos: pd.DataFrame, target, predicted) -> pd.DataFrame:
    t = oos.groupby("loss_year").agg(n=(target, "size"), actual=(target, "sum"),
                                     predicted=(predicted, "sum"))
    t["actual_over_expected"] = t["actual"] / t["predicted"]
    return t.reset_index()

def get_gains_table(oos: pd.DataFrame, n_bins: int = config.N_CALIBRATION_BINS) -> pd.DataFrame:
    """Gains table: claims sorted from highest to lowest predicted probability, cut into equal-size bins.

    Bin 1 = highest predicted probability. Cumulative columns run from the top bin downwards.
    """
    d = oos[[config.TARGET_LARGE, config.PROB_LARGE_COL, config.TARGET_INCURRED]].copy()
    d["excess_cost"] = (d[config.TARGET_INCURRED] - config.CAP).clip(lower=0)      # £ above the cap, actual
    # same ranking as get_calibration_table (ascending), then flip so that bin 1 = highest probabilities
    d["bin"] = n_bins - pd.qcut(d[config.PROB_LARGE_COL].rank(method="first"), n_bins, labels=False)

    t = d.groupby("bin").agg(n=(config.TARGET_LARGE, "size"), n_large=(config.TARGET_LARGE, "sum"),
                             excess_cost=("excess_cost", "sum")).reset_index()
    overall_rate = d[config.TARGET_LARGE].mean()
    t["share_of_claims"] = t["n"] / t["n"].sum()
    t["share_of_large"] = t["n_large"] / t["n_large"].sum()
    t["share_of_excess_cost"] = t["excess_cost"] / t["excess_cost"].sum()
    t["cum_share_of_claims"] = t["share_of_claims"].cumsum()
    t["cum_share_of_large"] = t["share_of_large"].cumsum()
    t["cum_share_of_excess_cost"] = t["share_of_excess_cost"].cumsum()
    t["lift"] = (t["n_large"] / t["n"]) / overall_rate             # bin rate vs overall rate
    return t


# plots
def plot_decile_ae(oos: pd.DataFrame, pred_col: str = config.PRED_CAPPED_COL, actual_col: str = config.TARGET_CAPPED,
                   title: str = "Actual vs predicted by decile"):
    t = get_decile_table(oos, pred_col, actual_col)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t["decile"], t["actual_mean"], marker="o", label="Actual")
    ax.plot(t["decile"], t["predicted_mean"], marker="s", label="Predicted")
    ax.set_xlabel("Predicted decile (1 = lowest)")
    ax.set_ylabel("Mean (£)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_lift(oos: pd.DataFrame, title: str = "Lift / gains curve"):
    """Cumulative share of total cost captured, claims ordered from highest to lowest prediction."""
    ordered = oos.sort_values(config.PRED_CAPPED_COL, ascending=False)
    cum_cost = ordered[config.TARGET_CAPPED].cumsum() / ordered[config.TARGET_CAPPED].sum()
    share_claims = np.arange(1, len(ordered) + 1) / len(ordered)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(share_claims, cum_cost.to_numpy())
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Random")
    ax.set_xlabel("Share of claims (highest predicted first)")
    ax.set_ylabel("Share of total capped claims cost")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_segment_ae(segment_table: pd.DataFrame, min_n: int = 30):
    t = segment_table[segment_table["n"] >= min_n].copy()
    t["label"] = t["segment"] + " = " + t["level"].astype(str)
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(t) + 1.5))
    ax.barh(t["label"], t["actual_over_expected"] - 1, left=1)
    ax.axvline(1, color="black", linewidth=1)
    ax.set_xlabel("Actual / expected")
    ax.set_title(f"A/E by segment (levels with at least {min_n} claims)")
    fig.tight_layout()
    return fig


def plot_year_ae(year_table: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(year_table["loss_year"].astype(str), year_table["actual_over_expected"])
    ax.axhline(1, color="black", linewidth=1)
    ax.set_ylabel("Actual / expected")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_residuals(oos: pd.DataFrame):
    resid = oos[config.TARGET_CAPPED] - oos[config.PRED_CAPPED_COL]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(oos[config.PRED_CAPPED_COL], resid, s=6, alpha=0.4)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xscale("symlog", linthresh=1000)
    ax.set_xlabel("Predicted capped severity (£, symlog scale)")
    ax.set_ylabel("Actual - predicted (£)")
    ax.set_title("Residuals vs prediction")
    fig.tight_layout()
    return fig


def plot_reliability(oos: pd.DataFrame, n_bins: int = config.N_CALIBRATION_BINS):
    """Reliability diagram (quantile bins) with the distribution of predicted probabilities underneath."""
    t = get_calibration_table(oos[config.TARGET_LARGE], oos[config.PROB_LARGE_COL], n_bins)
    top = max(t["predicted_rate"].max(), t["observed_rate"].max()) * 1.1

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, top], [0, top], color="grey", linestyle=":", label="Perfect")
    ax.plot(t["predicted_rate"], t["observed_rate"], marker="o", label="Model")
    for _, r in t.iterrows():
        ax.annotate(f"{int(r['n_large'])}/{int(r['n'])}", (r["predicted_rate"], r["observed_rate"]),
                    textcoords="offset points", xytext=(5, -10), fontsize=8)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed large-claim rate")
    ax.set_title("Reliability diagram (labels = large claims / claims in bin)")
    ax.legend()
    ax2.hist(oos[config.PROB_LARGE_COL], bins=30)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Claims")
    fig.tight_layout()
    return fig


def plot_large_count_by_band(oos: pd.DataFrame, n_bins: int = config.N_CALIBRATION_BINS):
    """Predicted vs actual number of large claims in each probability band."""
    t = get_calibration_table(oos[config.TARGET_LARGE], oos[config.PROB_LARGE_COL], n_bins)
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, t["predicted_n_large"], width=0.4, label="Predicted")
    ax.bar(x + 0.2, t["n_large"], width=0.4, label="Actual")
    ax.set_xticks(x)
    ax.set_xticklabels(t["bin"])
    ax.set_xlabel("Probability band (1 = lowest)")
    ax.set_ylabel("Number of large claims")
    ax.set_title("Predicted vs actual large claims by band")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_gains(oos: pd.DataFrame, n_bins: int = config.N_CALIBRATION_BINS):
    """Cumulative share of large claims captured as you move down the model's ranking (quintile steps)."""
    t = get_gains_table(oos, n_bins)
    x = np.r_[0, t["cum_share_of_claims"].to_numpy()]
    y = np.r_[0, t["cum_share_of_large"].to_numpy()]

    # perfect model: all large claims ranked first
    prevalence = oos[config.TARGET_LARGE].mean()
    perfect_x = [0, prevalence, 1]
    perfect_y = [0, 1, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y, marker="o", label="Model")
    ax.plot(perfect_x, perfect_y, linestyle="--", label="Perfect")
    ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Random")
    for xi, yi, (_, r) in zip(x[1:], y[1:], t.iterrows()):
        ax.annotate(f"{int(r['n_large'])} large", (xi, yi), textcoords="offset points", xytext=(5, -12), fontsize=8)
    ax.set_xlabel("Cumulative share of claims (highest predicted probability first)")
    ax.set_ylabel("Cumulative share of large claims")
    ax.set_title("Cumulative gains - large-loss model (quintiles)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_total_components(components_table: pd.DataFrame):
    """Actual vs predicted total, split into the capped part and the excess part."""
    t = components_table[components_table["component"] != "Total"]
    x = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, t["actual"] / 1e6, width=0.4, label="Actual")
    ax.bar(x + 0.2, t["predicted"] / 1e6, width=0.4, label="Predicted")
    ax.set_xticks(x)
    ax.set_xticklabels(t["component"])
    ax.set_ylabel("£ million")
    ax.set_title("Total cost components (oos)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_excess_sensitivity(total_cost_table: pd.DataFrame):
    """Total actual / expected under each excess-loading method."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(total_cost_table["excess_method"], total_cost_table["actual_over_expected"])
    ax.axhline(1, color="black", linewidth=1)
    ax.set_ylabel("Actual / expected (total cost)")
    ax.set_title("Sensitivity of total A/E to the excess loading")
    fig.tight_layout()
    return fig


# SHAP values and plots 
def get_shap_values(model, X: pd.DataFrame):
    """SHAP values for one fitted XGBoost model.

    Severity model: values are on the log scale (multiplicative effect on expected cost).
    Classifier: values are on the log-odds scale.
    """
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer(X)          # shap.Explanation object


def plot_shap_top_features(shap_values, n_features: int = 10, title: str = ""):
    """Beeswarm and mean-|SHAP| bar plot of the top n features. Returns (beeswarm_fig, bar_fig)."""
    import shap

    # max_display counts the "sum of other features" row too, hence + 1
    plt.figure()
    shap.plots.beeswarm(shap_values, max_display=n_features + 1, show=False)
    plt.title(f"{title} - SHAP beeswarm (top {n_features})")
    plt.tight_layout()
    fig_beeswarm = plt.gcf()

    plt.figure()
    shap.plots.bar(shap_values, max_display=n_features + 1, show=False)
    plt.title(f"{title} - mean |SHAP| (top {n_features})")
    plt.tight_layout()
    fig_bar = plt.gcf()
    return fig_beeswarm, fig_bar


def plotByFactor(
    dt,
    xname,
    yname="Response",
    pred1="Predicted",
    pred2=None,
    xacc=None,
    xmax=None,
    xmin=None,
    ind=None,
    logodds=False,
    y_axis=None,
    x_axis=None,
    text_size=12,
    chart_title=None,
    actual_name="Actuals",
    pred1_name=None,
    pred2_name=None,
    figsize=[1200, 700],
):

    if xacc is not None and dt[xname].dtype != "datetime64[ns]":
        dt["target"] = round(dt[xname] / xacc) * xacc

    elif dt[xname].dtype == "datetime64[ns]":
        dt["target"] = dt[xname].dt.round(xacc)

    else:
        dt["target"] = dt[xname]

    if xmin is not None:
        dt["target"] = dt["target"].mask(dt["target"] < xmin, xmin)

    if xmax is not None:
        dt["target"] = dt["target"].mask(dt["target"] > xmax, xmax)

    if x_axis is None:
        x_axis = xname

    if y_axis is None:
        y_axis = yname

    if pred1_name is None:
        pred1_name = pred1

    if pred2_name is None:
        pred2_name = pred2

    if chart_title is None:
        chart_title = "AvP by " + xname

    if pred2 is not None:
        gp = (
            dt[["target", yname, pred1, pred2]]
            .groupby("target")
            .agg(
                Act=(yname, "mean"),
                Pred=(pred1, "mean"),
                Pred2=(pred2, "mean"),
                count=(yname, "count"),
                Freq=(yname, "size"),
            )
            .reset_index()
        )

        if logodds is True:
            gp[["Act", "Pred", "Pred2"]] = gp[["Act", "Pred", "Pred2"]].apply(
                lambda x: np.log(x / (1 - x))
            )

    else:
        gp = (
            dt[["target", yname, pred1]]
            .groupby("target")
            .agg(
                Act=(yname, "mean"),
                Pred=(pred1, "mean"),
                count=(yname, "count"),
                Freq=(yname, "size"),
            )
            .reset_index()
        )

        if logodds is True:
            gp[["Act", "Pred"]] = gp[["Act", "Pred"]].apply(
                lambda x: np.log(x / (1 - x))
            )

    p = (
        go.Figure()
        .add_trace(
            go.Scatter(
                x=gp["target"],
                y=gp["Freq"],
                yaxis="y2",
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(0,0,0,0.3)",
                line=dict(color="rgba(0,0,0,0)"),
                name="Volume",
            )
        )
        .add_trace(
            go.Scatter(
                x=gp["target"],
                y=gp["Act"],
                mode="markers",
                marker=dict(color="black"),
                name=actual_name,
            )
        )
        .add_trace(
            go.Scatter(
                x=gp["target"],
                y=gp["Pred"],
                mode="lines",
                line=dict(color="red"),
                name=pred1_name,
            )
        )
    )

    if pred2 is not None:
        p = p.add_trace(
            go.Scatter(
                x=gp["target"],
                y=gp["Pred2"],
                mode="lines",
                line=dict(color="blue"),
                name=pred2_name,
            )
        )

    p = p.update_layout(
        title=dict(text="<b>" + chart_title + "</b>", y=0.9, x=0.5, xanchor="center"),
        yaxis2=dict(
            overlaying="y", side="right", title="Density Volume", zeroline=False
        ),
        xaxis=dict(title=x_axis, showgrid=False),
        yaxis=dict(title=y_axis, showgrid=False),
        plot_bgcolor="rgb(255,255,255)",
        legend=dict(orientation="h", y=-0.15),
        width=figsize[0],
        height=figsize[1],
        font=dict(size=text_size),
    )

    return p