import numpy as np

from src.features.build_features import build_baseline_features
from src.models.train_baseline import evaluate_auc, train_baseline_classifier


def test_feature_builder_returns_copy():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = build_baseline_features(df)
    assert list(out.columns) == ["a", "b"]
    assert out is not df


def test_baseline_train_and_auc_runs():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ]
    )
    y = np.array([0, 1, 0, 1, 0, 1])

    model = train_baseline_classifier(X, y)
    auc = evaluate_auc(model, X, y)

    assert 0.0 <= auc <= 1.0
