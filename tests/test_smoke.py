import numpy as np
import pandas as pd

from src.data.loaders import load_excel
from src.features.build_features import build_baseline_features
from src.models.train_baseline import evaluate_auc, train_baseline_classifier
from src.utils.utils import replace_abnormal_entries


def test_feature_builder_returns_copy():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = build_baseline_features(df)
    assert list(out.columns) == ["a", "b"]
    assert out is not df


def test_load_excel_lowercases_columns(tmp_path, monkeypatch):
    file_path = tmp_path / "input.xlsx"
    file_path.touch()
    frame = pd.DataFrame(columns=["Claim ID", "LOSS DATE"])
    monkeypatch.setattr(
        "src.data.loaders.pd.read_excel", lambda *_args, **_kwargs: frame
    )

    result = load_excel(file_path)

    assert list(result.columns) == ["claim id", "loss date"]


def test_load_excel_lowercases_columns_for_all_sheets(tmp_path, monkeypatch):
    file_path = tmp_path / "input.xlsx"
    file_path.touch()
    sheets = {
        "first": pd.DataFrame(columns=["Claim ID"]),
        "second": pd.DataFrame(columns=["LOSS DATE"]),
    }
    monkeypatch.setattr(
        "src.data.loaders.pd.read_excel", lambda *_args, **_kwargs: sheets
    )

    result = load_excel(file_path, sheet_name=None)

    assert [list(frame.columns) for frame in result.values()] == [
        ["claim id"],
        ["loss date"],
    ]


def test_replace_abnormal_entries_with_nan():
    dataframe = pd.DataFrame(
        {
            "claim": ["N/K", "n/k", "#", "   ", "valid", np.nan],
            "amount": [1, 2, 3, 4, 5, 6],
        }
    )

    cleaned = replace_abnormal_entries(dataframe)

    assert cleaned["claim"].iloc[:4].isna().all()
    assert cleaned["claim"].iloc[4] == "valid"
    assert pd.isna(cleaned["claim"].iloc[5])
    assert cleaned["amount"].tolist() == [1, 2, 3, 4, 5, 6]
    assert dataframe["claim"].iloc[0] == "N/K"


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
