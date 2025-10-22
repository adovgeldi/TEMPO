import pytest
import pandas as pd
import numpy as np
from tempo_forecasting.models.tree_based_model import TreeBasedModel  


class XGBoost(TreeBasedModel):
    """
    Dummy XGBoost subclass for testing (since TreeBasedModel is abstract)
    """
    def fit(self, train_data: pd.DataFrame):
        pass

    def predict(self, future_dates: pd.DataFrame):
        pass


@pytest.fixture
def tree_model():
    """Fixture for creating a DummyTreeModel instance."""
    return XGBoost(name="xgboost", date_col="ds", target_y="y")


@pytest.fixture
def sample_data():
    """Fixture for creating sample data."""
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    data = {
        "ds": dates,
        "y": np.arange(1, 11)
    }
    return pd.DataFrame(data)


def test_create_time_features(tree_model, sample_data):
    """Test _create_time_features method."""
    result = tree_model._create_time_features(sample_data)
    
    # Assert new columns are created
    expected_cols = ["dayofmonth", "dayofweek", "quarter", "month", "year", "dayofyear", "weekofyear"]
    for col in expected_cols:
        assert col in result.columns

    # Assert original date column is dropped
    assert "ds" not in result.columns

    # Assert time features are correct
    assert result["dayofmonth"].iloc[1] == 2  # Second day of the month
    assert result["month"].iloc[0] == 1  # January
    #assert result["month"].dtype == 'categorical'


def test_create_lag_features(tree_model, sample_data):
    """Test _create_lag_features method."""
    lags = [1, 2]
    result = tree_model._create_lag_features(sample_data, lags)

    # Assert lag columns are created
    for lag in lags:
        assert f"lag_{lag}" in result.columns

    # Assert lag values are correct
    assert result["lag_1"].iloc[1] == 1  # Value of the first row is lagged to the second row
    assert np.isnan(result["lag_1"].iloc[0])  # First row should be NaN


def test_create_rolling_features(tree_model, sample_data):
    """Test _create_rolling_features method."""
    windows = [2, 3]
    add_std = True  # Enable rolling standard deviation
    result = tree_model._create_rolling_features(sample_data, windows, add_std=add_std)

    # Assert rolling mean columns are created
    for window in windows:
        assert f"rolling_mean_{window}" in result.columns, f"Missing rolling mean column for window {window}"

    # Assert rolling std columns are created if add_std is True
    if add_std:
        for window in windows:
            assert f"rolling_std_{window}" in result.columns, f"Missing rolling std column for window {window}"

    # Assert rolling mean is correct for window=2
    # Rolling mean for window=2, shifted by 1 row
    expected_rolling_mean_2 = [
        np.nan,  # No rolling mean for the first row
        np.nan,  # Shifted value from second row calculation
        (1 + 2) / 2,  # Mean of rows 0 and 1, shifted to row 2
        (2 + 3) / 2,  # Mean of rows 1 and 2, shifted to row 3
    ]
    assert np.isclose(result["rolling_mean_2"].iloc[2], expected_rolling_mean_2[2]), (
        f"Unexpected value for rolling mean 2 at row 2: {result['rolling_mean_2'].iloc[2]}"
    )
    assert np.isnan(result["rolling_mean_2"].iloc[0]), "Expected NaN for rolling mean 2 at row 0"

    # Assert rolling std is correct for window=2 (if add_std=True)
    if add_std:
        # Rolling std for window=2, shifted by 1 row
        expected_rolling_std_2 = [
            np.nan,  # No rolling std for the first row
            np.nan,  # Shifted value from second row calculation
            np.sqrt(((1 - 1.5)**2 + (2 - 1.5)**2) / 1),  # Std of rows 0 and 1, shifted to row 2 with 1 ddof
        ]
        assert np.isclose(result["rolling_std_2"].iloc[2], expected_rolling_std_2[2]), (
            f"Unexpected value for rolling std 2 at row 2: {result['rolling_std_2'].iloc[2]}"
        )

def test_postprocess(tree_model, sample_data):
    """Test postprocess method."""
    # Add some dummy features
    sample_data["feature1"] = np.random.rand(len(sample_data))
    sample_data["feature2"] = np.random.rand(len(sample_data))
    
    result = tree_model.postprocess(sample_data)

    # Assert feature_cols are identified correctly
    expected_features = ["feature1", "feature2"]
    assert set(tree_model.feature_cols) == set(expected_features)

    # Assert returned DataFrame is unchanged
    pd.testing.assert_frame_equal(result, sample_data)

