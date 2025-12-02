import pytest
import pandas as pd
import numpy as np
from tempo_forecasting.pipeline.preprocessing_pipeline import (
    _approx_days_per_step,
    get_base_min_train_days, 
    get_train_horizon_multipliers, 
    get_forecast_horizon,
    get_dynamic_min_train_days,
    check_train_data_sufficiency
)


def test_invalid_model_raises():
    with pytest.raises(ValueError):
        get_base_min_train_days("not_a_model", freq='D')


def test_invalid_freq_raises():
    with pytest.raises(ValueError):
        get_base_min_train_days("moving_avg", freq="not_a_freq")


@pytest.mark.parametrize("freq, expected_days", [
    ('D', 1),
    ('W', 7),
    ('W-MON', 7),
    ('M', 30),
    ("MS", 30),
    ('Q', 90),
    ('Q-DEC', 90)
])
def test_approx_days_per_step(freq, expected_days):
    assert _approx_days_per_step(freq) == expected_days


@pytest.mark.parametrize("model_name, freq, expected_days", 
    [   
        # moving_avg: min_years=0.25 -> ceil(0.25*365 / step_days), min_points=20
        ("moving_avg", 'D', max(np.ceil(0.25 * 365 / 1), 20)), # max(183, 20) = 183
        ("moving_avg", 'W', max(np.ceil(0.25 * 365 / 7), 20)), # max(27, 20) = 27
        ("moving_avg", 'M', max(np.ceil(0.25 * 365 / 30), 20)), # max(7, 20) = 20
        ("moving_avg", 'Q', max(np.ceil(0.25 * 365 / 90), 20)), # max(3, 20) = 20

        # expsmooth: min_years=2.0 -> ceil(2*365 / step_days), min_points=30
        ("expsmooth", 'D', max(np.ceil(2.0 * 365 / 1), 30)), # 730
        ("expsmooth", 'W', max(np.ceil(2.0 * 365 / 7), 30)), # 105
        ("expsmooth", 'M', max(np.ceil(2.0 * 365 / 30), 30)), # 25 -> 30
        ("expsmooth", 'Q', max(np.ceil(2.0 * 365 / 90), 30)), # 9 -> 30

        # prophet: min_years=1.5 -> ceil(1.5*365 / step_days), min_points=50
        ("prophet", 'W', max(np.ceil(1.5 * 365 / 7), 50)), # 79

        # xgboost: min_years=2.0 -> ceil(2*365 / step_days), min_points=100
        ("xgboost", 'W', max(np.ceil(2.0 * 365 / 7), 100)), # 105
        ("xgboost", 'M', max(np.ceil(2.0 * 365 / 30), 100)), # 25 -> 100

        # knn: min_years=2.0 -> ceil(2*365 / step_days), min_points=100
        ("knn", 'Q', max(np.ceil(2.0 * 365 / 90), 100)), # 9 -> 100
])
def test_get_base_min_train_days(model_name, freq, expected_days):
    assert get_base_min_train_days(model_name, freq) == expected_days


def test_get_base_min_train_days_invalid():
    with pytest.raises(ValueError, match="Invalid model name: 'unknown'"):
        get_base_min_train_days("unknown", freq='D')


@pytest.mark.parametrize("model_name, expected_multiplier", [
    ("moving_avg", 2),
    ("knn", 3),
    ("expsmooth", 3),
    ("prophet", 3),
    ("xgboost", 3),
    ("lightgbm", 3),
])
def test_get_train_horizon_multipliers(model_name, expected_multiplier):
    assert get_train_horizon_multipliers(model_name) == expected_multiplier


def test_invalid_get_train_horizon_multipliers():
    with pytest.raises(ValueError):
        get_train_horizon_multipliers("unknown")


def test_get_forecast_horizon_daily():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
    args = {"date_col": "date", 
            "freq": "D", 
            "validate":False,
            "cv_dates": [["2024-01-01","2024-01-05","2024-01-10"]]}
    assert get_forecast_horizon(data, args) == 5


def test_get_forecast_horizon_monthly():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=6, freq='MS')})
    # args = {"date_col": "date", "cutoff_date": "2024-04-01", "freq": "M", "validate":False}
    args = {"date_col": "date", 
        "freq": "M", 
        "validate":False,
        "cv_dates": [["2024-01-01","2024-04-01","2024-06-01"]]}
    assert get_forecast_horizon(data, args) == 2


def test_get_forecast_horizon_invalid_freq():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5)})
    args = {"date_col": "date", "cutoff_date": "2024-01-10", "freq": "XYZ", "validate":False}
    args = {"date_col": "date", 
        "freq": "XYZ", 
        "validate":False,
        "cv_dates": [["2024-01-01","2024-01-05","2024-01-10"]]}
    with pytest.raises(ValueError, match="Invalid frequency 'XYZ'"):
        get_forecast_horizon(data, args)


@pytest.mark.parametrize("model_name, forecast_horizon, expected_days, freq", [
    ("moving_avg", 30, 92, 'D'), # 3 months training data (or 2x horizon)
    ("knn", 30, 730, 'D'), # 2 years training data (or 3x horizon)
    ("prophet", 30, 548, 'D'), # 1.5 years training data (or 3x horizon)
    ("xgboost", 30, 730, 'D'), # 2 years training data (or 3x horizon)
    ("lightgbm", 26, 105, 'W'), # 2 years training data (or 3x horizon)

    ("prophet", 52, 156, 'W'), # 3x horizon
    ("lightgbm", 52, 156, 'W'), # 3x horizon
])
def test_get_dynamic_min_train_days(model_name, forecast_horizon, expected_days, freq):
    assert get_dynamic_min_train_days(model_name, forecast_horizon, freq) == expected_days


@pytest.mark.parametrize("series_length, model_name, forecast_horizon, expected, freq", [
    (600, "prophet", 20, True, 'D'),   # Sufficient data (1.5 years to train) + 3x forecast horizon
    (600, "prophet", 365, False, 'D'), # Sufficient data (1.5 years to train) BUT less than 3x forecast horizon
    (40, "prophet", 12, False, 'W'),   # Insufficient data (less than base min 50 data points)
    (25,  "moving_avg", 7, False, 'D')  # Insufficient data (less than 3 months to train; 20 is sufficient number of data points)
])
def test_check_train_data_sufficiency(series_length, model_name, forecast_horizon, expected, freq):
    series = pd.Series(range(series_length))
    data_is_sufficient = check_train_data_sufficiency(series, model_name, forecast_horizon, freq)
    assert data_is_sufficient == expected