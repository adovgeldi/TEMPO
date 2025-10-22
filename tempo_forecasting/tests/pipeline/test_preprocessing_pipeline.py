import pytest
import pandas as pd
from tempo_forecasting.pipeline.preprocessing_pipeline import (
    get_models, 
    get_base_min_train_days, 
    get_train_horizon_multipliers, 
    get_forecast_horizon,
    get_dynamic_min_train_days,
    check_train_data_sufficiency
)

@pytest.mark.parametrize("model_name, expected_days", [
    ("moving_avg", 60),
    ("knn", 540),
    ("expsmooth", 365*2),
    ("prophet", 180),
    ("xgboost", 180),
    ("lightgbm", 180),
])
def test_get_base_min_train_days(model_name, expected_days):
    assert get_base_min_train_days(model_name) == expected_days


def test_get_base_min_train_days_invalid():
    with pytest.raises(ValueError, match="Invalid model name: 'unknown'"):
        get_base_min_train_days("unknown")


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


def test_get_train_horizon_multipliers_invalid():
    with pytest.raises(ValueError, match="Invalid model name: 'unknown'"):
        get_train_horizon_multipliers("unknown")


def test_get_forecast_horizon_daily():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10)})
    args = {"date_col": "date", 
            "freq": "D", 
            "validate":False,
            "cv_dates": [["2024-01-01","2024-01-05","2024-01-10"]]}
    assert get_forecast_horizon(data, args) == 5


def test_get_forecast_horizon_monthly():
    data = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5, freq='M')})
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


@pytest.mark.parametrize("model_name, forecast_horizon, expected_days", [
    ("moving_avg", 30, 60),
    ("knn", 30, 540),
    ("expsmooth", 30, 730),
    ("prophet", 30, 180),
    ("xgboost", 30, 180),
    ("lightgbm", 45, 180),
])
def test_get_dynamic_min_train_days(model_name, forecast_horizon, expected_days):
    assert get_dynamic_min_train_days(model_name, forecast_horizon) == expected_days


@pytest.mark.parametrize("series_length, model_name, forecast_horizon, expected", [
    (200, "prophet", 20, True),   # Sufficient data
    (100, "prophet", 20, False),  # Insufficient data
    (50,  "prophet", 100, False)  # Insufficient data
])
def test_check_train_data_sufficiency(series_length, model_name, forecast_horizon, expected):
    series = pd.Series(range(series_length))
    data_is_sufficient = check_train_data_sufficiency(series, model_name, forecast_horizon)
    assert data_is_sufficient == expected