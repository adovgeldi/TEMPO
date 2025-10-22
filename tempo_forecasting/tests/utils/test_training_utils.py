import pytest
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from tempo_forecasting.utils.training_utils import (calculate_metric, enforce_datetime_index, select_train_and_test, 
                                        train_and_validate, calculate_metric, cross_validate, calculate_time_periods)

# HELPERS
# Mock of the model to be used in tests
class DummyModel:
    def __init__(self, target_y='value', date_col = 'index'):
        self.fitted_vals = None
        self.target_y = target_y
        self.date_col = date_col

    def fit(self, train_data, model_param_dict):
        # Mock behavior for fitting
        # set fitted vals to be an array the same length as train data
        # all the same number passed in as "dummy_val" via the param dict
        self.fitted_vals = np.ones(len(train_data))*model_param_dict["dummy_val"]  

    def predict(self, val_data):
        # Mock prediction
        # will return a np array the same length as val_data
        # all the same number, in this case the same as the fitted vals
        return np.ones(len(val_data))*np.mean(self.fitted_vals)

# Mock data
def create_test_dataframe(n=5):
    data = {'value': np.arange(n)+1,}
    date_range = pd.date_range(start='2023-01-01', periods=n, freq='D')
    df = pd.DataFrame(data, index=date_range)

    return df

empty_test_df = create_test_dataframe(n=0)
four_day_test_df = create_test_dataframe(n=4)
ten_day_test_df = create_test_dataframe(n=10)


# TESTS
@pytest.mark.parametrize(
    "actual, predicted, metric, expected_result, expected_exception",
    [
        # Valid metrics
        (np.array([100, 150, 200]), np.array([90, 140, 190]), "wmape", 6.67, None),  # WMAPE = 6.67%
        (np.array([100, 150, 200]), np.array([90, 140, 190]), "rmse", 10.0, None),  # RMSE = 10
        (np.array([100, 150, 200]), np.array([90, 140, 190]), "mae", 10.0, None),   # MAE = 10

        # Invalid metric
        (np.array([100, 150, 200]), np.array([90, 140, 190]), "invalid_metric", None, ValueError),
    ]
)
def test_calculate_metric(actual, predicted, metric, expected_result, expected_exception):
    """
    Test `calculate_metric` with various inputs and scenarios.

    Parameters:
        actual (np.array): Ground truth values.
        predicted (np.array): Predicted values.
        metric (str): The metric to calculate ("wmape", "rmse", "mae").
        expected_result (float or None): The expected result for valid metrics.
        expected_exception (Exception or None): The expected exception for invalid metrics.
    """
    if expected_exception:
        with pytest.raises(expected_exception):
            calculate_metric(actual, predicted, metric)
    else:
        result = calculate_metric(actual, predicted, metric)
        assert np.isclose(result, expected_result) 


@pytest.mark.parametrize(
    "df, date_col, expected_exception, expected_index_type, match",
    [     
        ( # Valid DataFrame with date_col
            pd.DataFrame({
                "ds": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "value": [10, 20, 30]
            }),
            "ds",
            None,  # No exception expected
            pd.DatetimeIndex,
            None
        ),
        ( # Already a DatetimeIndex
            pd.DataFrame({
                "value": [10, 20, 30]
            }, index=pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"])),
            "ds",
            None,  # No exception expected
            pd.DatetimeIndex,
            None
        ),
        ( # Missing date_col
            pd.DataFrame({
                "other_col": ["2022-01-01", "2022-01-02", "2022-01-03"],
                "value": [10, 20, 30]
            }),
            "ds",
            ValueError,  # Exception expected
            None,
            "Index not DateTime and ds column not present in DataFrame."
        ),
        ( # Invalid date_col format
            pd.DataFrame({
                "ds": ["invalid_date1", "invalid_date2", "invalid_date3"],
                "value": [10, 20, 30]
            }),
            "ds",
            ValueError,  # Exception expected
            None,
            "Failed to convert index to DatetimeIndex:"
        )
    ]
)
def test_enforce_datetime_index(df, date_col, expected_exception, expected_index_type, match):
    """
    Parametrized test for enforce_datetime_index function.
    """
    if expected_exception:
        with pytest.raises(expected_exception, match=match):
            enforce_datetime_index(df, date_col)
    else:
        result = enforce_datetime_index(df, date_col)
        assert isinstance(result.index, expected_index_type)
        assert result.index[0] == pd.Timestamp("2022-01-01")


@pytest.mark.parametrize("max_date_str, n_test_months, n_train_months, n_validation_sets, cv_window_step_days, expected_output",[
    ("2023-12-31", 2, 3, 1, 30, 
     {
        'full_range': {'total_min': '2023-08-01',
                        'last_cv_min': '2023-08-01',
                        'last_cv_cutoff': '2023-10-31',
                        'total_max': '2023-12-31'},
        'cv_windows': [{'min': '2023-08-01','cutoff': '2023-10-31','max': '2023-12-31'}],
        'retrain_range': {'min': '2023-08-01', 'max': '2023-12-31'}
     }),
    ("2023-12-31", 2, 3, 2, 30, 
     {'full_range': {'total_min': '2023-07-02',
                     'last_cv_min': '2023-08-01',
                     'last_cv_cutoff': '2023-10-31',
                     'total_max': '2023-12-31'},
      'cv_windows': [{'min': '2023-07-02','cutoff': '2023-10-01','max': '2023-12-01'},
                     {'min': '2023-08-01', 'cutoff': '2023-10-31', 'max': '2023-12-31'}],
      'retrain_range': {'min': '2023-07-02', 'max': '2023-12-31'}})
])
def test_calculate_time_periods(max_date_str, 
                                n_test_months, 
                                n_train_months, 
                                n_validation_sets, 
                                cv_window_step_days, 
                                expected_output):
    
    output = calculate_time_periods(max_date_str, 
                                    n_test_months, 
                                    n_train_months, 
                                    n_validation_sets, 
                                    cv_window_step_days)

    # CONFIRM OUTPUT MATCHES ANSWER KEY    
    assert output["full_range"]["total_min"] == expected_output["full_range"]["total_min"]
    assert output["full_range"]["last_cv_min"] == expected_output["full_range"]["last_cv_min"]
    assert output["full_range"]["last_cv_cutoff"] == expected_output["full_range"]["last_cv_cutoff"]
    assert output["full_range"]["total_max"] == expected_output["full_range"]["total_max"]

    assert len(output["cv_windows"]) == len(expected_output["cv_windows"])
    for window_i in range(len(expected_output["cv_windows"])):
        assert output["cv_windows"][window_i]["min"] == expected_output["cv_windows"][window_i]["min"]
        assert output["cv_windows"][window_i]["cutoff"] == expected_output["cv_windows"][window_i]["cutoff"]
        assert output["cv_windows"][window_i]["max"] == expected_output["cv_windows"][window_i]["max"]

    assert output["retrain_range"]["min"] == expected_output["retrain_range"]["min"]
    assert output["retrain_range"]["max"] == expected_output["retrain_range"]["max"]

    # INTERNAL CONSISTENCY
    assert output["full_range"]["total_min"] == output["cv_windows"][0]["min"]
    assert output["full_range"]["last_cv_min"] == output["cv_windows"][-1]["min"]
    assert output["full_range"]["last_cv_cutoff"] == output["cv_windows"][-1]["cutoff"]
    assert output["full_range"]["total_max"] == output["cv_windows"][-1]["max"]

    assert output["retrain_range"]["min"] == output["full_range"]["total_min"]
    assert output["retrain_range"]["max"] == output["full_range"]["total_max"]

    # NUMERIC CONSISTENCY
    # max_date_str matches actual max date outputted
    assert output["full_range"]["total_max"] == max_date_str

    # last cv window has expected number of months in train and test period
    max_date = pd.Timestamp(output["full_range"]["total_max"]).date()
    last_cutoff_date = pd.Timestamp(output["full_range"]["last_cv_cutoff"]).date()
    last_min_date = pd.Timestamp(output["full_range"]["last_cv_min"]).date()

    assert((last_cutoff_date + relativedelta(months = n_test_months)) == max_date) 
    assert((last_min_date + relativedelta(months = 3, days = -1)) == last_cutoff_date)

    # each non-final cv window has:
    #   - same number of training days as last cv window's training days
    #   - same number of testing days as last cv window's testing days
    #   - correct number of days between start of this window and the next
    if len(output["cv_windows"]) > 1:
        expected_test_days_between = (max_date - last_cutoff_date).days
        expected_train_days_between = (last_cutoff_date - last_min_date).days
        for window_i in range(len(output["cv_windows"]))[:-1]:
            next_cv_start_date = pd.Timestamp(output["cv_windows"][window_i+1]["min"])
            cv_start_date = pd.Timestamp(output["cv_windows"][window_i]["min"])

            assert ((next_cv_start_date - cv_start_date).days == cv_window_step_days)

            cv_cutoff_date = pd.Timestamp(output["cv_windows"][window_i]["cutoff"])
            cv_max_date = pd.Timestamp(output["cv_windows"][window_i]["max"])
            assert ((cv_max_date-cv_cutoff_date).days == expected_test_days_between)
            assert ((cv_cutoff_date-cv_start_date).days == expected_train_days_between)


@pytest.mark.parametrize("data, date_col, min_date_str, cutoff_date_str, max_date_str, expected_train, expected_test", [
    # Case 1: Basic case where cutoff divides the dataset into train and test
    (four_day_test_df,'date', '2023-01-01', '2023-01-02', '2023-01-04', 
     pd.DataFrame({"value": [1, 2]}, index=pd.to_datetime(["2023-01-01", "2023-01-02"])).asfreq('d'),
     pd.DataFrame({"value": [3, 4]}, index=pd.to_datetime(["2023-01-03", "2023-01-04"])).asfreq('d')),

    # Case 2: No data falls into train and test (empty data frame)
    (empty_test_df, 'date', '2023-01-01', '2023-01-02', '2023-01-03', empty_test_df, empty_test_df),

    # Case 3: All data falls into train phase
    (four_day_test_df,'date', '2023-01-01', '2023-01-04', '2023-01-04', 
     four_day_test_df,
     empty_test_df),

    # Case 4: Train and test exclude some data from the full input dataset
    (four_day_test_df,'date', '2022-12-31', '2023-01-01', '2023-01-02', 
     pd.DataFrame({"value": [1]}, index=pd.to_datetime(["2023-01-01"])).asfreq('d'),
     pd.DataFrame({"value": [2]}, index=pd.to_datetime(["2023-01-02"])).asfreq('d'))    
])
def test_select_train_and_test(data, date_col, min_date_str, cutoff_date_str, max_date_str, expected_train, expected_test):
    train, test = select_train_and_test(data, date_col, min_date_str, cutoff_date_str, max_date_str)
    
    pd.testing.assert_frame_equal(train, expected_train)
    pd.testing.assert_frame_equal(test, expected_test)


def test_select_train_and_test_invalid_date_range():
    modeling_data = create_test_dataframe()
    date_col = 'index'
    
    # Cutoff date is earlier than min date
    with pytest.raises(AssertionError):
        select_train_and_test(modeling_data, date_col, '2023-01-03', '2023-01-01', '2023-01-05')


@pytest.mark.parametrize(
    "train_data, val_data, model_param_dict, metric_types, expected_metrics",
    [
        (ten_day_test_df[:5],  # Simple case
        ten_day_test_df[5:],
        {"dummy_val": 10}, 
        ['wmape', 'mae'], 
        {'WMAPE': 25.0, 'MAE': 2.0}),  # expected results calculated manually
    ]
)
def test_train_and_validate(train_data, val_data, model_param_dict, metric_types, expected_metrics):
    model = DummyModel()  # Using the mock model to prevent dependency on real implementation
    results = train_and_validate(model, model_param_dict, train_data, val_data, metric_types)

    # Check whether the results contain the expected metrics
    for key, value in expected_metrics.items():
        assert results['metrics'][key] == value
        
    # Ensure fitted_train_vals and test_pred_vals are returned
    assert results['fitted_train_vals'] is not None
    assert len(results['test_pred_vals']) == len(val_data)


@pytest.mark.parametrize(
    "data, cv_dates, model_param_dict, metrics, expected_output",
    [
        (ten_day_test_df,  # Simple case
         [["2023-01-01", "2023-01-01", "2023-01-06"],
          ["2023-01-03", "2023-01-03", "2023-01-08"],
          ["2023-01-05", "2023-01-05", "2023-01-10"]],
        {"dummy_val": 1}, 
        ['wmape', 'mae'], 
        [{'metrics': {'WMAPE': 75.0, 'MAE': 3.0},
            'fitted_train_vals': np.array([1.]),
            'test_pred_vals': np.array([1., 1., 1., 1., 1.]),
            'dates': ['2023-01-01', '2023-01-01', '2023-01-06']},
         {'metrics': {'WMAPE': 83.33, 'MAE': 5.0},
            'fitted_train_vals': np.array([1.]),
            'test_pred_vals': np.array([1., 1., 1., 1., 1.]),
            'dates': ['2023-01-03', '2023-01-03', '2023-01-08']},
         {'metrics': {'WMAPE': 87.5, 'MAE': 7.0},
            'fitted_train_vals': np.array([1.]),
            'test_pred_vals': np.array([1., 1., 1., 1., 1.]),
            'dates': ['2023-01-05', '2023-01-05', '2023-01-10']}]),  
    ]
)
def test_cross_validate(data, cv_dates, model_param_dict, metrics, expected_output):
    cv_results = cross_validate(data = data, 
                                    date_col = "index", 
                                    target_col = "value", 
                                    model_class = DummyModel, 
                                    model_param_dict = model_param_dict, 
                                    cv_dates = cv_dates, 
                                    metrics = metrics)
    len(expected_output) == len(cv_results)
    for cv_window_i in range(len(expected_output)):
        for metric in metrics:
            assert expected_output[cv_window_i]["metrics"][metric.upper()] == cv_results[cv_window_i]["metrics"][metric.upper()]
        assert (expected_output[cv_window_i]["fitted_train_vals"]==cv_results[cv_window_i]["fitted_train_vals"]).all()
        assert (expected_output[cv_window_i]["test_pred_vals"]==cv_results[cv_window_i]["test_pred_vals"]).all()
        assert expected_output[cv_window_i]["dates"]==cv_results[cv_window_i]["dates"]