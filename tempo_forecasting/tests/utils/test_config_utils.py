import pytest
from tempo_forecasting.optuna_opt.optuna_param import SearchSpaceParam
from tempo_forecasting.utils.config_utils import get_default_search_params, get_test_search_params, get_models

def test_get_models_keys():
    models = get_models()
    assert set(models.keys()) == {"expsmooth", "prophet", "xgboost", "lightgbm", "knn"}


def test_get_models_values():
    models = get_models()
    assert all(callable(model) for model in models.values())


@pytest.mark.parametrize("model_name", ["moving_avg", "expsmooth", "prophet", "xgboost", "lightgbm"])
def test_get_default_search_params_valid_model(model_name):
    """Test that default parameters are returned for valid model names."""
    params = get_default_search_params(model_name)
    assert isinstance(params, list), "Output should be a list."
    assert all(isinstance(p, SearchSpaceParam) for p in params), "Each element should be a SearchSpaceParam."
    assert len(params) > 0, "The list should not be empty for valid models."


@pytest.mark.parametrize("model_name", ["moving_avg", "expsmooth", "prophet", "xgboost", "lightgbm"])
def test_get_test_search_params_valid_model(model_name):
    """Test that test parameters are returned for valid model names."""
    params = get_test_search_params(model_name)
    assert isinstance(params, list), "Output should be a list."
    assert all(isinstance(p, SearchSpaceParam) for p in params), "Each element should be a SearchSpaceParam."
    assert len(params) > 0, "The list should not be empty for valid models."


@pytest.mark.parametrize("model_name", ["invalid_model", "unknown_model", ""])
def test_get_default_search_params_invalid_model(model_name):
    """Test that an invalid model name raises a KeyError."""
    with pytest.raises(KeyError):
        get_default_search_params(model_name)


@pytest.mark.parametrize("model_name", ["invalid_model", "unknown_model", ""])
def test_get_test_search_params_invalid_model(model_name):
    """Test that an invalid model name raises a KeyError."""
    with pytest.raises(KeyError):
        get_test_search_params(model_name)

    
@pytest.mark.parametrize("model_name, param_names", [
    ("moving_avg", ["n_days"]),
    ("expsmooth", ["seasonal_periods", "smoothing_level", "smoothing_seasonal"]),
    ("prophet", ["automanual_changepoints", "changepoint_prior_scale", "seasonality_prior_scale", "seasonality_mode", "weekly_seasonality"]),
    ("xgboost", ["max_depth", "learning_rate", "n_estimators", "min_child_weight"]),
    ("lightgbm", ["max_depth", "learning_rate", "min_data_in_leaf", "lambda_l2"]),
])
def test_search_space_param_properties(model_name, param_names):
    """Test that parameters have correct properties for a valid model."""
    params = get_default_search_params(model_name)
    
    param_names_in_params = [p.get_name() for p in params]
    for param_name in param_names:
        assert param_name in param_names_in_params, f"Expected parameter {param_name} not found."