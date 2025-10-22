"""
These tests check that the methods of the SearchSpaceParam for Optuna tuning parameters function correctly.

Currently, all methods are tested except SearchSpaceParam.to_optuna()
"""

import pytest
from tempo_forecasting.optuna_opt.optuna_param import SearchSpaceParam

dummy_search_space_settings = [
    {"name":"bounded_int", "dtype": "int", "search_type": "bounded", "bounds": [1, 10]},
    {"name":"bounded_float", "dtype": "float", "search_type": "bounded", "bounds": [1.0, 10.0]},
    {"name":"bounded_float_with_logscale", "dtype": "float", "search_type": "bounded", "bounds": [1.0, 10.0], "logscale": True},
    {"name":"bounded_int_with_stepsize", "dtype": "int", "search_type": "bounded", "bounds": [1, 10], "step_size": 2},
    {"name":"bounded_float_with_stepsize", "dtype": "float", "search_type": "bounded", "bounds": [1.0, 10.0], "step_size": 2.5},
    {"name":"categorical_str", "dtype": "str", "search_type": "categorical", "choices": ["a","b","c"]},
    {"name":"categorical_bool", "dtype": "bool", "search_type": "categorical", "choices": [True, False, True]},
    {"name":"bad_categorical", "dtype": "str", "search_type": "categorical", "choices": ["a","b","c"], "bounds": [1.0, 10.0], "step_size": 2.5, "logscale": False},
    {"name":"bad_bounded", "dtype": "int", "search_type": "bounded", "bounds": [1, 10], "choices": ["a","b","c"]}
    ]

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_name(settings):
    """Test that .get_name() returns the correct parameter name"""
    param = SearchSpaceParam(**settings)
    assert settings["name"] == param.get_name(), "Name should be set to {}. get_name() returns {}".format(settings["name"],param.get_name())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_set_name(settings):
    """Test that .set_name() sets the correct parameter name"""
    param = SearchSpaceParam(**settings)
    param.set_name("foo")
    assert "foo" == param.get_name(), "Name should be set to 'foo'. .get_name() returns {}".format(param.get_name())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_dtype(settings):
    """Test that .get_dtype() returns the correct parameter data type"""
    param = SearchSpaceParam(**settings)
    assert settings["dtype"] == param.get_dtype(), "Data type should be set to {}. get_dtype() returns {}".format(settings["dtype"],param.get_dtype())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_set_dtype(settings):
    """Test that .set_dtype() sets the correct parameter data type"""
    param = SearchSpaceParam(**{"name":"bounded_int", "dtype": "int", "search_type": "bounded", "bounds": [1, 10]})
    param.set_dtype(settings["dtype"])
    assert settings["dtype"] == param.get_dtype(), "Data type should be set to {}. get_dtype() returns {}".format(settings["dtype"],param.get_dtype())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_search_type(settings):
    """Test that .get_search_type() returns the correct parameter search type"""
    param = SearchSpaceParam(**settings)
    assert settings["search_type"] == param.get_search_type(), "Search type should be set to {}. get_search_type() returns {}".format(settings["search_type"],param.get_search_type())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_bounds(settings):
    """Test that .get_bounds() returns the correct parameter bounds"""
    param = SearchSpaceParam(**settings)
    if settings["search_type"] == "bounded":
        assert settings.get("bounds", None) == param.get_bounds(), "Bounds should be set to {}. get_bounds() returns {}".format(settings["bounds"],param.get_bounds())
    else:
        assert param.get_bounds() == None, "For categorical parameters, .get_bounds() should return None. get_bounds() returns {}".format(param.get_bounds())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_step_size(settings):
    """Test that .get_step_size() returns the correct parameter step size"""
    param = SearchSpaceParam(**settings)
    if settings["search_type"] == "bounded":
        if settings["dtype"] == "int":
            assert settings.get("step_size", 1) == param.get_step_size(), "Bounded ints should return a step_size of 1 if step_size is not set. Step size provided: {}, get_step_size() returns {}".format(settings.get("step_size", None),param.get_step_size())
        else:
            assert settings.get("step_size", None) == param.get_step_size(), "Step size should be set to {}. get_step_size() returns {}".format(settings.get("step_size", None),param.get_step_size())
    else:
        assert param.get_step_size() == None, "For categorical parameters, .get_step_size() should return None. get_step_size() returns {param.get_step_size()}".format()

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_logscale(settings):
    """Test that .get_logscale() returns the correct parameter logscale setting"""
    param = SearchSpaceParam(**settings)
    if settings["search_type"] == "bounded":
        assert settings.get("logscale", False) == param.get_logscale(), "Logscale should be set to {}. get_logscale() returns {}".format(settings.get("logscale", False),param.get_logscale())
    else:
        assert param.get_logscale() == None, "For categorical parameters, .get_logscale() should return None. get_logscale() returns {}".format(param.get_logscale())

@pytest.mark.parametrize("settings", dummy_search_space_settings)
def test_get_choices(settings):
    """Test that .get_choices() returns the correct parameter choices"""
    param = SearchSpaceParam(**settings)
    if settings["search_type"] == "categorical":
        assert settings.get("choices", None) == param.get_choices(), "Choices should be set to {}. get_choices() returns {}".format(settings.get("choices", None),param.get_choices())
    else:
        assert param.get_choices() == None, "For bounded parameters, .get_choices() should return None. get_choices() returns {}".format(param.get_choices())