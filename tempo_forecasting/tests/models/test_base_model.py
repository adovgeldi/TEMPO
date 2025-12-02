"""
These tests check that the BaseModel subclasses (MovingAvgModel, ExpSmoothingModel, ProphetModel, XGBoostModel, LightGBMModel)
correctly inherit the methods of BaseModel.

Currently all methods are tested except .fit() and .predict().

Methods particular to individual classes, (ProphetModel._detect_changepoints()) are not being tested.
"""

import pytest
import json
from tempo_forecasting.models import MovingAvgModel, ExpSmoothingModel, ProphetModel, XGBoostModel, LightGBMModel
import importlib.resources


@pytest.mark.parametrize("ModelClass, expected_name", [
    (MovingAvgModel, "moving_avg"),
    (ExpSmoothingModel,"expsmooth"), 
    (ProphetModel,"prophet"),
    (XGBoostModel,"xgboost"),
    (LightGBMModel,"lightgbm"),
    ])
def test_get_name(ModelClass, expected_name):
    """Test that correct model names are being set for each model class"""
    model = ModelClass(date_col="ds", target_y="y")
    name = model.get_name()
    assert name == expected_name, f".get_name() should return the model's expected name ({expected_name}), not {name}"


@pytest.mark.parametrize("ModelClass", [MovingAvgModel,ExpSmoothingModel,ProphetModel,XGBoostModel,LightGBMModel])
def test_get_date_col(ModelClass):
    """test that date_col is being set and returned correctly for all model classes"""
    model = ModelClass(date_col="ds", target_y="y")
    date_col = model.get_date_col()
    assert date_col == "ds", f"date_col initialized to 'ds', but .get_date_col() returns {date_col}"


@pytest.mark.parametrize("ModelClass", [MovingAvgModel,ExpSmoothingModel,ProphetModel,XGBoostModel,LightGBMModel])
def test_set_default_params_from_json(ModelClass):
    model = ModelClass(date_col="ds", target_y="y")
    initial_default_params = model.default_model_params
    assert initial_default_params, "model initialization should set .default_model_params"

    test_param_path = "tests/utils/test_model_params.json"
    with importlib.resources.files("tempo_forecasting").joinpath(test_param_path).open("r") as f:
        params_to_set = json.load(f)[model.get_name()]

    model._set_default_params_from_json(test_param_path)
    new_default_params = model.default_model_params

    assert initial_default_params != new_default_params, "model's default parameters should have changed from the initial default"
    assert params_to_set == new_default_params, f"model's new default parameters should match the test parameters provided at {test_param_path}"


@pytest.mark.parametrize("ModelClass", [MovingAvgModel,ExpSmoothingModel,ProphetModel,XGBoostModel,LightGBMModel])
def test_set_model_params_default(ModelClass):
    model = ModelClass(date_col="ds", target_y="y")

    default_param_path = "config/default_model_params.json"
    with importlib.resources.files("tempo_forecasting").joinpath(default_param_path).open("r") as f:
        intended_default_params = json.load(f)[model.get_name()]

    model._set_model_params()

    assert model.default_model_params == intended_default_params, f"default model parameters should match those at {default_param_path}"
    assert model.model_params == intended_default_params, f"model parameters should match those at {default_param_path}"


@pytest.mark.parametrize("ModelClass", [MovingAvgModel,ExpSmoothingModel,ProphetModel,XGBoostModel,LightGBMModel])
def test_get_model_params(ModelClass):
    model = ModelClass(date_col="ds", target_y="y")
    model._set_model_params()

    model_params = model.model_params
    get_mp_result = model.get_model_params()

    assert model_params == model.default_model_params, "_set_model_params() should pull from model.default_model_params"
    assert model_params == get_mp_result, f".get_model_params() should return model.model_params({model_params}), not {get_mp_result}"


@pytest.mark.parametrize("ModelClass", [MovingAvgModel,ExpSmoothingModel,ProphetModel,XGBoostModel,LightGBMModel])
def test_set_model_params_nondefault(ModelClass):
    model = ModelClass(date_col="ds", target_y="y")

    with importlib.resources.files("tempo_forecasting").joinpath('config/default_model_params.json').open("r") as f:
        intended_default_params = json.load(f)[model.get_name()]

    with importlib.resources.files("tempo_forecasting").joinpath('tests/utils/test_model_params.json').open("r") as f:
        params_to_set = json.load(f)[model.get_name()]

    model._set_model_params(params_to_set)
    assert model.model_params == {**intended_default_params, **params_to_set}, "should set .model_params to a combination of default and provided parameters"


def _check_correct_param_categories(categorized_params, model_name):
    """Helper function to check correct parameter categorization"""
    param_categories_path = "config/param_categories.json"
    with importlib.resources.files("tempo_forecasting").joinpath(param_categories_path).open("r") as f:
        param_categories = json.load(f)[model_name]

    for param_category in ["build_params","fit_params","custom_params"]:
        set_categorized_params_category = set(categorized_params[param_category].keys())

        set_full_param_category = set(param_categories[param_category])
        params_in_both = set_categorized_params_category.intersection(set_full_param_category)
        params_in_subset_not_full = set_categorized_params_category ^ params_in_both

        assert params_in_subset_not_full == set(), f"parameters categorized to {param_category} should be represented there in {param_categories_path}. Noncompliant parameters: {params_in_subset_not_full} "

@pytest.mark.parametrize("ModelClass", [MovingAvgModel,ExpSmoothingModel,ProphetModel,XGBoostModel,LightGBMModel])
def test_combine_and_categorize_params(ModelClass):
    """Test that provided and default parameters are being combined correctly, then check that they are being categorized correctly"""
    model = ModelClass(date_col="ds", target_y="y")

    default_param_path = "config/default_model_params.json"
    with importlib.resources.files("tempo_forecasting").joinpath(default_param_path).open("r") as f:    
        intended_default_params = json.load(f)[model.get_name()]

    test_param_path = "tests/utils/test_model_params.json"
    with importlib.resources.files("tempo_forecasting").joinpath(test_param_path).open("r") as f:
        params_to_set = json.load(f)[model.get_name()]

    # DEFAULT VALUES ONLY
    default_only_categorized_params = model._combine_and_categorize_params()
    extracted_def_categorized_params = {param:val for param_type in default_only_categorized_params for (param,val) in default_only_categorized_params[param_type].items()}
    assert extracted_def_categorized_params == intended_default_params, "Without a param dict and with fill_in_defaults, categorized params keys and values should match the default parameters."
    _check_correct_param_categories(default_only_categorized_params, model.get_name())

    # PROVIDED VALUES ONLY
    provided_only_categorized_params = model._combine_and_categorize_params(model_param_dict = params_to_set,
                                                                            fill_in_defaults = False)
    extracted_prov_categorized_params = {param:val for param_type in provided_only_categorized_params for (param,val) in provided_only_categorized_params[param_type].items()}
    assert extracted_prov_categorized_params == params_to_set, f"With a param dict and without fill_in_defaults, categorized params keys and values should match the test parameters at {test_param_path}."
    _check_correct_param_categories(provided_only_categorized_params, model.get_name())

    # COMBINATION OF PROVIDED AND DEFAULT VALUES
    combined_categorized_params = model._combine_and_categorize_params(model_param_dict = params_to_set)
    extracted_combo_categorized_params = {param:val for param_type in combined_categorized_params for (param,val) in combined_categorized_params[param_type].items()}
    assert extracted_combo_categorized_params == {**intended_default_params, **params_to_set}, f"With a param dict and fill_in_defaults, categorized params keys and values should be a combination of provided and default settings."
    _check_correct_param_categories(combined_categorized_params, model.get_name())

    # NO VALUES
    # This should trigger a warning if our logger is turned on
    no_categorized_params = model._combine_and_categorize_params(fill_in_defaults = False)
    assert no_categorized_params == {'build_params': {}, 'fit_params': {}, 'custom_params': {}}, "Without a param dict and with fill_in_defaults, the param dict should be empty."