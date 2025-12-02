import pytest
import pandas as pd
import json
import numpy as np
import importlib.resources

from tempo_forecasting.models import ProphetModel, XGBoostModel, LightGBMModel, ExpSmoothingModel, MovingAvgModel, KNNModel

# Constants
start_date = '2018-01-01'
end_date = '2020-12-31'

model_args = {
    'date_col': 'date',
    'target_y': 'val',
    'freq': 'D',
    'min_date': start_date, # Start date of training dataset
    'cutoff_date': end_date # Last day of training set
}

dummy_time_series = pd.DataFrame()
dummy_time_series["date"] = pd.date_range(start=start_date, end=end_date, freq=model_args['freq'])
n_steps = len(dummy_time_series)
dummy_time_series["val"] = np.append(np.sin(np.arange(int(np.floor(n_steps/2)))/120)*10+20,
                                     np.sin(np.arange(int(np.ceil(n_steps/2)))/10)*10+20)

# Helper Function
def get_fitted_params(model, 
                      model_name: str, 
                      param_categories: dict) -> dict:
    """
    Retrieve the fitted parameters of a specified model.

    This function extracts the parameters that have been fitted using the base class structure.
    It supports non-custom parameter of a specified model based on the model's type. It supports
    
    Parameters
        model (model object): 
            The fitted model object from which to extract the parameters. 
        model_name (str): 
            The name of the model type (e.g., "xgboost", "lightgbm", "prophet", 
            "expsmooth", "moving_avg") used to determine the specific parameter 
            retrieval logic.
        param_categories (dict)
            A dictionary containing parameter information for different models, 
            as read from the param_categories.json file.

    Returns
        fitted_model_params (dict)
            A dictionary containing the fitted parameters of the specified model.
            The keys are parameter names, and the values are the corresponding 
            fitted values. If a parameter retrieval is not supported, it may return
            a placeholder string indicating that the parameter could not be retrieved.
            Some models may only return fitted parameters corresponding to the parameter
            names in param_categories.

    Notes:
    -----
        - Some parameters are ignored for the expsmooth model due to incomplete
        retrieval logic and caveats noted within the function.
        - The function currently does not support certain parameters or model types,
        and those cases may print warnings or return placeholder messages.
    """

    fitted_model_params = {}

    if model_name in ["xgboost", "lightgbm"]:
        fitted_model_params = model.model.get_params()

    elif model_name in ["prophet"]:
        known_build_params = param_categories[model_name]["build_params"]
        known_fit_params = param_categories[model_name]["fit_params"]
        noncustom_params = known_build_params + known_fit_params

        for param in noncustom_params:
            fitted_model_params[param] = model.model.__getattribute__(param)
            
    elif model_name in ["expsmooth"]:
        known_build_params = param_categories[model_name]["build_params"]
        known_fit_params = param_categories[model_name]["fit_params"]

        for param in known_build_params:
            if param == "seasonal":
                fitted_model_params[param] = model.model.model.seasonal
            elif param == "seasonal_periods":
                fitted_model_params[param] = model.model.model.seasonal_periods
            elif param == "trend":
                fitted_model_params[param] = model.model.model.trend
            elif param == "initialization_method":
                fitted_model_params[param] = model.model.model._initialization_method
            elif param == "freq":
                fitted_model_params[param] = "PARAMETER RETRIEVAL NOT SUPPORTED"
            else:
                print(f"Parameter check not supported for unknown expsmooth param: {param}")

        for param in known_fit_params:
            fitted_model_params[param] = model.model.params.get(param, "PARAMETER RETRIEVAL NOT SUPPORTED")
            
            # EXPSMOOTH NOTES AND CAVEATS
            # To DO: fix these and build more thorough parameter retrieval logic. For now, not worth the time.
            #   Ignoring: ["optimized","damping_slope","maxiter","method","freq"]
            #   fit param "optimized" needs additional info: we pass in one bool but the model has an array of bools. Need conversion logic
            #   fit param "damping_slope" not found even in docs - looks like the name changed and this is now passed in as "damping_slope" 
            #       but stored as "damping trend"?
            #   fit param "maxiter" also not found even in docs
            #   fit param "method" cannot find in this model, though it's clearly a fit parameter in the docs (??)
            #   build param "freq" stored in model.model.model._index_freq, but where we pass in "D" the model stores <Day>
            #       since other settings for this param will result in many different model.model.model._index_freq vals, ignoring for now
    elif model_name == "knn":
        fitted_model_params["len_q"] = model.len_q
        fitted_model_params["forecast_chunk_size"] = model.forecast_chunk_size
        fitted_model_params["k"] = model.k

    elif model_name == "moving_avg":
        fitted_model_params["n_steps"] = model.n_steps

    return fitted_model_params

## TEST
@pytest.mark.parametrize("model_name, model_class", [
    ("moving_avg", MovingAvgModel),
    ("knn", KNNModel),
    ("expsmooth", ExpSmoothingModel),
    ("prophet", ProphetModel),
    ("xgboost", XGBoostModel),
    ("lightgbm", LightGBMModel),
])
def test_parameters_are_correctly_passed_in(model_name, model_class, 
                                            model_args = model_args, 
                                            data = dummy_time_series):
    """
    Test that parameters passed into .fit() are have actually been set in the final model
    """
    # default parameters
    with importlib.resources.files("tempo_forecasting").joinpath("tests/utils/test_model_params.json").open("r") as f:    
        default_params = json.load(f)

    # param_categories
    with importlib.resources.files("tempo_forecasting").joinpath("config/param_categories.json").open("r") as f:    
        param_categories = json.load(f)

    model = model_class(target_y=model_args['target_y'], date_col=model_args['date_col'])
    params_to_set = default_params[model_name]
    model.fit(data, model_param_dict = params_to_set)

    fitted_model_params = get_fitted_params(model, model_name, param_categories)

    for param, param_val in params_to_set.items():
        non_custom_param = (param not in param_categories[model_name]["custom_params"])
        if non_custom_param or (model_name in ["moving_avg","knn"]):
            ################################################################
            # for every non-custom param (and all-custom moving_avg and knn):
            #    does the final model contain a value for that parameter?
            #    does that value equal what we set it to?
            ################################################################
            fitted_val = fitted_model_params.get(param, "NO FITTED VALUE FOUND")

            # handle special cases
            if (model_name == "lightgbm") and (param == "callbacks") and (param_val == None):
                # callbacks is a fit param for lgbm and shows up differently than for xgb
                pass
            elif (model_name == "prophet"):
                # changepoints, n_changepoints, and automanual_changepoints can all affect each other
                if param in ["changepoints", "n_changepoints"]:
                    if params_to_set.get("automanual_changepoints",False):
                        # if the prophet fit is using automanual_changepoints, 
                        # ignore the changepoints and n_changepoints parameters
                            pass
                    else:
                        # if the prophet fit is NOT using automanual_changepoints, 
                        # and we have either "changepoints" or "n_changepoints" specified
                        # make sure at least one of those two values matches the setting
                        n_changepoints_to_set = params_to_set.get("n_changepoints", "no n_changepoints passed into model")
                        fitted_model_n_changepoints = fitted_model_params.get("n_changepoints", "no n_changepoints in fitted model")
                        n_changepoints_specified = (n_changepoints_to_set == fitted_model_n_changepoints)

                        changepoints_to_set = params_to_set.get("changepoints", "no changepoints passed into model")
                        fitted_model_changepoints = fitted_model_params.get("changepoints", "no n_changepoints in fitted model")
                        changepoints_specified = (changepoints_to_set == fitted_model_changepoints)

                        assert n_changepoints_specified or changepoints_specified, "Either changepoints or n_changepoints set appropriately"
            elif (model_name == "expsmooth") and param in ["optimized","damping_slope","maxiter","method","freq"]:
                # These four fit parameters were particularly tricky to extract from the final fitted model
                pass
            else:
                assert param_val == fitted_val, f"Parameter {param} specified value {param_val} should match model's fitted value {fitted_val}"
        else:
            ################################################################
            # check that non-custom params were set
            # currently: 
            #    windows and lags for xgb and lgbm
            #    automanual_changepoints for prophet
            ################################################################
            assert param in ["windows","lags","automanual_changepoints"], "Only checks for these custom parameters are supported"

            if param in ["windows","lags"]:
                assert model_name in ["xgboost","lightgbm"], "Only tree based models should have lags and windows"
                ################################################################
                # for tree based model custom params "windows" and "lags":
                #    convert our param settings to list of ints if not already
                #    pull feature names from the final model
                #    pull lag or window column names based on standard naming convention
                #    extract lag or window values from column names
                #    compare set of lag or window values we supplied to set that had column
                #    do those sets match?
                ################################################################
                
                if type(param_val) == str:
                    json.loads(param_val)

                col_name_map = {"windows": "rolling_mean", "lags": "lag"} 

                if model_name == "xgboost":
                    fitted_feat_names = model.model.feature_names_in_
                else:
                    # lgbm
                    fitted_feat_names = model.model.feature_name_

                param_feats = [f for f in fitted_feat_names if col_name_map[param] in f]
                fitted_val = [int(f.split("_")[-1]) for f in param_feats]
                
                # print(param_val,fitted_val)
                assert set(param_val) == set(fitted_val), f"Parameter {param} specified value {param_val} should match model's fitted value {fitted_val}"

            elif param == "automanual_changepoints":
                assert model_name == "prophet"

                if param_val:
                    # changepoints = fitted_model_params.get("changepoints", None)
                    assert model.model.specified_changepoints, "If automanual_changepoints selected, fitted model should have had specified changepoints"
                    # Note: does not confirm that the correct changepoints were passed in from 
                    # model._detect_changepoints(). Only confirms that changepoints were passed 
                    # in to the prophet model directly. Edge cases where the user has set both
                    # "changepoints" and "automanual_changepoints" may exist

                    # TO DO: build pytest for confirming correct function of model._detect_changepoints()
                else:
                    pass
                    # If we don't use automanual_changepoints, validation is handled above
                    # for the n_changepoints and changepoints edge case 
            else:
              # Should not ever trigger due to above assert
              print(f"Require special handling for custom parameter {param}")