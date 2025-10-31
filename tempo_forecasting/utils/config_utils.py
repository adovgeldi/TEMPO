from tempo_forecasting.optuna_opt.optuna_param import SearchSpaceParam
from tempo_forecasting.models import MovingAvgModel, KNNModel, ExpSmoothingModel, ProphetModel, XGBoostModel , LightGBMModel
from typing import Dict, Any, Optional, Tuple, Sequence


def get_models(how: str = "default") -> Dict[str, Any]:
    """
    Returns a dictionary mapping model names to their corresponding classes.
    
    Parameters:
        how (str): which set of models to return
            - "default": the default models
            - "simple": the simple models, for datasets with insufficient data
            - "all": every available model type

    Returns:
        dict: A dictionary where keys are model names (str) and values are model classes.
    """
    if how == "default":
        return {
            # "moving_avg": MovingAvgModel,
            "knn": KNNModel,
            "expsmooth": ExpSmoothingModel,
            "prophet": ProphetModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel
            }
    elif how == "simple":
        return {
            "moving_avg": MovingAvgModel,
            "knn": KNNModel
            }
    elif how == "all":
        return {
            "moving_avg": MovingAvgModel,
            "knn": KNNModel,
            "expsmooth": ExpSmoothingModel,
            "prophet": ProphetModel,
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel
            }
        

def get_default_search_params(model_name: str) -> Sequence[SearchSpaceParam]:
    """
    Retrieves the default hyperparameter search space for the specified model.

    This function defines the hyperparameters to optimize for various machine learning 
    and time series models. The parameters include both bounded and categorical search spaces.

    Parameters:
        model_name (str): The name of the model for which to retrieve the search space. 
                          Valid options include "moving_avg", "expsmooth", "prophet", "xgboost", 
                          and "lightgbm".

    Returns:
        Sequence[SearchSpaceParam]: A sequence of SearchSpaceParam objects defining 
                                    the search space for the specified model.
    """
    if model_name == "moving_avg":
        return [
            # SearchSpaceParam("n_steps", "int", "categorical", choices=[30,60,90,120,365]),
            SearchSpaceParam("n_steps", "int", "categorical", choices=[1,4,8,12]),
        ]
    if model_name == "knn":
        return [
            # SearchSpaceParam("len_q", "int", "categorical", choices=[90,120,180]),
            # SearchSpaceParam("forecast_chunk_size", "int", "categorical", choices=[60,90,120]),
            SearchSpaceParam("len_q", "int", "categorical", choices=[4,12,26]),
            SearchSpaceParam("forecast_chunk_size", "int", "categorical", choices=[4,8,12,26]),
            SearchSpaceParam("k", "int", "categorical", choices=[2,3]),
        ]
    elif model_name =="expsmooth":
        return [
            # SearchSpaceParam("seasonal_periods", "int", "categorical", choices=[184,365]), # annual or biannual seasonality
            SearchSpaceParam("seasonal_periods", "int", "categorical", choices=[26,52]), # annual or biannual seasonality
            SearchSpaceParam("smoothing_level", "float", "bounded", bounds=[0.1, 1.0], step_size=0.1), # smoothing factor for data
           # SearchSpaceParam("smoothing_trend", "float", "bounded", bounds=[0.01, 0.31], step_size=0.05), # smoothing factor for trend
            SearchSpaceParam("smoothing_seasonal", "float", "bounded", bounds=[0.01, 0.51], step_size=0.05), # smoothing factor for seasonality
           # SearchSpaceParam("damping_trend", "float", "bounded", bounds=[0.86,0.94], step_size=0.02), # damping_slope for longer-term forecasts
        ]
    elif model_name == "prophet":
        # https://facebook.github.io/prophet/docs/diagnostics.html
        return [
            SearchSpaceParam("automanual_changepoints", "bool", "categorical", choices = [True, False]), # determines whether to use our custom changepoint detection or builtin func
            SearchSpaceParam("changepoint_prior_scale", "float", "bounded", bounds = [0.001,0.5], logscale = True), # controls model flexibility in detecting changepoints
            SearchSpaceParam("seasonality_prior_scale", "float", "bounded", bounds = [0.01,10.0], logscale = True), # regularization strength of seasonal components
        #    SearchSpaceParam("holiday_prior_scale", "float", "bounded", bounds = [0.01,10.0], logscale = True), # controls flexibility of holiday effects
            SearchSpaceParam("seasonality_mode", "str", "categorical", choices = ["additive","multiplicative"]),
            SearchSpaceParam("weekly_seasonality", "bool", "categorical", choices = [True, False]),
        ]
    elif model_name == "xgboost":
        return [
            SearchSpaceParam("max_depth", "int", "bounded", bounds=[3, 8]), # used to combat overfitting; larger value = larger complexity
            SearchSpaceParam("learning_rate", "float", "bounded", bounds=[0.01, 0.3], logscale=True), # controls how much each new decision tree contributes to final; smaller = more stable predictions
            SearchSpaceParam("n_estimators", "int", "bounded", bounds=[100,800], step_size=100), # higher values should help with more complex seasonality patterns
            SearchSpaceParam("min_child_weight", "int", "bounded", bounds=[1,7]), # higher values can help combat overfitting
            # SearchSpaceParam("windows", "str", "categorical", choices = ["[7, 30, 92, 184, 365]","[7, 30, 60, 90]"]),
            # SearchSpaceParam("lags", "str", "categorical", choices = ["[184, 365]","[]"])
            SearchSpaceParam("windows", "str", "categorical", choices = ["[1, 4, 12, 26]","[1, 4, 8, 12]"]),
            SearchSpaceParam("lags", "str", "categorical", choices = ["[26, 52]","[]"])
        ] 
    elif model_name == "lightgbm":
        return [
            SearchSpaceParam("max_depth", "int", "bounded", bounds=[3, 8]), # used to combat overfitting; larger value = larger complexity
            SearchSpaceParam("learning_rate", "float", "bounded", bounds=[0.01, 0.1], logscale=True), # controls how much each new decision tree contributes to final; smaller = more stable predictions
            SearchSpaceParam("n_estimators", "int", "bounded", bounds=[100,800], step_size=100), # higher values should help with more complex seasonality patterns
            SearchSpaceParam("min_data_in_leaf", "int", "bounded", bounds=[40, 140], step_size=20), # may help: higher values for more expensive machines with sparse rentals
            SearchSpaceParam("lambda_l2", "float", "bounded", bounds=[1e-2, 10.0], logscale=True), # L2 regularization
            # SearchSpaceParam("windows", "str", "categorical", choices = ["[7, 30, 92, 184, 365]","[7, 30, 60, 90]"]),
            # SearchSpaceParam("lags", "str", "categorical", choices = ["[184, 365]","[]"])
            SearchSpaceParam("windows", "str", "categorical", choices = ["[1, 4, 12, 26]","[1, 4, 8, 12]"]),
            SearchSpaceParam("lags", "str", "categorical", choices = ["[26, 52]","[]"])
        ]
    else:
        raise KeyError(f'Invalid model name: {model_name} provided.')
    

def get_test_search_params(model_name: str) -> Sequence[SearchSpaceParam]:
    """
    Retrieves a simplified hyperparameter search space for testing purposes.

    This function returns a limited and fixed search space for each model, suitable for 
    faster testing and debugging scenarios. The parameters are constrained to a minimal 
    set of values to reduce computational overhead.

    Parameters:
        model_name (str): The name of the model for which to retrieve the test search space. 
                          Valid options include "moving_avg", "expsmooth", "prophet", "xgboost", 
                          and "lightgbm".

    Returns:
        Sequence[SearchSpaceParam]: A sequence of SearchSpaceParam objects defining 
                                    the test search space for the specified model.
    """
    if model_name == "moving_avg":
        return [
            SearchSpaceParam("n_steps", "int", "categorical", choices=[1]),
        ]
    elif model_name =="expsmooth":
        return [
            SearchSpaceParam("seasonal", "str", "categorical", choices=["additive"]),
            SearchSpaceParam("trend", "str", "categorical", choices=["additive"]),
            SearchSpaceParam("seasonal_periods", "int", "categorical", choices=[7]),
            SearchSpaceParam("smoothing_level", "float", "bounded", bounds=[0.5, 0.5]),
            SearchSpaceParam("smoothing_slope", "float", "bounded", bounds=[0.5, 0.5]), 
            SearchSpaceParam("smoothing_seasonal", "float", "bounded", bounds=[0.5, 0.5]),
            SearchSpaceParam("damping_trend", "float", "bounded", bounds=[0.8,0.8]), 
        ]
    elif model_name == "prophet":
        return [
            SearchSpaceParam("automanual_changepoints", "bool", "categorical", choices = [True]), 
            SearchSpaceParam("changepoint_prior_scale", "float", "bounded", bounds = [0.5,0.5]), 
            SearchSpaceParam("seasonality_prior_scale", "float", "bounded", bounds = [10.0,10.0]),
        #    SearchSpaceParam("holiday_prior_scale", "float", "bounded", bounds = [10.0,10.0]), 
            SearchSpaceParam("seasonality_mode", "str", "categorical", choices = ["additive"]),
            SearchSpaceParam("weekly_seasonality", "bool", "categorical", choices = [True]),
        ]
    elif model_name == "xgboost":
        return [
            SearchSpaceParam("max_depth", "int", "bounded", bounds=[3, 3]), 
            SearchSpaceParam("learning_rate", "float", "bounded", bounds=[0.3, 0.3]),
            SearchSpaceParam("n_estimators", "int", "bounded", bounds=[100,100]), 
            SearchSpaceParam("min_child_weight", "int", "bounded", bounds=[1,1]), 
        ] 
    elif model_name == "lightgbm":
        return [
            SearchSpaceParam("max_depth", "int", "bounded", bounds=[3, 3]),
            SearchSpaceParam("learning_rate", "float", "bounded", bounds=[0.3, 0.3]),
            SearchSpaceParam("num_leaves", "int", "bounded", bounds=[20,20]),
            SearchSpaceParam("feature_fraction", "float", "bounded", bounds=[1.0, 1.0]), 
            SearchSpaceParam("min_data_in_leaf", "int", "bounded", bounds=[10, 10]), 
            SearchSpaceParam("lambda_l1", "float", "bounded", bounds=[10.0, 10.0], logscale=True),
            SearchSpaceParam("lambda_l2", "float", "bounded", bounds=[10.0, 10.0], logscale=True), 
        ]
    else:
        raise KeyError(f'Invalid model name: {model_name} provided.')