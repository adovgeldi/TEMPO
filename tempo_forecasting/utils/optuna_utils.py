from tempo_forecasting.optuna_opt.optuna_param import SearchSpaceParam
from typing import Sequence


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
            SearchSpaceParam("n_days", "int", "categorical", choices=[1,7,14,30,60,90,120]),
        ]
    elif model_name =="expsmooth":
        return [
            SearchSpaceParam("seasonal", "str", "categorical", choices=["additive", "multiplicative"]),
            SearchSpaceParam("seasonal_periods", "int", "categorical", choices=[7,14,30,60,90,365]), # weekly intervaled seasonality
            SearchSpaceParam("smoothing_level", "float", "bounded", bounds=[0.1, 1.0], step_size=0.1), # smoothing factor for data
            SearchSpaceParam("smoothing_trend", "float", "bounded", bounds=[0.01, 0.46], step_size=0.05), # smoothing factor for trend
            SearchSpaceParam("smoothing_seasonal", "float", "bounded", bounds=[0.01, 0.46], step_size=0.05), # smoothing factor for seasonality
            SearchSpaceParam("damping_trend", "float", "bounded", bounds=[0.8,0.9], step_size=0.02), # damping_slope for longer-term forecasts
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
            SearchSpaceParam("n_estimators", "int", "bounded", bounds=[100,1000], step_size=100), # higher values should help with more complex seasonality patterns
            SearchSpaceParam("min_child_weight", "int", "bounded", bounds=[1,7]), # higher values can help combat overfitting
            SearchSpaceParam("windows", "str", "categorical", choices = ["[7, 30, 90, 365]","[365,90,45,30,14,7,2,1]"]), # also accepts list(int), but that triggers a warning
            SearchSpaceParam("lags", "str", "categorical", choices = ["[1, 7, 30]","[]"])
        ] 
    elif model_name == "lightgbm":
        return [
            SearchSpaceParam("max_depth", "int", "bounded", bounds=[3, 8]), # used to combat overfitting; larger value = larger complexity
            SearchSpaceParam("learning_rate", "float", "bounded", bounds=[0.01, 0.3], logscale=True), # controls how much each new decision tree contributes to final; smaller = more stable predictions
            SearchSpaceParam("num_leaves", "int", "bounded", bounds=[20,200], step_size=10), # helps control model complexity
            SearchSpaceParam("feature_fraction", "float", "bounded", bounds=[0.5, 1.0], step_size=0.1), # 50% subset of features to include on each iteration; helpful to combat overfitting
            SearchSpaceParam("min_data_in_leaf", "int", "bounded", bounds=[10, 100], step_size=10), # may help: higher values for more expensive machines with sparse rentals
            SearchSpaceParam("lambda_l1", "float", "bounded", bounds=[1e-4, 10.0], logscale=True), # L1 regularization
            SearchSpaceParam("lambda_l2", "float", "bounded", bounds=[1e-4, 10.0], logscale=True), # L2 regularization
            SearchSpaceParam("windows", "list", "categorical", choices = ["[7, 30, 90, 365]","[365,90,45,30,14,7,2,1]"]),
            SearchSpaceParam("lags", "list", "categorical", choices = ["[1, 7, 30]","[]"])
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
            SearchSpaceParam("n_days", "int", "categorical", choices=[1]),
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