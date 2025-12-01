import pandas as pd
from pandas.tseries.frequencies import to_offset

import numpy as np
from typing import Dict, Any, Tuple, Optional

from tempo_forecasting.utils.training_utils import select_train_and_test
from tempo_forecasting.utils.config_utils import get_models

import traceback
from tempo_forecasting.models import MovingAvgModel



def _approx_days_per_step(freq: str):
    """
    Approximates the number of days per time step based on the frequency string.
    
    Supports daily ('D'), weekly ('W', 'W-SUN', 'W-MON'), monthly ('M', 'MS'), and quarterly ('Q') frequencies.
    
    Parameters:
        freq (str): The time frequency for the data (e.g., 'D', 'W', 'M', 'Q').
        
    Returns:
        int: Approximate number of days per time step.
    """

    f = to_offset(freq)
    name = f.name.upper()

    if name.startswith('W'):      # Weekly frequency (e.g., 'W', 'W-SUN', 'W-MON')
        return 7
    if name.startswith('M'):      # Monthly frequency (e.g., 'M', 'MS')
        return 30
    if name.startswith('Q'):      # Quarterly frequency
        return 90
    else:                         # Default to daily frequency
        return 1
    

def get_base_min_train_days(
        model_name: str,
        freq: str
        ) -> int:
    """
    Compute the minimun number of training steps required for a given model and frequency.

    This function is designed to beL
    - **Model-aware**: Different models have different data needs.
    - **Frequency-aware**: Requirements scale witht the coareness of the time step.

    For each model we specify:
    - min_years[model_name]: How many years of data are minimally required.
    - min_points[model_name]: How many total data points are minimally required, regardless of frequency..
    
    Parameters:
        model_name (str): The name of the model.
        freq (str): The time frequency for the data (e.g., 'D', 'M', 'MS', 'Q', 'Q-DEC', etc.).
    
    Returns:
        int: The minimum number of training steps required for the given model and frequency.

    Raises:
        ValueError: If the model_name is not recognized or freq is not a valid pandas offset alias.
    """

    min_years: Dict[str, float] = {
        "moving_avg": 0.25,
        "expsmooth": 2.0,
        "prophet": 1.5,
        "xgboost": 2.0, 
        "lightgbm": 2.0,
        "knn": 2.0
    }

    min_points: Dict[str, int] = {
        "moving_avg": 20,
        "expsmooth": 30,
        "prophet": 50,
        "xgboost": 100,
        "lightgbm": 100,
        "knn": 100
    }
    
    if model_name not in min_years or model_name not in min_points:
        raise ValueError(
            f"Invalid model name: '{model_name}'. "
            f"Allowed models: {list(min_years.keys())}"
            )
    
    # Validate frequency
    try:
        _ = to_offset(freq) 
    except ValueError as e:
        raise ValueError(
            f"Invalid frequency '{freq}'. "
            "Use a pandas-recognized frequency like 'D', 'W', 'W-SUN', 'M', etc.)."
            ) from e
    
    step_days = _approx_days_per_step(freq)
    years_required = min_years[model_name]
    points_required = min_points[model_name]

    required_by_years = np.ceil((years_required * 365) / step_days)
    required_by_points = points_required

    required_steps = max(required_by_years, required_by_points)

    return max(required_steps, 1)


def get_train_horizon_multipliers(model_name: str) -> int:
    """
    Returns the train horizon multiplier for a given model.

    This multiplier defines how many multiples of the forecast horizon we want as a minimum training window, on top of the base requirement.
    That is, if we want to forecast 6 months out, and the multiplier is 3, we want at least 18 months of training data.
    
    Parameters:
      - model_name (str): The name of the model, ["moving_avg","expsmooth","prophet","xgboost","lightgbm"]
    
    Returns:
      - int: The multiplier for determining train horizon length.
    """
    multipliers = {
        "moving_avg": 2,  
        "knn": 3,
        "expsmooth": 3,  
        "prophet": 3,  
        "xgboost": 3,  
        "lightgbm": 3  
    }
    
    if model_name not in multipliers:
        raise ValueError(f"Invalid model name: '{model_name}'. Allowed models: {list(multipliers.keys())}")
    
    return multipliers[model_name]


def get_forecast_horizon(category_data: pd.DataFrame, 
                         args: Dict[str, Any]
                         ) -> int:
    """
    Computes forecast horizon (in steps) based on the data and the final cross-validation window.

    The horizon is defined as the number of periods between the CV cutoff date and the last date that is both in the data and within the CV test window.
    
    Parameters:
      - category_data (pd.DataFrame): DataFrame containing time series data.
      - args (Dict[str, Any]): A dictionary containing, at minimum, the following keys:
            - 'date_col' (str): The name of the column containing datetime information.
            - 'freq' (str): The time frequency for the data (e.g., 'D' for daily, 'M' for monthly).
            - 'cv_dates' (Sequence[Sequence[str]]): a list of lists of dates defining cross validation windows  
    Returns:
      - int: Forecast horizon in steps; the number of time steps to forecast into the future.
    """
    date_col = args['date_col']
    freq_arg = args['freq']

    # Check frequency argument (raises error if invalid)
    try:
        freq_name = to_offset(freq_arg).name # e.g., 'D', 'W', 'W-SUN', 'M', etc.
    except ValueError as e:
        raise ValueError(
            f"Invalid frequency '{freq_arg}'."
            "Use a pandas-recognizd frequency like 'D', 'W', 'W-SUN', 'M', etc.)."
            ) from e

    # Ensure the date column is in datetime format
    category_data[date_col] = pd.to_datetime(category_data[date_col])
    
    # Get the latest available date in the dataset
    data_max_date = category_data[date_col].max()

    # Get the cv test window for the final set of cv dates
    cv_cutoff_date = pd.Timestamp(args["cv_dates"][-1][1])
    cv_max_date = pd.Timestamp(args["cv_dates"][-1][2])

    # Get the true max:
    # last day in the cv test window that exists in the data
    max_date = min(data_max_date,cv_max_date)

    # If nothing to forecast, horizon is 0
    if pd.isna(max_date) or pd.isna(cv_cutoff_date) or max_date <= cv_cutoff_date:
        return 0
    
    pr = pd.period_range(start=cv_cutoff_date, end=max_date, freq=freq_name)
    horizon = max(len(pr)-1, 0)

    return horizon


def get_dynamic_min_train_days(model_name: str, 
                               forecast_horizon: int,
                               freq: str
                               ) -> int:
    """
    Computes the dynamic minimum number of training steps required for a model, given the forecast horizon and frequency.
    
    Logic: 
    - Start with the base requirement:
            base_steps = get_base_min_train_days(model_name, freq) 
        which enforces a minimum calendar span AND minimum point count.
    - Then compute a horizon-scaled requirement:
            dynamic_steps = get_train_horizon_multipliers(model_name) * forecast_horizon
        to ensure we have multiple horizons lengths of historoical data.
    - Take the maximum of the two:
            required_steps = max(base_steps, dynamic_steps)

    Parameters:
        - model_name (str): The name of the model, ["moving_avg","expsmooth","prophet","xgboost","lightgbm"]
        - forecast_horizon (int): Number of time steps to forecast into the future.
        - freq (str): The time frequency for the data (e.g., 'D' for daily, 'M' for monthly, etc.).

    Returns:
        - int: the minimum required training days based on the forecast horizon
    """
    base_steps = get_base_min_train_days(model_name, freq)
    multiplier = get_train_horizon_multipliers(model_name)

    dynamic_min_steps = multiplier * max(forecast_horizon, 0)
    required_steps = max(base_steps, dynamic_min_steps)

    print(f"Model {model_name}: base_steps={base_steps}, dynamic_min_steps={dynamic_min_steps}")
    return required_steps


def check_train_data_sufficiency(
        train_series: pd.Series, 
        model_name: str, 
        forecast_horizon: int,
        freq: str
        ) -> bool:
    """
    Ensures the training set has enough data relative to the forecast horizon, given the specified frequency.

    Parameters:
        - train_series (pd.Series): The training data
        - model_name (str): The name of the model, ["moving_avg","expsmooth","prophet","xgboost","lightgbm"]
        - forecast_horizon (int): Number of time steps to forecast into the future.
        - freq (str): The time frequency for the data (e.g., 'D' for daily, 'M' for monthly).

    Returns:
        - bool: whether or not the training data contains enough data to reasonably forecast into the
                requested forecast horizon
    """
    required_train_steps = get_dynamic_min_train_days(
        model_name=model_name, 
        forecast_horizon=forecast_horizon, 
        freq=freq
        )
    # Work with a simple integer index
    train_series = train_series.reset_index(drop=True)

    nonzero_val_indices = np.array(train_series[train_series > 0].index)
    if len(nonzero_val_indices) > 0:
        len_train_days_in_nonzero_range = int(max(nonzero_val_indices) - min(nonzero_val_indices) + 1)
    else:
        len_train_days_in_nonzero_range = 0

    sufficient_data = (len_train_days_in_nonzero_range >= required_train_steps)
    # suff_data_str = f"({len_train_days_in_nonzero_range} >= {required_train_steps} = {sufficient_data})"

    return sufficient_data


def preprocess_pipeline(category: str,
                        category_data: pd.DataFrame, 
                        args: Dict[str, Any],
                        logger = None
                        ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Checks if the dataset has enough training data for the forecast horizon.
    
    - Ensures that models have **sufficient training data**.
    - Ensures at least 2 weeks of **test data**.
    - If all models are removed, defaults to `moving_avg`.

    Parameters:
        - category (str): Name of the dataset being processed, for logging purposes
        - category_data (pd.Dataframe): All of the time series data to be used in the model
        - args (Dict[str, Any]): A dictionary containing the following keys:
            - 'date_col' (str): The name of the column containing datetime information.
            - 'target_y' (str): The name of the target variable column.
            - 'min_date' (str): The minimum date to include in the dataset (format: 'YYYY-MM-DD').
            - 'cutoff_date' (str): The date marking the start of the test dataset (format: 'YYYY-MM-DD').
            - 'freq' (str): The time frequency for the data (e.g., 'D' for daily, 'M' for monthly).
            - 'validate' (bool): Whether or not to use two periods of testing data in order to validate the model

    Returns:
        - model_set (dict): Dictionary of models that have enough data for training.
        - train_data (pd.DataFrame): The data to use to train the model
        - test_data (pd.DataFrame): The data to use to test the model
    """
    if logger is None: # Use standard logging
        from tempo_forecasting.utils.logging_utils import logger as default_logger
        logger = default_logger
        log_func_info = lambda msg: logger.info(msg)
        log_func_warn = lambda msg: logger.warning(msg)
        log_func_error = lambda msg: logger.error(msg)
    elif logger == "print":
        log_func_info = lambda msg: print(msg)
        log_func_warn = lambda msg: print(msg)
        log_func_error = lambda msg: print(msg)
    else:
        # Use the WorkerLogger's category-aware logging if passed in
        log_func_info = lambda msg, details="Preprocessing Pipeline": logger.info(message=msg, category=category, details=details)
        log_func_warn = lambda msg, details="Preprocessing Pipeline": logger.warning(message=msg, category=category, details=details)
        log_func_error = lambda msg, details="Preprocessing Pipeline": logger.error(message=msg, category=category, details=details)

    try:
        date_col = args['date_col']
        target_y = args['target_y']
        freq = args['freq']

        # Determine how long to forecast out
        forecast_horizon = get_forecast_horizon(category_data, args=args)

        # Break out the minimum necessary dataset: the final cv train/test sets
        final_cv_dataset_dates = args["cv_dates"][-1]
        final_cv_train, final_cv_test = select_train_and_test(modeling_data = category_data, 
                                                date_col = date_col, 
                                                min_date_str = final_cv_dataset_dates[0],
                                                cutoff_date_str = final_cv_dataset_dates[1], 
                                                max_date_str = final_cv_dataset_dates[2])
        
        final_cv_train_series = final_cv_train[target_y]
        final_cv_test_series = final_cv_test[target_y]
        
        if final_cv_train.empty or final_cv_test.empty:
            log_func_error(f"Skipping category {category} due to insufficient data.")
            raise ValueError(f"Category {category} has insufficient data. Either train or test set are empty.")
        
        log_func_info(f"Category {category} has data in train and test.")

        # Enforce volume requirement
        # Data sets with low values in the target_y col have insufficient data for complex models
        simple_model_cutoff = args['min_target_value']
        get_models_type = "simple" if np.max(final_cv_test_series) < simple_model_cutoff else "default"
        candidate_models = get_models(how = get_models_type)

        # Enforce duration requirement for models
        # Remove models that don't have enough **training data** in the final cv window
        final_model_set = {
            model_name: model_class
            for model_name, model_class in candidate_models.items()
            if check_train_data_sufficiency(final_cv_train_series, model_name, forecast_horizon, freq)
        }

        # If all models were removed, default to moving_avg
        if not final_model_set:
            log_func_warn(f"Category {category}: Insufficient training data for any model, defaulting to moving_avg.")
            final_model_set = {'moving_avg': MovingAvgModel}

        log_func_info(f"Category {category} generic candidate models: {candidate_models.keys()}.")
        log_func_info(f"Category {category} using models: {final_model_set.keys()} after data sufficiency check.")

        # Enforce duration requirement for cv windows
        # We have verified that the last cv window has sufficient data, but all others must be confirmed
        revised_cv_dates = []

        if len(args["cv_dates"])>1:
            for cv_dates in args["cv_dates"][:-1]:
                suff_data_for_cv_dates = []
                for model_name in final_model_set:
                    cv_train_series, _ = select_train_and_test(modeling_data = category_data, 
                                                    date_col = date_col, 
                                                    min_date_str = cv_dates[0],
                                                    cutoff_date_str = cv_dates[1], 
                                                    max_date_str = cv_dates[2])
                    
                    window_has_enough_data = check_train_data_sufficiency(cv_train_series, model_name, forecast_horizon, freq)
                    # print(f"For window {cv_dates},{model_name} has sufficient_data = {window_has_enough_data}")

                    suff_data_for_cv_dates += [check_train_data_sufficiency(cv_train_series, model_name, forecast_horizon, freq)]

                all_models_have_sufficient_data = min(suff_data_for_cv_dates)
                if all_models_have_sufficient_data:
                    revised_cv_dates += [cv_dates]

        revised_cv_dates += [args["cv_dates"][-1]]
        if len(revised_cv_dates) == len(args["cv_dates"]):
            revised_cv_dates = None
        log_func_info(f"Category {category} revised cv dates: {revised_cv_dates}")

            
        return final_model_set, revised_cv_dates, logger
    
    except Exception as e:
        log_func_error(f"Error during hyperparameter optimization: {str(e)}; {traceback.format_exc()}")
        empty_df = pd.DataFrame()
        return {}, empty_df, empty_df, logger