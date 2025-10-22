import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional

from tempo_forecasting.utils.training_utils import select_train_and_test
from tempo_forecasting.utils.config_utils import get_models

import traceback
from tempo_forecasting.models import MovingAvgModel


def get_base_min_train_days(model_name: str) -> int:
    """
    Returns the base minimum number of training days required for a given model.
    
    Parameters:
        model_name (str): The name of the model.
    
    Returns:
        int: The minimum number of training days required for the model.
    """
    base_min = {
        "moving_avg": 60,  # At least 2 months of training data
        "knn": 540,        # At least 1.5 years of training data
        "expsmooth": 365*2,# At least 2 years of training data
        "prophet": 180,    # At least 6 months of training data
        "xgboost": 180,    # At least 3 months of training data
        "lightgbm": 180    # At least 3 months of training data
    }
    
    if model_name not in base_min:
        raise ValueError(f"Invalid model name: '{model_name}'. Allowed models: {list(base_min.keys())}")
    
    return base_min[model_name]


def get_train_horizon_multipliers(model_name: str) -> int:
    """
    Returns the train horizon multiplier for a given model.
    
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
    Computes forecast horizon based on `args['freq']`.
    
    Parameters:
      - category_data (pd.DataFrame): DataFrame containing time series data.
       - args (Dict[str, Any]): A dictionary containing, at minimum, the following keys:
            - 'date_col' (str): The name of the column containing datetime information.
            - 'freq' (str): The time frequency for the data (e.g., 'D' for daily, 'M' for monthly).
            - 'cv_dates' (Sequence[Sequence[str]]): a list of lists of dates defining cross validation windows  
    Returns:
      - int: Forecast horizon, the number of time steps to forecast into the future.
    """
    # Ensure the date column is in datetime format
    category_data[args['date_col']] = pd.to_datetime(category_data[args['date_col']])
    
    # Get the latest available date in the dataset
    data_max_date = category_data[args['date_col']].max()

    # Get the cv test window for the final set of cv dates
    cv_cutoff_date = pd.Timestamp(args["cv_dates"][-1][1])
    cv_max_date = pd.Timestamp(args["cv_dates"][-1][2])

    # Get the true max:
    # last day in the cv test window that exists in the data
    max_date = min(data_max_date,cv_max_date)

    # Compute raw difference in days
    horizon_days = (max_date - cv_cutoff_date).days

    # Frequency mapping to numeric step size
    freq_mapping = {
        "D": 1,   # Daily
        "W": 7,   # Weekly (1 step = 7 days)
        "M": 30,  # Monthly (approximate as 30 days)
        "Q": 90,  # Quarterly (approximate as 90 days)
        "Y": 365  # Yearly (approximate as 365 days)
    }

    # Get frequency step size (default to daily)
    step_size = freq_mapping.get(args.get('freq', "D"))
 
    if step_size is None:
        raise ValueError(f"Invalid frequency '{args['freq']}'. Allowed values: {list(freq_mapping.keys())}")

    # Convert horizon to frequency-based time steps
    forecast_horizon = max(horizon_days // step_size, 0)

    return forecast_horizon


def get_dynamic_min_train_days(model_name: str, 
                               forecast_horizon: int
                               ) -> int:
    """
    Computes the minimum required training days based on the forecast horizon.
    Uses a model-specific multiplier to ensure sufficient historical data.

    Parameters:
        - model_name (str): The name of the model, ["moving_avg","expsmooth","prophet","xgboost","lightgbm"]
        - forecast_horizon (int): Number of time steps to forecast into the future.

    Returns:
        - int: the minimum required training days based on the forecast horizon
    """
    default_min_days = get_base_min_train_days(model_name)
    dynamic_min_days = get_train_horizon_multipliers(model_name) * forecast_horizon
    return max(default_min_days, dynamic_min_days)


def check_train_data_sufficiency(train_series: pd.Series, 
                                model_name: str, 
                                forecast_horizon: int
                                ) -> bool:
    """
    Ensures the training set has enough data relative to the forecast horizon.

    Parameters:
        - train_series (pd.Series): The training data
        - model_name (str): The name of the model, ["moving_avg","expsmooth","prophet","xgboost","lightgbm"]
        - forecast_horizon (int): Number of time steps to forecast into the future.

    Returns:
        - bool: whether or not the training data contains enough data to reasonably forecast into the
                requested forecast horizon
    """
    required_train_days = get_dynamic_min_train_days(model_name, forecast_horizon)

    train_series = train_series.reset_index(drop=True)
    nonzero_val_indices = np.array(train_series[train_series > 0].index)
    if len(nonzero_val_indices) > 0:
        len_train_days_in_nonzero_range = max(nonzero_val_indices) - min(nonzero_val_indices) + 1
    else:
        len_train_days_in_nonzero_range = 0

    sufficient_data = (len_train_days_in_nonzero_range >= required_train_days)

    suff_data_str = f"({len_train_days_in_nonzero_range} >= {required_train_days} = {sufficient_data})"

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
        # Determine how many days to forecast out
        forecast_horizon = get_forecast_horizon(category_data, args=args)

        # Break out the minimum necessary dataset: the final cv train/test sets
        final_cv_dataset_dates = args["cv_dates"][-1]
        final_cv_train, final_cv_test = select_train_and_test(modeling_data = category_data, 
                                                date_col = args["date_col"], 
                                                min_date_str = final_cv_dataset_dates[0],
                                                cutoff_date_str = final_cv_dataset_dates[1], 
                                                max_date_str = final_cv_dataset_dates[2])
        
        final_cv_train_series = final_cv_train[args['target_y']]
        final_cv_test_series = final_cv_test[args['target_y']]
        
        if final_cv_train.empty or final_cv_test.empty:
            log_func_error(f"Skipping category {category} due to insufficient data.")
            raise ValueError(f"Category {category} has insufficient data. Either train or test set are empty.")
        
        log_func_info(f"Category {category} has data in train and test.")

        # Enforce volume requirement
        # Data sets with low values in the target_y col have insufficient data for complex models
        simple_model_cutoff = 5
        get_models_type = "simple" if np.max(final_cv_test_series) < simple_model_cutoff else "default"
        candidate_models = get_models(how = get_models_type)

        # Enforce duration requirement for models
        # Remove models that don't have enough **training data** in the final cv window
        final_model_set = {
            model_name: model_class
            for model_name, model_class in candidate_models.items()
            if check_train_data_sufficiency(final_cv_train_series, model_name, forecast_horizon)
        }

        # If all models were removed, default to moving_avg
        if not final_model_set:
            log_func_warn(f"Category {category}: Insufficient training data for any model, defaulting to moving_avg.")
            final_model_set = {'moving_avg': MovingAvgModel}

        log_func_info(f"Category {category} using models: {final_model_set.keys()}")

        # Enforce duration requirement for cv windows
        # We have verified that the last cv window has sufficient data, but all others must be confirmed
        revised_cv_dates = []

        if len(args["cv_dates"])>1:
            for cv_dates in args["cv_dates"][:-1]:
                suff_data_for_cv_dates = []
                for model_name in final_model_set:
                    cv_train_series, _ = select_train_and_test(modeling_data = category_data, 
                                                    date_col = args["date_col"], 
                                                    min_date_str = cv_dates[0],
                                                    cutoff_date_str = cv_dates[1], 
                                                    max_date_str = cv_dates[2])
                    
                    window_has_enough_data = check_train_data_sufficiency(cv_train_series, model_name, forecast_horizon)
                    # print(f"For window {cv_dates},{model_name} has sufficient_data = {window_has_enough_data}")

                    suff_data_for_cv_dates += [check_train_data_sufficiency(cv_train_series, model_name, forecast_horizon)]

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