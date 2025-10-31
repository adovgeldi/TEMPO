import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Sequence
from dateutil.relativedelta import relativedelta
from pandas.tseries.frequencies import to_offset


from tempo_forecasting.utils.logging_utils import logger


def calculate_metric(actual: np.array, 
                     predicted: np.array, 
                     metric: str = "wmape",
                     round_precision: int = 2
                     ) -> float:
    """
    Calculates a specified evaluation metric for actual vs. predicted values.

    Parameters:
        actual (np.array): The ground truth values.
        predicted (np.array): The predicted values from the model.
        metric (str, optional): The metric to calculate. Supported metrics include:
            - "wmape" (default): Weighted Mean Absolute Percentage Error.
            - "rmse": Root Mean Squared Error.
            - "mae": Mean Absolute Error.
        round_precision (int, optional): Number of decimal places to round to. -1 returns no rounding.

    Returns:
        float: The calculated metric value.
    """
    try:
        if metric.lower() == "wmape":
            denominator = np.sum(actual)
            if denominator == 0:
                logger.error("Division by zero encountered in WMAPE calculation. Sum of actual values is zero.")
                return np.nan
            wmape = np.sum(np.abs(np.array(actual) - np.array(predicted))) / denominator * 100
            metric = wmape
        elif metric.lower() == "rmse":
            rmse = np.sqrt(np.mean((np.array(actual) - np.array(predicted)) ** 2))
            metric = rmse
        elif metric.lower() == "mae":
            mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))
            metric = mae
        elif metric.lower() == "cid":
            # complexity_estimates
            ce_actual = np.sqrt(np.sum(np.diff(actual)**2))
            ce_pred = np.sqrt(np.sum(np.diff(predicted)**2))
            # complexity correction factor
            eps = np.finfo("float32").eps
            ccf = (max(ce_actual,ce_pred)+eps)/(min(ce_actual,ce_pred)+eps)
            # euclidean distance
            ed = np.sqrt(np.sum((predicted-actual)**2))
            # complexity invariant distance
            cid = ed*ccf
            metric = cid
        else:
            logger.error(f"Invalid metric provided: {metric}")
            raise ValueError("Provided metric not in available options. Options include 'wmape', 'mae', or 'rmse'.")

        if round_precision == -1:
            return metric
        else:
            return np.round(metric, round_precision)
    except ValueError:
        logger.error(f"Invalid metric provided: {metric}")
        raise
    except Exception as e:
        logger.error(f"Error while calculating metric '{metric}': {str(e)}")
        return np.nan
    

def enforce_datetime_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Takes in a dataframe, checks whether the index is currently a date column, and if not
    sets index to the model's date_col and reformats to datetime if necessary in order to
    ensure uniform formats across models.

    Parameters:
        df (pd.DataFrame): A dataframe which may be indexed by a column of unique dates
        date_col (str): The name of the column in df which contains date information

    Returns:
        pd.DataFrame: a reformatted version of the dataframe passed in where index is a datetime variable
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if date_col in df.columns:
            df.set_index(date_col, inplace=True)
        else:
            logger.error(f"Error: Index not DateTime and {date_col} column not present in DataFrame.")
            raise ValueError(f"Error: Index not DateTime and {date_col} column not present in DataFrame.")
        try:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
            # logger.debug("Index successfully converted to DatetimeIndex.")
        except Exception as e:
            raise ValueError(f"Failed to convert index to DatetimeIndex: {e}")
    return df.sort_index()


def calculate_time_periods(max_date_str: str,
                           n_test_months: int,
                           n_train_months: int,
                           n_validation_sets: int,
                           cv_window_step_days: int = 30,
                           verbose: bool = False):
    """
    Calculates the various date periods necessary for walk forward cross-validation and data preparation 
    in the context of time series forecasting. This function computes the dates of the full date range, 
    walk-forward cross validation windows, and a retrain window based on given parameters. 

    Parameters:
        max_date_str (str): The maximum date in string format (YYYY-MM-DD), representing
                            the most recent data point that will be used for modelig.
        n_test_months (int): The number of months to include in the test periods.
        n_train_months (int): The number of months to include in the training periods.
        n_validation_sets (int): The number of walk forward cross validation sets to use.
        cv_window_step_days (int, optional): The number of days between cv window start dates. 
                                 Default is 30.
        verbose (bool, optional): If True, will print nicely formatted, detailed information 
                                about the date ranges being calculated. Default is False.

    Returns:
        dict: A dictionary containing the following keys:
            - full_range (dict): A dictionary with keys 'total_min', 'last_cv_min', 
            'last_cv_cutoff', and 'total_max' containing the respective calculated date strings.
            - cv_windows (list): A list of dictionaries, each containing relevant dates for one 
                cross validation window. Each cv window will have an equivalent number of days as 
                the other cv windows and will be evenly spaced, as defined by the input parameters.
                Each cv_window dict contains key/value pairs:
                    'min': (str, YYYY-MM-DD), the minimum date in the training period
                    'cutoff': (str, YYYY-MM-DD), the maximum date in the training period
                    'max': (str, YYYY-MM-DD), the maximum date in the test period
            - retrain_range (dict): A dictionary with 'min' and 'max' keys representing 
                the retrain window range. Current logic sets the retrain range to all data available,
                which is the same as the full_range. 

    Notes:
        - If `n_validation_sets` is 1, the `cv_window_step_days` parameter will be ignored.
    """

    if verbose & (n_validation_sets == 1):
        print("Single validation set. cv_window_step_days will be ignored.")

    date_ranges = {}

    # Calculate Full Date Range
    # Necessary for initial data prep: min and max dates to pull from the tables
    max_date = pd.Timestamp(max_date_str)
    cutoff_date = max_date - relativedelta(months=n_test_months)  # cutoff for test set if using simple cv
    simple_cv_min_date = max_date - relativedelta(months=n_test_months+n_train_months) + relativedelta(days=1)
    total_data_min_date = (max_date 
                            - relativedelta(months=n_test_months+n_train_months) 
                            - relativedelta(days=(n_validation_sets-1)*cv_window_step_days-1))

    date_ranges["full_range"] = {
        "total_min": str(total_data_min_date)[:10],
        "last_cv_min": str(simple_cv_min_date)[:10],
        "last_cv_cutoff": str(cutoff_date)[:10],
        "total_max": max_date_str
    }

    # Calculate Validation Windows
    # Necessary for cross validation
    n_test_days = int((max_date-cutoff_date)/np.timedelta64(1, 'D'))
    n_train_days = int((cutoff_date-simple_cv_min_date)/np.timedelta64(1, 'D')) + 1 # Adding 1 day to include both min and cutoff date

    new_window_max_date = max_date
    validation_date_ranges = []
    for i in range(n_validation_sets):
        window_max_date = new_window_max_date
        window_cutoff_date = window_max_date - relativedelta(days=n_test_days)
        window_min_date = window_max_date  - relativedelta(days=n_test_days+n_train_days-1)
        window_dates = {
            "min": str(window_min_date)[:10],
            "cutoff": str(window_cutoff_date)[:10],
            "max": str(window_max_date)[:10]
        }
        validation_date_ranges = [window_dates] + validation_date_ranges

        new_window_max_date = window_max_date - relativedelta(days=cv_window_step_days)

    date_ranges["cv_windows"] = validation_date_ranges

    # Calculate Final Retrain Window
    # Note: old method likely to result in final retrain set with a slightly different number of days 
    #       than the validation train sets
    # old_min = str(max_date - relativedelta(months=n_train_months) + relativedelta(days=1))[:10]
    retrain_date_range = {
        "min": str(total_data_min_date)[:10],
        "max": max_date_str
    }
    date_ranges["retrain_range"] = retrain_date_range

    if verbose:
        for (date_type,ranges) in date_ranges.items():
            print(date_type)
            if type(ranges) == list:
                for r in ranges:
                    print("\t",r)
            else:
                print("\t",ranges)

    return date_ranges


def select_train_and_test(modeling_data: pd.DataFrame, 
                          date_col: str, 
                          min_date_str: str, 
                          cutoff_date_str: str, 
                          max_date_str: str):
    """
    Selects training and testing datasets from the provided modeling data based on 
    specified date ranges using the provided minimum date, cutoff date, and maximum date. 
    The training data consists of entries from the minimum date up to and including the cutoff date. 
    The testing data consists of entries after the cutoff date up to and including the maximum date.

    Parameters:
        modeling_data (pd.DataFrame): The complete dataset containing the time series data.
        date_col (str): The name of the column in `modeling_data` that contains date information.
        min_date_str (str): The minimum date as a string to define the start of the training set.
        cutoff_date_str (str): The cutoff date as a string to separate the training set from the test set.
        max_date_str (str): The maximum date as a string to define the end of the testing set.

    Returns:
        tuple:
            - train (pd.DataFrame): A DataFrame containing the training data filtered 
                                    based on the specified date range.
            - test (pd.DataFrame): A DataFrame containing the testing data filtered 
                                     based on the specified date range.
    """

    min_date = pd.Timestamp(min_date_str)
    cutoff_date = pd.Timestamp(cutoff_date_str)
    max_date = pd.Timestamp(max_date_str)

    assert min_date <= cutoff_date
    assert cutoff_date <= max_date

    data = enforce_datetime_index(df = modeling_data.copy(), date_col = date_col)

    train = data[(data.index >= min_date )&
                 (data.index <= cutoff_date)]
    
    test = data[(data.index > cutoff_date )&
                (data.index <= max_date)]
    
    return train, test


def train_and_validate(model,
                       model_param_dict: Dict[str,Any],
                       train_data: pd.DataFrame,
                       val_data: pd.DataFrame,
                       metric_types: Sequence[str],
                       round_precision = 2):
    """
    Trains the given model on the training data and validates it using the validation data.

    This function fits the model with the specified parameters on the training data, 
    offers predictions on the validation data, and calculates the specified evaluation metrics. 
    The results include the calculated metrics, the fitted values from the training dataset, 
    and the predicted values from the validation dataset.

    Parameters:
        model: The model class instance to be trained and validated.
        model_param_dict (dict): A dictionary of parameters to be used for fitting the model.
        train_data (pd.DataFrame): The training dataset utilized for fitting the model.
        val_data (pd.DataFrame): The validation dataset used for evaluating the model's performance.
        metric_types (list): A list of metric names to be calculated on the validation predictions.
            Note that all metric_type values must be valid metrics available via calculate_metric.
        round_precision (int, optional): The number of decimal places to round the metric results. 
            Default is 2.

    Returns:
        dict: A dictionary containing:
            - metrics (dict): A dictionary of calculated metrics with metric names as keys and their respective values.
            - fitted_train_vals (np.ndarray): The fitted values obtained from the model on the training data.
            - test_pred_vals (np.ndarray): The predicted values from the model on the validation data.
    """
    
    model.fit(train_data, model_param_dict)
    fitted_train_vals = model.fitted_vals

    y_pred = model.predict(val_data)
    y_actual = val_data[model.target_y]
    metrics = {}
    for m in metric_types:
        metrics[m.upper()] = calculate_metric(y_actual, y_pred, metric=m, round_precision = round_precision)
    
    results = {
        "metrics": metrics,
        "fitted_train_vals": fitted_train_vals,
        "test_pred_vals": y_pred,
    }

    return results


def cross_validate(data: pd.DataFrame, 
                   date_col: str, 
                   target_col: str, 
                   model_class, 
                   model_param_dict: Dict[str,Any], 
                   cv_dates: Sequence[Tuple[str, str, str]], 
                   metrics: Sequence[str]):
    """
    Performs cross-validation on a specified time series model over any number of cross validation windows 
    using the data and cross-validation dates provided.

    For each set of cross validation dates provided (min, cutoff, and max), this function selects the 
    relevant time period and splits the data into train (min-cutoff) and test [cutoff-max) datasets. 
    Then it trains the specified model using the specified parameters on the train set, and evaluates 
    performance on the test set, returning the specified performance metrics from the test set.

    Parameters:
        data (pd.DataFrame): The dataset containing the date and time series values for the model.
                             Must contain the date_col and target_col.
        date_col (str): The name of the column containing date information.
        target_col (str): The name of the target variable to be predicted by the model.
        model_class: The class of the model to be used for training and validation.
        model_param_dict (Dict[str, Any]): A dictionary containing parameters for the model
        cv_dates (Sequence[Tuple[str, str, str]]): A list of tuples containing the minimum, cutoff, 
                            and maximum dates for each cross-validation iteration.
        metrics (Sequence[str]): A list of metric names to evaluate the model performance on 
                            the validation datasets. Metrics names must be compatible with
                            the calculate_metric function.

    Returns:
        List[Dict[str, Any]]:
            A list of dictionaries where each dictionary contains the results of the cross-validation 
            for a given date range, including: 
                - metrics
                - fitted training values
                - validation predictions
                - specific dates used for that cross-validation window
    """
    
    modeling_data = data.copy().reset_index()#[[date_col,target_col]]
    
    model = model_class(target_y=target_col, date_col=date_col)
    results = []

    for (cv_min, cv_cutoff, cv_max) in cv_dates:
        train_data, val_data = select_train_and_test(modeling_data = modeling_data, 
                                                    date_col = model.date_col, 
                                                    min_date_str = cv_min, 
                                                    cutoff_date_str = cv_cutoff, 
                                                    max_date_str = cv_max)
        
        cv_results = train_and_validate(model = model,
                                        model_param_dict = model_param_dict,
                                        train_data = train_data,
                                        val_data = val_data,
                                        metric_types = metrics,
                                        round_precision = 2)
        
        # cv results already contains metrics, fitted_train_vals, and test_pred_vals 
        # for this round of cv
        cv_results["dates"] = [cv_min, cv_cutoff, cv_max]
        results += [cv_results]
    
    return results


def _validate_pandas_completeness(df: pd.DataFrame, 
                                  date_col: str, 
                                  freq: str, 
                                  group_col: str = None
                                  ) -> None:
    """
    Validate that the pandas DataFrame has no missing time steps for the specified frequency by checking for missing dates based on the specified frequency.

    Parameters:
        df (pd.DataFrame): DataFrame containing the time series data
        date_col (str): Name of the date column in the DataFrame
        freq (str): Frequency string (e.g., 'D' for daily, 'W' for weekly, 'W-SUN' for weekly starting on Sunday)
        group_col (str, optional): If provided, check the completeness within each group

    Raises:
        ValueError: If any expected dates are missing.
    """
    freq = to_offset(freq).freqstr
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])

    if group_col:
        missing_all = []

        for key, group in data.groupby(group_col):
            group = group.sort_values(by=date_col)
            expected = pd.date_range(start=group[date_col].min(), 
                                     end=group[date_col].max(), 
                                     freq=freq)
            actual = pd.to_datetime(pd.Series(group[date_col].unique()))
            missing = expected.difference(actual)

            if not missing.empty:
                missing_all.append((key, missing))
            
        if missing_all:
            msg = "\n".join([f"Group '{key}': Missing dates: {list(missing)}" for key, missing in missing_all])
            raise ValueError("Missing time steps in pandas DataFrame:\n{msg}")
        else:
            print("No missing time steps found in pandas DataFrame.")   
        
    else:
        expected = pd.date_range(start=data[date_col].min(), 
                                 end=data[date_col].max(), 
                                 freq=freq)
        actual = pd.DatetimeIndex(data[date_col].unique())
        missing = expected.difference(actual)

        if not missing.empty:
            raise ValueError(f"Missing time steps in pandas DataFrame: {list(missing)}")    
        else:
            print("No missing time steps found in pandas DataFrame.")    


def _validate_pyspark_completeness(df,
                                   date_col: str,
                                   freq: str,
                                   group_col: str = None
                                   ) -> None:
    """
    Validate that the pandas DataFrame has no missing time steps for the specified frequency by checking for missing dates based on the specified frequency.

    Parameters:
        df (pyspark.sql.DataFrame): PySpark DataFrame containing the time series data
        date_col (str): Name of the date column in the DataFrame
        freq (str): Frequency string. Currently supports 'D' (daily), 'W' (weekly), 'M' (monthly)
        group_col (str, optional): If provided, check the completeness within each group

    Raises:
        ValueError: If any expected dates are missing.
    """
    freq = to_offset(freq).freqstr.upper()
    supported_freqs = {'D': 'DAY', 'W': 'WEEK', 'M': 'MONTH'}

    if freq.startswith("D"):
        interval = "1 DAY"
    elif freq.startswith("W"):
        interval = "1 WEEK"
    elif freq.startswith("M"):
        interval = "1 MONTH"
    else:
        raise ValueError(f"Unsupported frequency '{freq}' for PySpark completeness check. Supported frequencies are: {list(supported_freqs.keys())}")
    
    if group_col:
        df_min_max = df.groupBy(group_col).agg(
            F.min(F.col(date_col)).alias("min_date"),
            F.max(F.col(date_col)).alias("max_date")
        )

        df_expected = df_min_max.withColumn("expected_dates",
                                            F.sequence(F.col("min_date"), F.col("max_date"), F.lit(interval))
        ).select(group_col, F.explode(F.col("expected_dates")).alias("expected_date"))

        df_actual = df.select(F.col(group_col), F.col(date_col).alias("actual_date")).dropDuplicates()

        missing = (df.expected.join(df_actual,
                                    (df_expected[group_col] == df_actual[group_col]) &
                                    (df_expected.expected_date == df_actual.actual_date),
                                    how = "left_anti"))
        if missing.limit(1).count() > 0:
            sample = missing.limit(10).toPandas()
            raise ValueError(f"Missing time steps in PySpark DataFrame. Example:\n{sample}")
        else:
            print("No missing time steps found in PySpark DataFrame.")
            
    else: 
        minmax = df.agg(
            F.min(F.col(date_col)).alias("min_date"),
            F.max(F.col(date_col)).alias("max_date")
        ).collect()[0]

        start, end = minmax['min_date'], minmax['max_date']
        expected = df.SparkSession.createDataFrame(
            pd.date_range(start=start, end=end, freq=freq).to_frame(index=False, name="expected_date")
        )
        actual = df.select(F.col(date_col).alias("actual_date")).dropDuplicates()

        missing = expected.join(actual,
                                expected.expected_date == actual.actual_date,
                                how="left_anti")
        
        if missing.limit(1).count() > 0:
            sample = missing.limit(10).toPandas()
            raise ValueError(f"Missing time steps in PySpark DataFrame Example:\n{sample}")
        else:
            print("No missing time steps found in PySpark DataFrame.")


def validate_time_series_completeness(
        df, 
        date_col: str, 
        freq: str, 
        group_col: str = None
        ) -> None:
    """
    Validate that the time series data has no missing time steps for the specified frequency.
    Supports both pandas and PySpark DataFrames.
    
    Parameters:
        df (DataFrame; either pandas or pyspark): DataFrame containing the time series data
        date_col (str): Name of the date column in the DataFrame
        freq (str): Frequency string (e.g., 'D' for daily, 'W' for weekly)
        group_col (str, optional): If provided, check the completeness within each group
    
    Raises:
        ValueError: If any expected dates/timestamps are missing.
    """
    is_pandas = isinstance(df, pd.DataFrame)

    if is_pandas:
        _validate_pandas_completeness(df, date_col, freq, group_col)
    else:
        _validate_pyspark_completeness(df, date_col, freq, group_col)