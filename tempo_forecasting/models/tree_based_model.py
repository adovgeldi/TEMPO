from .base_model import BaseModel
import pandas as pd
from typing import List
import json
from tempo_forecasting.utils.training_utils import enforce_datetime_index

class TreeBasedModel(BaseModel):
    """
    Abstract base class for tree-based models used to generate lagged, rolling, and 
    time-based features from a given DataFrame.

    Attributes:
        name (str): Name of the model.
        date_col (str): Name of the column containing date information in the DataFrame.
        target_y (str): Name of the target variable for prediction.
        feature_cols (list): List of feature column names generated during preprocessing.
        default_model_params (dict): A dictionary which stores the default parameters for the model.
        
    Methods:
        _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
            Takes in a one column time series df
            returns a df with the time components broken out into different columns

        _create_lag_features(df: pd.DataFrame, lags: List[int]|str) -> pd.DataFrame:
            Takes in a one column time series df and a list of ints for lag periods
            returns a df with one column for each specified lag period

        _create_rolling_features(df: pd.DataFrame, windows: List[int]|str, add_std: bool = False) -> pd.DataFrame):
            Takes in a one column time series df, a list of ints for rolling window periods, 
                and a bool for whether to build standard deviation columns.
            returns a df with one column per specified rolling window period and optional std
            
        postprocess(df: pd.DataFrame) -> pd.DataFrame):
            Detects feature columns and sets self.feature_cols
    """
    def __init__(self, 
                 name: str,
                 date_col: str,
                 target_y: str
                 ):
        """
        Initializes the TreeBasedModel with the specified parameters.

        Parameters:
            name (str): Name of the model.
            date_col (str): Name of the column containing date information in the DataFrame.
            target_y (str): Name of the target variable for prediction.
        """
        super().__init__(name,
                         date_col=date_col,
                         target_y=target_y
                         )
        self.feature_cols = []

    def _train_val_pct_split(self,
                             data,
                             target_y,
                             date_col,
                             train_pct):
        assert 0 < train_pct < 1
        
        df = data.copy()
        split_idx = int(len(df) * train_pct)
        train = df.iloc[:split_idx].copy()
        val = df.iloc[split_idx:].copy()

        # logger.debug(f"Tree-based train/validate split lengths: {len(train), len(val)}")
        return train, val
    
    def _create_time_features(self, 
                              df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates time-based features from the date column of the DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame. If `date_col` is not a column, index must be a DateTime instance.

        Returns:
            pd.DataFrame: A DataFrame with added time-based features, excluding the `date_col`.
        """
        if self.date_col not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df[self.date_col] = df.index
            else:
                raise ValueError("Missing Date Column from DataFrame and Index is not DateTime Type.")
            
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        df['dayofmonth'] = df[self.date_col].dt.day
        df['dayofweek'] = df[self.date_col].dt.dayofweek
        df['quarter'] = df[self.date_col].dt.quarter
        df['month'] = df[self.date_col].dt.month
        df['year'] = df[self.date_col].dt.year
        df['dayofyear'] = df[self.date_col].dt.dayofyear
        df['weekofyear'] = df[self.date_col].dt.isocalendar().week

        col_names = ['dayofmonth', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'weekofyear']
        for col in col_names:
            if col in df.columns:
                df[col] = df[col].astype('category')
        return df.drop(self.date_col, axis=1)
    
    def _create_lag_features(self, 
                             df: pd.DataFrame, 
                             lags: List[int]|str) -> pd.DataFrame:
        """
        Creates lagged features for the target variable.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing the `target_y` column.
            lags (List[int])|str: List of lag values to create lagged features.
                May also be string of format "[int,int,int,int...]"

        Returns:
            pd.DataFrame: A DataFrame with added lagged features.
        """
        if type(lags) == str:
            # str compatability added to avoid optuna warnings
            # Note: optuna produces a warning when passing in list parameters like lag and windows
            #       This shouldn't cause errors now, but may result in type-related storage issues some day
            #       see discussion here: https://github.com/optuna/optuna/issues/2341
            lags = json.loads(lags)

        for lag in lags:
            df[f'lag_{lag}'] = df[self.target_y].shift(lag).astype(float)
                                                
        return df
        
    def _create_rolling_features(self, 
                                 df: pd.DataFrame, 
                                 windows: List[int]|str, 
                                 add_std: bool = False) -> pd.DataFrame:
        """
        Creates rolling window features (mean and optionally std) for the target variable.

        Parameters:
            df (pd.DataFrame): Input DataFrame containing the `target_y` column.
            windows (List[int])|str: List of window sizes to calculate rolling statistics.
                May also be string of format "[int,int,int,int...]"
            add_std (Boolean): Boolean option to additionally calculate rolling standard deviation as feature.

        Returns:
            pd.DataFrame: A DataFrame with added rolling window features.
        """
        if type(windows) == str:
            # str compatability added to avoid optuna warnings
            # Note: optuna produces a warning when passing in list parameters like lag and windows
            #       This shouldn't cause errors now, but may result in type-related storage issues some day
            #       see discussion here: https://github.com/optuna/optuna/issues/2341
            windows = json.loads(windows)

        for window in windows:
            df[f'rolling_mean_{window}'] = df[self.target_y].rolling(window).mean().shift()
            if add_std:
                df[f'rolling_std_{window}'] = df[self.target_y].rolling(window, min_periods=1).std().shift()
                                                
        return df
    
    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Post-processes the DataFrame by identifying feature columns for modeling, and sets self.feature_cols

        Parameters:
            df (pd.DataFrame): Input DataFrame containing the processed features.

        Returns:
            pd.DataFrame: The post-processed DataFrame with feature columns identified.
        """

        self.feature_cols = [col for col in df.columns if col not in [self.target_y, self.date_col]]
        return df
    