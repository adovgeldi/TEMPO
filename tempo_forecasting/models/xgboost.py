from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from typing import Dict, Any
from .tree_based_model import TreeBasedModel

from tempo_forecasting.utils.training_utils import enforce_datetime_index
from tempo_forecasting.utils.logging_utils import logger


class XGBoostModel(TreeBasedModel):
    """
    A time series forecasting model that leverages the XGBoost algorithm. 
    This class extends the TreeBasedModel base class and provides functionality 
    for training and predicting time series data with lagged and rolling window features.

    Attributes:
        date_col (str): The name of the column containing date information.
        target_y (str): The name of the target variable to be forecasted.
        model (LGBMRegressor): The LightGBM regression model used for training and prediction.
        train (pd.DataFrame): The training dataset used to fit the model.
        fitted_vals (np.ndarray): Array of fitted values for the training time period 
        feature_cols (List[str]): List of feature column names after preprocessing.
        lags (List[int]): List of lag values for creating lagged features. Default is [1, 7, 30].
        windows (List[int]): List of rolling window sizes for feature engineering. Default is [7, 30, 90, 365].
        default_model_params (dict): A dictionary which stores the default parameters for the model.
        
    Methods:
        fit(train_data: pd.DataFrame, model_param_dict: Dict[str, Any]) -> None:
            Abstract method to fit the model to the given training data.

        predict(test_data: pd.DataFrame) -> np.ndarray:
            Abstract method to generate predictions for the specified future dates.
    """
    def __init__(self, 
                 date_col: str,
                 target_y: str,
                 ):
        """
        Initializes the XGBoostModel with the specified configurations.

        Parameters:
            date_col (str): The name of the column containing date information.
            target_y (str): The name of the target variable to be forecasted.
            model_params (Dict[str, Any], optional): Hyperparameters for the XGBRegressor model. 
                If None, default parameters are used.
        """
        super().__init__(name="xgboost",
                         date_col = date_col,
                         target_y = target_y
                        )
        
        self.model = None
        self.fitted_vals = None
        self.train = None
        self.lags = None    # Now set in .fit()
        self.windows = None # Now set in .fit()
        logger.debug(f"Initialized XGBoostModel with target_y: '{target_y}' and date_col: '{date_col}'")


    def fit(self, 
            train_data: pd.DataFrame, 
            model_param_dict: Dict[str, Any]
            ) -> None:
        """
        Trains the XGBoost model on the given training data.

        Parameters:
            train_data (pd.DataFrame): A DataFrame containing the training data. 
                The index must be a DatetimeIndex.
            model_param_dict (dict, optional): A dictionary of parameters to specify for the model.
                Defaults to self.default_params
        """
        logger.debug("Starting model fitting.")
        try:
            train_data = enforce_datetime_index(df=train_data, date_col=self.date_col)
            self.train = train_data.copy()
            
            # Handle parameters
            final_params = self._combine_and_categorize_params(model_param_dict)
            logger.debug(f"Model build parameters: {final_params['build_params']}")
            logger.debug(f"Model fit parameters: {final_params['fit_params']}")

            self.lags = final_params["custom_params"].get("lags",[])
            self.windows = final_params["custom_params"].get("windows",[])

            train_data = self._create_time_features(train_data)
            train_data = self._create_lag_features(train_data, self.lags)
            train_data = self._create_rolling_features(train_data, self.windows)
            train_data = self.postprocess(train_data)
            
            # Train the model
            self.model = XGBRegressor(
                **final_params["build_params"], 
                eval_metric="rmse"
                )
            
            np.random.seed(42)

            self.model.fit(
                train_data[self.feature_cols], 
                train_data[self.target_y],
                verbose=False
            )
            
            # Preprocess full data before predicting on it
            full_train = self.train.copy()
            full_train = self._create_time_features(full_train)
            full_train = self._create_lag_features(full_train, self.lags)
            full_train = self._create_rolling_features(full_train, self.windows)
            full_train = self.postprocess(full_train)

            self.fitted_vals = self.model.predict(full_train[self.feature_cols])
            logger.debug("Model fitting completed successfully.")

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise
    

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generates forecasts for the given test data using recursive prediction.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing the test data. 
                The index must be a DatetimeIndex. If `date_col` exists in the DataFrame, 
                it is used to set the index.

        Returns:
            np.ndarray: An array of forecasted values corresponding to the test data indices.
        """
        logger.debug("Starting prediction.")
        try:
            test_data = enforce_datetime_index(df=test_data, date_col=self.date_col)

            # Start recursive prediction
            X = self.train.copy()
            forecast = []

            for idx_date in test_data.index:
                # Predict the next step
                new_row = pd.DataFrame(
                    {self.target_y: np.nan}, index=[idx_date]
                )  
                X_test = pd.concat([X, new_row])

                # Create features for the current dataset
                X_test = self._create_time_features(X_test)
                X_test = self._create_lag_features(X_test, self.lags)
                X_test = self._create_rolling_features(X_test, self.windows)

                curr_features = X_test[self.feature_cols].iloc[-1:]
                pred = self.model.predict(curr_features)[0]

                forecast.append(pred)
                X.at[idx_date, self.target_y] = pred
            
            assert len(forecast) == len(test_data.index)
            forecast = np.array(forecast)
            clipped_forecast = np.clip(forecast, a_min=0, a_max=None)

            logger.debug("Prediction completed successfully.")
            if np.any(forecast < 0):
                logger.warning("Negative forecast values were clipped to 0.")

            return clipped_forecast
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise