from .base_model import BaseModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
from typing import Dict, Any

from tempo_forecasting.utils.training_utils import enforce_datetime_index
from tempo_forecasting.utils.logging_utils import logger


class ExpSmoothingModel(BaseModel):
    """
    A time series forecasting model based on the Exponential Smoothing technique
    from the statsmodels library. This implementation supports additive trend
    and seasonal components for seasonal time series data.

    Attributes:
        target_y (str): The name of the target variable to be forecasted.
        date_col (str): The name of the column containing date information.
        model (ExponentialSmoothing): The fitted Exponential Smoothing model.
        fitted_vals (np.ndarray): Array of fitted values for the training time period 
        default_model_params (dict): A dictionary which stores the default parameters for the model.

    
    Methods:
        fit(train_data: pd.DataFrame, model_param_dict: Dict[str, Any]) -> None:
            Method to fit the model to the given training data.

        predict(test_data: pd.DataFrame) -> np.ndarray:
            Method to generate predictions for the specified future dates.

    """
    def __init__(self, 
                 target_y: str,
                 date_col: str
                 ):
        """
        Initializes the ExpSmoothingModel with the specified target and date column.

        Parameters:
            target_y (str): The name of the target variable to be forecasted.
            date_col (str): The name of the column containing date information.
        """
        super().__init__("expsmooth",
                         date_col=date_col,
                         target_y=target_y
                         )

        self.model = None
        self.fitted_vals = None
        logger.debug(f"Initialized ExpSmoothingModel with target_y: '{target_y}' and date_col: '{date_col}'")


    def fit(self, 
            train_data: pd.DataFrame, 
            model_param_dict: Dict[str,Any] = {}
            ) -> None:
        """
        Fits the Exponential Smoothing model to the training data.

        Parameters:
            train_data (pd.DataFrame): A DataFrame containing the training data. 
                Must have columns `date_col` and `target_y`.
            model_param_dict (dict, optional): A dictionary of parameters to specify for the model.
                Defaults to self.default_params
        """
        logger.debug("Starting model fitting.")
        try:
            train = train_data.copy()
            train = enforce_datetime_index(df=train, date_col=self.date_col)
            train_series = train[self.target_y].astype(float)

            final_params = self._combine_and_categorize_params(model_param_dict)
            logger.debug(f"Model build parameters: {final_params['build_params']}")
            logger.debug(f"Model fit parameters: {final_params['fit_params']}")
            
            # Initialize the model
            self.model = ExponentialSmoothing(endog=train_series, **final_params["build_params"])

            # Fit the model with the extracted parameters
            self.model = self.model.fit(**final_params["fit_params"])
            self.fitted_vals = np.array(self.model.fittedvalues)
            logger.debug("Model fitting completed successfully.")
        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise

    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generates forecasts for the specified future dates.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing the future dates for which 
                predictions are to be made. The index or `date_col` must be convertible to a DatetimeIndex.

        Returns:
            np.ndarray: An array of forecasted values for the specified future dates. 
                Values are clipped to a minimum of 0 to avoid negative forecasts.
        """
        logger.debug("Starting prediction.")
        try:
            test = test_data.copy()
            test = enforce_datetime_index(df=test, date_col=self.date_col)
            steps = len(test)

            forecast = self.model.forecast(steps)
            clipped_forecast = np.clip(forecast.values, a_min=0, a_max=None)

            logger.debug("Prediction completed successfully.")
            if np.any(forecast.values < 0):
                logger.warning("Negative forecast values were clipped to 0.")

            return clipped_forecast

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise