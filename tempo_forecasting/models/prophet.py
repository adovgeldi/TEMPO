from .base_model import BaseModel
from prophet import Prophet

import pandas as pd
import numpy as np
from typing import Dict, Any

from tempo_forecasting.utils.training_utils import enforce_datetime_index
from tempo_forecasting.utils.logging_utils import logger

import logging
# Set logging so that prophet doesn't print out two lines every time it trains
# Ideally we could also set cmdstanpy to warn, but that doesn't seem to work in prophet
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").disabled=True

class ProphetModel(BaseModel):
    """
    A time series forecasting model based on Facebook's Prophet library. 
    This class extends the BaseModel and provides methods to train and predict 
    time series data while incorporating changepoint detection, seasonality, 
    and holiday effects.

    Attributes:
        target_y (str): The name of the target variable to be forecasted.
        date_col (str): The name of the column containing date information.
        model (Prophet): An instance of the Prophet forecasting model.
        default_model_params (dict): A dictionary which stores the default parameters for the model.

        
    Methods:
        _detect_changepoints(df: pd.Dataframe) -> list:
            Method to detect changepoints in time series data.

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
        Initializes the ProphetModel with the specified target and date column.

        Parameters:
            target_y (str): The name of the target variable to be forecasted.
            date_col (str): The name of the column containing date information.
        """
        super().__init__("prophet",
                         date_col=date_col, 
                         target_y=target_y
                         )

        self.model = None
        self.fitted_vals = None
        logger.debug(f"Initialized ProphetModel with target_y: '{target_y}' and date_col: '{date_col}'")


    def _detect_changepoints(self, df):
        """
        Detects changepoints in the time series data based on significant residual deviations.

        Parameters:
            df (pd.DataFrame): The Prophet-compatible DataFrame with columns 'ds' (dates) and 'y' (values).

        Returns:
            list: A list of changepoint dates (`ds`) where significant changes occur.
        """
        try:
            logger.debug(f"Starting changepoint detection.")
            df = df.copy() 

            df['smoothed_y'] = df['y'].rolling(window=7, center=True).mean() # Smooth over 7-day rolling window
            df['res'] = df['y'] - df['smoothed_y']
            thresh = 2 * df['res'].std()
            df['is_changepoint'] = abs(df['res']) > thresh # flag residual deviations > 2 stds as a "swing"

            changepoints = df.loc[df['is_changepoint'], 'ds'].tolist()
            logger.debug(f"Detected {len(changepoints)} changepoints.")

            return changepoints
        
        except Exception as e:
            logger.error(f"Error during changepoint detection: {str(e)}")
            raise
    

    def fit(self, 
            train_data: pd.DataFrame, 
            model_param_dict: Dict[str, Any] = {}
            ) -> None:
        """
        Trains the Prophet model on the given training data.

        Parameters:
            train_data (pd.DataFrame): A DataFrame containing the training data. Must have `date_col` and `target_y` columns.
            model_param_dict (dict, optional): A dictionary of parameters to specify for the model. Defaults to self.default_params.
        """
        logger.debug("Starting model fitting.")
        try:
            final_params = self._combine_and_categorize_params(model_param_dict)
            logger.debug(f"Model build parameters: {final_params['build_params']}")
            logger.debug(f"Model fit parameters: {final_params['fit_params']}")
            prophet_df = train_data.reset_index().rename(columns={self.date_col: "ds", self.target_y: "y"})

            am_changepoints = final_params["custom_params"].get("automanual_changepoints",None)
            if am_changepoints:
                final_params["build_params"]["changepoints"] = self._detect_changepoints(prophet_df)
                # note: if "changepoints" is supplied, "n_changepoints" is not used.
                # https://github.com/facebook/prophet/blob/main/python/prophet/forecaster.py
                
                # potential problem with this method:
                #   If optuna has a range of n_changepoints values and automanual_changepoints [True, False] to try,
                #   something like (am_cp = True, n_cp = 25) and (am_cp = True, n_cp = 10) 
                #   will result in the same self.model_params but may look different to optuna.
                #   This might not interact with self._pull_repeat_metrics() as desired.
                #   .fit() should work fine, but it would be less efficient. Needs testing.

            self.model = Prophet(**final_params["build_params"])

            # TO DO: consider custom params for 
            # self.model = self.model.add_seasonality(name='weekly', period=7, fourier_order=3)
            # self.model = self.model.add_country_holidays(country_name='US')

            np.random.seed(42)
            self.model.fit(prophet_df)
            self.fitted_vals = np.clip(self.model.predict(prophet_df)["yhat"].values, a_min=0, a_max=None)
            logger.debug("Model fitting completed successfully.")

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise
    

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generates forecasts for the specified future dates.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing the future dates. The index must be a DatetimeIndex or convertible to one.

        Returns:
            np.ndarray: An array of predicted values (`yhat`) for the future dates. Values are clipped to a minimum of 0.
        """
        logger.debug("Starting prediction.")
        try:
            test_data = enforce_datetime_index(df=test_data, date_col=self.date_col)
            
            df = pd.DataFrame({'ds': test_data.index})
            forecast = self.model.predict(df)['yhat'].values
            clipped_forecast = np.clip(forecast, a_min=0, a_max=None)

            logger.debug("Prediction completed successfully.")
            if np.any(forecast < 0):
                logger.warning("Negative forecast values were clipped to 0.")

            return clipped_forecast

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise