from .base_model import BaseModel
import pandas as pd
import numpy as np
from typing import Dict, Any

from tempo_forecasting.utils.training_utils import enforce_datetime_index
from tempo_forecasting.utils.logging_utils import logger

class MovingAvgModel(BaseModel):
    """
    A simple moving average model for time series forecasting.

    This model forecasts future values by averaging the most recent `n_steps` 
    of observed or predicted values.

    Attributes:
        target_y (str): The name of the target variable to be forecasted.
        date_col (str): The name of the column containing date information.
        n_steps (int): The number of steps over which to calculate the moving average.
        y_vals (List[float]): A list containing the last `n_steps` values of the 
            training data's target variable, extended with predicted values during forecasting.
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
        Initializes the MovingAvgModel with the specified target and date columns.

        Parameters:
            target_y (str): The name of the target variable to be forecasted.
            date_col (str): The name of the column containing date information.
        """
        super().__init__("moving_avg",
                         date_col=date_col, 
                         target_y=target_y
                         )
        
        self.n_steps = None 
        self.y_vals = None
        self.fitted_vals = None
        logger.debug(f"Initialized MovingAvgModel with target_y: '{target_y}' and date_col: '{date_col}'")


    def fit(self, 
            train_data: pd.DataFrame, 
            model_param_dict: Dict[str,Any]
            ) -> None:
        """
        Prepares the moving average model by initializing the historical data.

        Parameters:
            train_data (pd.DataFrame): A DataFrame containing the training data.
                Must include the target variable (`target_y`).
            model_param_dict (dict, optional): A dictionary of parameters to specify for the model.
                Defaults to self.default_params
        """
        logger.debug("Starting model fitting.")
        # using "fit" to mean more "initialize" in this case
        try:
            train = train_data.copy()
            train = enforce_datetime_index(df=train, date_col=self.date_col)

            final_params = self._combine_and_categorize_params(model_param_dict)
            logger.debug(f"Model build parameters: {final_params['build_params']}")
            logger.debug(f"Model fit parameters: {final_params['fit_params']}")
            self.n_steps = final_params["custom_params"]["n_steps"]

            self.y_vals = train[self.target_y].iloc[-1*self.n_steps:].tolist()
            self.fitted_vals = np.clip(train[self.target_y].rolling(window=self.n_steps,min_periods=1).mean().reset_index(drop=True),
                                        a_min=0, 
                                        a_max=None)
            logger.debug("Model fitting completed successfully.")

        except Exception as e:
            logger.error(f"Error during model fitting: {str(e)}")
            raise


    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Generates forecasts for the given test data using the moving average.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing the test data.
                The index must be a `DatetimeIndex`.

        Returns:
            np.ndarray: A 1-dimensional array of forecasted values corresponding to 
                the indices of the test data.
        """
        logger.debug("Starting prediction.")
        try:
            test = test_data.copy()
            test = enforce_datetime_index(df=test, date_col=self.date_col)
            
            y_vals_w_preds = self.y_vals
            for _ in range(len(test)):
                lookback_period = np.min([len(y_vals_w_preds),self.n_steps])
                avg = np.mean(y_vals_w_preds[-1*lookback_period:])
                y_vals_w_preds.extend([avg])

            df = pd.DataFrame({'ds': test.index})
            df['yhat'] = y_vals_w_preds[-1*len(test):]
            forecast = np.clip(df['yhat'].values, a_min=0, a_max=None)

            logger.debug("Prediction completed successfully.")
            if np.any(df['yhat'] < 0):
                logger.warning("Negative forecast values were clipped to 0.")

            return forecast

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise