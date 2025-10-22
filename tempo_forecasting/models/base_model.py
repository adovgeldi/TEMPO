from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import json
from typing import Dict, Optional, Any
import importlib.resources

from tempo_forecasting.utils.logging_utils import logger

class BaseModel(ABC):
    """
        Abstract base class for all time series models. This class defines the 
    interface and common attributes for all derived time series models, including the 
    mechanisms for fitting the model to training data, generating predictions, and 
    managing model parameters.

    Attributes:
        name (str): The name of the model.
        model (Any): Placeholder for the model instance to be implemented by subclasses.
        date_col (str): The name of the column containing date information. Defaults to 'ds'.
        target_y (str): The name of the target variable to be forecasted. Defaults to 'y'.
        model_params (dict): A dictionary for storing the best parameters for the model.
            model_params is not set automatically and must be set by calling _set_model_params
        default_model_params (dict): A dictionary which stores the default parameters for the model.

    Methods:
        fit(train_data: pd.DataFrame, model_param_dict: Dict[str, Any]) -> None:
            Abstract method to fit the model to the given training data. Must be implemented by subclasses.

        predict(test_data: pd.DataFrame) -> np.ndarray:
            Abstract method to generate predictions for the specified future dates. Must be implemented by subclasses.

        _set_model_params(param_dict: Optional[Dict[str, Any]] = None) -> None:
            Set model parameters by merging default parameters with user-specified parameters.

        _set_default_params_from_json(path: str = "config/default_model_params.json") -> None:
            Pulls default parameters for a specific model from a JSON file and assigns them as default model parameters.

        _combine_and_categorize_params(
            model_param_dict: Dict[str, Any],
            param_category_path: str = "config/param_categories.json",
            fill_in_defaults: bool = True,
            default_param_path: str = "config/default_model_params.json"
        ) -> None:
            Combines user-specified parameters with default parameters and categorizes them according to the provided 
            parameter category definitions.

        get_model_params() -> Dict[str, Any]:
            Retrieve the model parameters.

        get_name() -> str:
            Retrieve the name of the model.

        get_date_col() -> str:
            Retrieve the name of the date column used in the model.
    """

    def __init__(self, 
                 name: str,
                 date_col: str = 'ds',
                 target_y: str = 'y'
                 ):
        """
        Initializes the BaseModel with the specified name, date column, and target variable.

        Parameters:
            name (str): The name of the model.
            date_col (str): The name of the column containing date information. Default is 'ds'.
            target_y (str): The name of the target variable to be forecasted. Default is 'y'.
        """
        self._name = name
        self.model = None
        self.target_y = target_y
        self.date_col = date_col

        self.model_params = None # Dictionary set as best params
        self._set_default_params_from_json() # sets default model parameters from a json file

        
    @abstractmethod
    def fit(self, 
            train_data: pd.DataFrame, 
            model_param_dict: Dict[str,Any]
            ) -> None:
        """
        Abstract method to fit the model to the given training data. 
        Must be implemented by subclasses.

        Parameters:
            train_data (pd.DataFrame): A DataFrame containing the training data. 
                The format and required columns depend on the specific model implementation.
            model_param_dict (dict, optional): A dictionary of parameters to feed into the model.
                Unspecified parameters will retain their default settings.
                Parameters with an invalid name will be ignored.
        """
        pass
    

    @abstractmethod
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Abstract method to generate predictions for the specified future dates. 
        Must be implemented by subclasses.

        Parameters:
            test_data (pd.DataFrame): A DataFrame containing the future dates for which 
                predictions are to be made. The format depends on the specific model implementation.

        Returns:
            np.ndarray: An array of forecasted values corresponding to the future dates.
        """
        pass


    def _set_model_params(self, param_dict: Optional[Dict[str, Any]] = None) -> None:
        """
        Set model parameters by merging default parameters with user-specified parameters.
        
        Parameters:
            param_dict (dict): User-specified parameters to override defaults.
        """
        if not param_dict:
            # If no parameters are specified, use all default parameters
            self.model_params = self.default_model_params
            return

        # Merge default parameters with user-specified parameters
        # If a key exists in both, use the value from param_dict
        self.model_params = {**self.default_model_params, **param_dict}


    def _set_default_params_from_json(self,
                                path: str = "config/default_model_params.json") -> None:
        """
        Pulls default parameters for a specific model out of a json file, then assigns them to default_model_params.

        Parameters:
            path (str): path to the json storing the default model parameter values
        """
        with importlib.resources.files("tempo_forecasting").joinpath(path).open("r") as f:
            all_default_params = json.load(f)

        self.default_model_params = all_default_params[self.get_name()]

    def _combine_and_categorize_params(self,
                                       model_param_dict: Dict[str, Any] = {},
                                       param_category_path: str = "config/param_categories.json",
                                       fill_in_defaults: bool = True,
                                       default_param_path: str = "config/default_model_params.json"
                                       ) -> None:
        """
        Handles combination of default parameters (from json) and specified parameters (from dict) and
        breaks down the final combination of parameters into categories (as defined in a separate json),
        so that build, fit, and custom parameters can be handled more easily later in the modeling process.

        Parameters:
            model_param_dict (dict, optional): A dictionary of parameters to specify for the model.
            param_category_path (str): path to the json storing different known parameter categories
            fill_in_defaults (bool): whether or not to impose default values when available for parameters not in the model_param_dict
            default_param_path (str): path to the json storing the default model parameter values

        Returns:
            dict(dict(str)): dictionary of dictionaries (build_params, fit_params, and custom_params), with 
                each sub-dictionary containing parameter names and values for all parameters in that category
        """
        # Determine final set of parameters and values to pass to model
        if fill_in_defaults:
            with importlib.resources.files("tempo_forecasting").joinpath(default_param_path).open("r") as f:
                default_params = json.load(f)[self.get_name()]

            combined_params = {**default_params, **model_param_dict}
        else:
            if model_param_dict == {}:
                logger.warning("No parameters will be set in the model. No model_param_dict provided and fill_in_defaults = False")
            combined_params = model_param_dict

        # Reorganize parameter values into sub-dictionaries of build, fit, and custom parameters
        # in order to make handling the parameters easier later
        with importlib.resources.files("tempo_forecasting").joinpath(param_category_path).open("r") as f:
            param_categories = json.load(f)[self.get_name()]

        final_params = {"build_params": {},
                        "fit_params":{ },
                        "custom_params": {}}

        for param_name in combined_params.keys():
            if param_name in param_categories["build_params"]:
                final_params["build_params"][param_name] = combined_params[param_name]
            elif param_name in param_categories["fit_params"]:
                final_params["fit_params"][param_name] = combined_params[param_name]
            elif param_name in param_categories["custom_params"]:
                final_params["custom_params"][param_name] = combined_params[param_name]
            else:
                logger.warning(f"Parameter {param_name} not assigned a category in param_categories json file and will be treated as a fit param.")
                final_params["fit_params"][param_name] = combined_params[param_name]

        return final_params

    
    def get_model_params(self) -> Dict:
        """
        Retrieve the model parameters.

        Returns:
            Dict: A dictionary containing the parameters currently set for the model.
        """
        return self.model_params
    

    def get_name(self) -> str:
        """
        Retrieve the name of the model.

        Returns:
            str: The name of the model.
        """
        return self._name
    

    def get_date_col(self) -> str:
        """
        Retrieve the name of the date column used in the model.

        Returns:
            str: The name of the column containing date information.
        """
        return self.date_col
        
    
    
