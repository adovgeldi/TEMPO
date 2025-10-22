from typing import Optional, Sequence, Union
from tempo_forecasting.utils.logging_utils import logger

class SearchSpaceParam:
    """
    Class representing a parameter in the search space for hyperparameter optimization.

    This class encapsulates the properties of a single search space parameter, 
    including its name, data type, search type (bounded or categorical), bounds, 
    choices, step size, and log scaling options. It also provides methods to 
    retrieve these properties.

    Attributes:
        name (str): The name of the parameter.
        dtype (str): The data type of the parameter. Must be one of 'float', 'int', 
             'str', 'bool', or 'list'.
        search_type (str): The type of search for the parameter. Must be either 
             'bounded' for continuous or integer values, or 'categorical' for discrete choices.
        bounds (Optional[Sequence[Union[int, float]]]): A sequence containing the bounds for 
             the parameter if the search type is 'bounded'. Must be provided as a two-element 
             sequence (min, max).
        choices (Optional[Union[Sequence[Union[int, float, str]], Sequence[Sequence[Union[int, float, str]]]]]):
              A sequence of choices for the parameter if the search type is 'categorical'. 
              Must be provided for categorical search.
        step_size (Optional[Union[float, int]]): The step size for the parameter when 
               using bounded search. Defaults to 1 for integers or None for floats if not specified.
        logscale (bool): If True, the parameter will be sampled on a logarithmic scale when using bounded search; 
               defaults to False.

    Raises:
        AssertionError: If the provided `search_type` or `dtype` is invalid, or if 
                        the conditions for bounded or categorical parameters are not met.

    Methods:
        get_name() -> str:
            Returns the name of the parameter.
            
        set_name(value: str) -> None:
            Sets a new name for the parameter.
            
        get_dtype() -> str:
            Returns the data type of the parameter.
            
        set_dtype(value: str) -> None:
            Sets a new data type for the parameter. Must be 'float', 'int', or 'str'.
            
        get_search_type() -> str:
            Returns the type of search for the parameter.
            
        get_bounds() -> Optional[Sequence[Union[int, float]]]:
            Returns the bounds for the parameter if the search type is 'bounded'.
            
        get_choices() -> Optional[Union[Sequence[Union[int, float, str]], Sequence[Sequence[Union[int, float, str]]]]]:
            Returns the choices for the parameter if the search type is 'categorical'.
            
        get_step_size() -> Optional[Union[int, float]]:
            Returns the step size for the parameter if the search type is 'bounded'.
            
        get_logscale() -> bool:
            Returns whether the parameter is set to be sampled on a logarithmic scale if the search type is 'bounded'.
            
        to_optuna(trial) -> Union[int, float, str]:
            Returns a value for the parameter by sampling from the Optuna trial 
            based on the search type and corresponding properties.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        search_type: str,
        bounds: Optional[Sequence[Union[int, float]]] = None,
        choices: Optional[Union[Sequence[Union[int, float, str]], Sequence[Sequence[Union[int, float, str]]]]] = None,
        step_size: Optional[Union[float, int]] = None,
        logscale: bool = False
    ) -> None:
        """
        Initializes a SearchSpaceParam instance.

        Parameters:
            _name (str): The name of the parameter.
            _dtype (str): The data type of the parameter ("float", "int", "str", "bool").
            _search_type (str): The search type, either "bounded" for continuous ranges 
                or "categorical" for discrete values.
            _bounds (Optional[Sequence[Union[int, float]]], optional): The min and max bounds for bounded search.
                Required if search_type is "bounded".
            _step_size (Optional[Union[float, int]], optional): The step size for bounded search. Defaults to 1 for integers.
            _logscale (bool, optional): Whether to use a logarithmic scale for bounded search. Defaults to False.
            _choices (Optional[Union[Sequence[Union[int, float, str]], Sequence[Sequence[Union[int, float, str]]]]], optional): 
                The list of categorical options for categorical search. Required if search_type is "categorical".
        """
        logger.debug(f"Initializing SearchSpaceParam with name={name}, dtype={dtype}, search_type={search_type}, "
                     f"bounds={bounds}, choices={choices}, step_size={step_size}, logscale={logscale}")
        
        # Input validation
        assert search_type in ["bounded", "categorical"], "search_type must be 'bounded' or 'categorical'."
        logger.debug(f"Search type '{search_type}' is valid.")

        assert dtype in ["float", "int", "str", "bool", "list"], "dtype must be 'float', 'int', 'str', or 'bool'."
        logger.debug(f"Dtype '{dtype}' is valid.")

        self._name = name
        self._dtype = dtype
        self._search_type = search_type

        if self._search_type == "bounded":
            assert bounds is not None and len(bounds) == 2, "Bounds must be a tuple or list with two elements (min, max)."
            logger.debug(f"Bounds '{bounds}' are valid for bounded search.")

            assert dtype in ["float", "int"], "Bounded search requires dtype to be 'float' or 'int'."
            logger.debug(f"Dtype '{dtype}' is valid for bounded search.")

            self._bounds = bounds
            self._step_size = step_size or (1 if dtype == "int" else None)
            self._logscale = logscale or False
        elif self._search_type == "categorical":
            assert choices is not None, "Categorical search requires choices to be provided."
            logger.debug(f"Choices '{choices}' are valid for categorical search.")
            self._choices = choices
        else:
            logger.error(f"Invalid search_type: {search_type} provided for {name}")
            raise ValueError("Invalid search_type provided.")

    # Getters and Setters
    def get_name(self) -> str:
        """
        Retrieves the name of the parameter.

        Returns:
            str: The name of the parameter.
        """
        return self._name

    def set_name(self, value: str) -> None:
        """
        Sets a new name for the parameter.

        Parameters:
            value (str): The new name to assign to the parameter.
        """
        self._name = value

    def get_dtype(self) -> str:
        """
        Retrieves the data type of the parameter.

        Returns:
            str: The data type of the parameter ("float", "int", "str", "bool").
        """
        return self._dtype

    def set_dtype(self, value: str) -> None:
        """
        Sets a new data type for the parameter.

        Parameters:
            value (str): The new data type ("float", "int", "str", "bool").
        """
        assert value in ["float", "int", "str", "bool"], "dtype must be 'float', 'int', 'str' or 'bool'."
        logger.debug(f"Setting dtype to {value}.")
        self._dtype = value

    def get_search_type(self) -> str:
        """
        Retrieves the search type of the parameter.

        Returns:
            str: The search type ("bounded" or "categorical").
        """
        return self._search_type

    def get_bounds(self) -> Optional[Sequence[Union[int, float]]]:
        """
        Retrieves the bounds for bounded search parameters.

        Returns:
            Optional[Sequence[Union[int, float]]]: The min and max bounds if the search type is "bounded", 
                                                otherwise None.
        """
        return self._bounds if self._search_type == "bounded" else None

    def get_choices(self) -> Optional[Union[Sequence[Union[int, float, str]], Sequence[Sequence[Union[int, float, str]]]]]:
        """
        Retrieves the choices for categorical search parameters.

        Returns:
            Optional[Union[Sequence[Union[int, float, str]], Sequence[Sequence[Union[int, float, str]]]]]: 
                The list of choices if the search type is "categorical", otherwise None.
        """
        return self._choices if self._search_type == "categorical" else None

    def get_step_size(self) -> Optional[Union[int, float]]:
        """
        Retrieves the step size for bounded search parameters.

        Returns:
            Optional[Union[int, float]]: The step size if the search type is "bounded", otherwise None.
        """
        return self._step_size if self._search_type == "bounded" else None

    def get_logscale(self) -> bool:
        """
        Retrieves whether the bounded search uses logarithmic scaling.

        Returns:
            bool: True if logarithmic scaling is applied, False otherwise.
        """
        return self._logscale if self._search_type == "bounded" else None

    def to_optuna(self, trial) -> Union[int, float, str]:
        """
        Converts the parameter into an Optuna-compatible suggestion based on its configuration.

        Parameters:
            trial (optuna.Trial): The Optuna trial object to create the suggestion.

        Returns:
            Union[int, float, str]: The suggested value based on the search type and configuration.
        """
        if self._search_type == "bounded":
            if self._dtype == "float":
                return trial.suggest_float(self._name, self._bounds[0], self._bounds[1], step=self._step_size, log=self._logscale)
            elif self._dtype == "int":
                return trial.suggest_int(self._name, self._bounds[0], self._bounds[1], step=self._step_size, log=self._logscale)
        elif self._search_type == "categorical":
            return trial.suggest_categorical(self._name, self._choices)
        else:
            raise ValueError("Invalid search_type.")