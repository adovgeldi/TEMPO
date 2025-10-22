import optuna
import numpy as np
import pandas as pd
from typing import Sequence, Optional, Dict, Union, Any
import time
from dataclasses import dataclass
from scipy.stats import linregress
from pymannkendall import original_test

from tempo_forecasting.utils.training_utils import enforce_datetime_index, calculate_metric, cross_validate
from tempo_forecasting.utils.config_utils import get_default_search_params, get_test_search_params
from tempo_forecasting.optuna_opt.optuna_param import SearchSpaceParam

from optuna.samplers import TPESampler, GridSampler
from optuna.trial import TrialState

# Set logging level so we don't get an output for every. single. trial.
optuna.logging.set_verbosity(optuna.logging.WARNING) 
import traceback

@dataclass
class OptunaConfig:
    """
    Configuration class for setting up and controlling the parameters for Optuna hyperparameter optimization.

    Attributes:
        n_trials (int): The total number of trials to run for hyperparameter optimization. 
            Default is 25.
        timeout_sec (int): The maximum amount of time (in seconds) for the optimization 
            study to run. Default is 300 seconds.
        n_startup_trials (int): The number of initial trials to perform before optimizing 
            hyperparameters. Default is 10.
        performance_plateau_threshold (float): The threshold value that determines when 
            the model's performance is considered to have plateaued. Default is 1e-5.
        plateau_trials_window (int): The number of trials to check for performance plateau. 
            Default is 5.
        eval_metric (str): The metric used to evaluate the performance of the hyperparameter 
            configurations. Default is "wmape" (Weighted Mean Absolute Percentage Error).
    """
    n_trials: int = 25
    timeout_sec: int = 300
    n_startup_trials: int = 10
    performance_plateau_threshold: float = 1e-5
    plateau_trials_window: int = 5
    eval_metric: str = "wmape"

class OptunaCallbacks:
    """
    Class containing static methods to implement callbacks for Optuna studies.

    These callbacks help manage the optimization process by providing mechanisms 
    to stop the study based on time limits or to detect performance plateaus.

    Methods:
        time_limit_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            Monitors the elapsed time of the study. If the time exceeds the specified
            limit, it stops the study and logs a warning message.

        performance_plateau_callback(study: optuna.Study, trial: optuna.Trial) -> None:
            Checks for a performance plateau in the recent trials. If the values of 
            the last specified number of trials indicate that there has been no 
            significant improvement, the study is stopped, and an info message is logged.

    Parameters for performance_plateau_callback:


    Parameters for time_limit_callback:
        study (optuna.Study): An instance of the Optuna study being evaluated.
        trial (optuna.Trial): The current trial being executed.
    """
    
    @staticmethod
    def time_limit_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback to stop study after timeout.

        Parameters:
            study (optuna.Study): An instance of the Optuna study being evaluated.
            trial (optuna.Trial): The current trial being executed.
        """
        elapsed_time = time.time() - study.user_attrs["start_time"]
        if elapsed_time > study.user_attrs["timeout"]:
            # TO DO: fix logging here
            # self.log_warning("Time limit reached. Stopping the study.")
            study.stop()

    @staticmethod
    def performance_plateau_callback(study: optuna.Study, trial: optuna.Trial) -> None:
        """
        Callback to stop the study if no improvement after N trials.

        Parameters:
            study (optuna.Study): An instance of the Optuna study being evaluated.
            trial (optuna.Trial): The current trial being executed.
        """
        config = study.user_attrs["config"]
        if len(study.trials) < config.plateau_trials_window:
            return
        
        # Retrieve the values of the most recent trials
        recent_trials = study.trials[-config.plateau_trials_window:]
        recent_values = [
            t.value for t in recent_trials if t.value is not None
        ]

        # If there are fewer valid trials than the window size, skip
        if len(recent_values) < config.plateau_trials_window:
            return

        # Check if consecutive values are within the plateau threshold
        consecutive_differences = [
            abs(recent_values[i] - recent_values[i - 1])
            for i in range(1, len(recent_values))
        ]
        if all(diff < config.performance_plateau_threshold for diff in consecutive_differences):
            # Logger not instantiated by this point.
            # TO DO: update logging?
            # self.log_info(
            #     f"Performance plateau detected. No significant improvement "
            #     f"in the last {config.plateau_trials_window} trials."
            # )
            study.stop()

class OptunaObjective():
    """
    An Optuna-compatible objective class for hyperparameter optimization.

    This class defines the objective function for Optuna studies. It supports 
    generating an Optuna search space, evaluating hyperparameter configurations, 
    and reusing results from previously completed trials to avoid redundant computations.

    Attributes:
        model: The model to be optimized, which must implement `fit`, `predict`, 
               and provide methods like `get_name` and `get_date_col`.
        config (OptunaConfig): The optuna config settings
        train (pd.DataFrame): The training dataset with features and the target column.
        val (pd.DataFrame): The validation dataset used to evaluate the model.
        default_search_space (Sequence[SearchSpaceParam]): The default optuna parameter search space
        search_space (Sequence[SearchSpaceParam]): The final optuna parameter search space
        evaluated_trials (dict): A dictionary storing details of evaluated trials

    Methods:
        _validate_search_space: Validates the search space to ensure all parameters are of type SearchSpaceParam.
        _build_optuna_search_space: Constructs the Optuna search space for a trial.
        _pull_repeat_metrics: Checks for previously completed trials with the same parameters 
                              to reuse their evaluation metrics.
        __call__: Executes the objective function for a given trial.
    """
    def __init__(self,
                 model,
                 data: pd.DataFrame,
                 cv_dates: pd.DataFrame,
                 param_search_space: Optional[Sequence[SearchSpaceParam]], 
                 config: OptunaConfig,
                 is_test: bool,
                 logger = None,
                 category = None
                 ):
        """
        Initializes the OptunaObjective class.

        Parameters:
            model: The model to be optimized.
            train_data (pd.DataFrame): The training dataset.
            val_data (pd.DataFrame): The validation dataset.
            param_search_space (Optional[Sequence[SearchSpaceParam]]): 
                The hyperparameter search space. If not provided, default parameters are used.
            config: The optuna config settings
            is_test (bool): Whether the optimization is in test mode.
        """
        self.model = model
        self.config = config

        date_col = model.get_date_col()
        self.data = enforce_datetime_index(df=data.copy(), date_col=date_col)

        self.cv_dates = cv_dates

        model_name = model.get_name()
        self.default_search_space = get_test_search_params(model_name) if is_test else get_default_search_params(model_name)
        self.search_space = param_search_space or self.default_search_space
        self._validate_search_space()

        # Cache for storing evaluated trials
        self.evaluated_trials = {}

        self.category = category or ""
        self.logger = logger

        if logger is None:
            from tempo_forecasting.utils.logging_utils import logger as default_logger
            self.logger = default_logger
            self.log_info = lambda msg: default_logger.info(msg)
            self.log_warning = lambda msg: default_logger.warning(msg)
            self.log_error = lambda msg: default_logger.error(msg)
            self.log_debug = lambda msg: default_logger.debug(msg)
        elif logger == "print":
            self.log_info = lambda msg: print(msg)
            self.log_warning = lambda msg: print(msg)
            self.log_error = lambda msg: print(msg)
            self.log_debug = lambda msg: print(msg)
        else:
            # Use the WorkerLogger's category-aware logging
            self.log_info = lambda msg, details="Optuna Objective": logger.info(message=msg, category=self.category, details=details)
            self.log_warning = lambda msg, details="Optuna Objective": logger.warning(message=msg, category=self.category, details=details)
            self.log_error = lambda msg, details="Optuna Objective": logger.error(message=msg, category=self.category, details=details)
            self.log_debug = lambda msg, details="Optuna Objective": logger.debug(message=msg, category=self.category, details=details)

        self.log_info(f"Initialized OptunaObjective for model: {model_name}")


    def get_model_specific_params(self) -> Dict:
        """
        Get model-specific optimization parameters.
        
        Returns:
            dict(str): parameter settings for the specific type of model in the optuna study
        """
        model_name = self.model.get_name().lower()
        
        params = {
            'prophet': {
                'mcmc_samples': 0,  # Disable MCMC
                'uncertainty_samples': 100,
            },
            'exponential_smoothing': {
                'optimized': True,
                'use_boxcox': False,
                'remove_bias': False,
                'method': 'L-BFGS-B',
                'maxiter': 1000
            },
            'xgboost': {
                'tree_method': 'hist',
                'grow_policy': 'lossguide',
                'max_leaves': 32,
                'max_bin': 256,
            },
            'lightgbm': {
                'force_col_wise': True,
                'feature_fraction_bynode': 0.8,
                'max_bin': 255,
                'min_data_in_leaf': 20,
            }
        }
        return params.get(model_name, {})
    

    def _detect_trend(self, series: pd.Series) -> tuple:
        """
        Determines if a trend exists and calculates its strength relative to data scale.

        Parameters:
            series (pd.Series): Time series data.

        Returns:
            tuple (bool, float): (Has trend?, Relative Trend Strength)
        """
        # Ensure data is clean
        series = series.dropna()
        if len(series) < 10:
            return (False, 0)  # Not enough data

        # Check if a trend exists using the Mann-Kendall Test
        mk_result = original_test(series)
        has_trend = mk_result.trend != "no trend"

        # Compute Trend Strength using Linear Regression
        x = np.arange(len(series))
        slope, _, _, _, _ = linregress(x, series.values)

        # Convert Trend Strength to a Relative Scale
        relative_trend_strength = abs(slope) / series.mean() if series.mean() != 0 else 0

        return has_trend, relative_trend_strength
    

    def _validate_search_space(self):
        """
        Validates the search space to ensure all parameters are of type SearchSpaceParam.

        Raises:
            ValueError: If any item in the search space is not a SearchSpaceParam instance.
        """
        if not all(isinstance(p, SearchSpaceParam) for p in self.search_space):
            raise ValueError("All items in param_search_space must be of type SearchSpaceParam.")


    def _build_optuna_search_space(self, trial: optuna.Trial) -> Dict:
        """
        Constructs the Optuna search space with model-specific optimizations

        Returns:
            dict: a dict of search space parameters
        """
        optuna_search_space = {}
        
        # Add base parameters from search space
        for p in self.search_space:
            param_name = p.get_name()
            try:
                optuna_search_space[param_name] = p.to_optuna(trial)
            except Exception as e:
                raise ValueError(f"Error constructing search space for {param_name}: {e}")
        
        # if self.model.get_name().lower() == "expsmooth":
        #     has_trend, rel_trend_strength = self._detect_trend(self.train[self.model.target_y])
            
        #     # Threshold for considering a trend (2% of mean change)
        #     thresh_pct = 0.01

        #     if has_trend:
        #         optuna_search_space["trend"] = trial.suggest_categorical("trend", ["additive"])
                
        #         # Use trend strength to decide if `damped_trend` should be True/False
        #         if rel_trend_strength > thresh_pct:  
        #             optuna_search_space["damped_trend"] = trial.suggest_categorical("damped_trend", [True, False])
        #         else:
        #             optuna_search_space["damped_trend"] = False  # Force disable damped trend for weak trends

        # Add model-specific optimization parameters
        optuna_search_space.update(self.get_model_specific_params())
        
        return optuna_search_space
    

    def _get_trial_hash(self, params: Dict) -> int:
        """
        Generate a unique hash for trial parameters

        Parameters:
            params (dict): the model parameters names and values to hash

        Returns:
            hash(frozenset(dict)): a hash of the trial parameters
        """
        return hash(frozenset(params.items()))


    def _pull_repeat_metrics(self, trial: optuna.Trial) -> Optional[float]:
        """
        Checks if the current trial's parameters have been evaluated in previous trials.

        Parameters:
            trial (optuna.Trial): The current Optuna trial.

        Returns:
            Optional[float]: The metric value from a previous trial with the same parameters, 
                             or None if no matching trial is found.
        """
        # TO DO: could add sort(set()) to stringification for added thoroughness
        completed_trials = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
        #hashable_trial_params = {k:(v if type(v) != list else str(v)) for k,v in trial.params.items()}        
        hashable_trial_params = {k: (",".join(map(str, sorted(v))) if isinstance(v, list) else str(v)) for k, v in sorted(trial.params.items())}
        trial_hash = self._get_trial_hash(hashable_trial_params)
        
        # Check in-memory cache first
        if trial_hash in self.evaluated_trials:
            return self.evaluated_trials[trial_hash]

        for prev_trial in reversed(completed_trials):
            #hashable_prev_trial_params = {k:(v if type(v) != list else str(v)) for k,v in prev_trial.params.items()}
            hashable_prev_trial_params = {k: (",".join(map(str, sorted(v))) if isinstance(v, list) else str(v)) for k, v in sorted(prev_trial.params.items())}
            if self._get_trial_hash(hashable_prev_trial_params) == trial_hash:
                # Update cache and return value
                if prev_trial.value is not None:
                    self.log_debug(f"Found repeated trial with hash {trial_hash}. Value: {prev_trial.value}")
                    self.evaluated_trials[trial_hash] = prev_trial.value
                    return prev_trial.value
            
        # No match found
        self.log_debug(f"No matching trial found for hash {trial_hash}.")
        return None
    

    def suggest_grid_search(self, max_combinations: int = 20) -> Union[bool, Dict]:
        """
        Determines whether to use a grid search for hyperparameter optimization and constructs the grid search space.

        This function checks if the parameter search space consists of entirely categorical parameters and/or 
        bounded parameters with specified step sizes. If the total number of parameter combinations is less than 
        the `max_combinations` threshold, it recommends using a grid search and constructs the search space.

        Parameters:
            max_combinations (int): The maximum number of combinations allowed for a grid search. 
                                    If the number of combinations exceeds this value, a TPE sampler is preferred.

        Returns:
            tuple:
                - suggest_grid_search (bool): True if a grid search is recommended, False otherwise.
                - grid_search_space (dict): A dictionary where keys are parameter names and values are 
                                            lists of possible values for the parameters in the grid.
        """
        # Separate the parameters into categorical and bounded groups
        categorical_params = [p for p in self.search_space if p.get_search_type() == "categorical"]
        bounded_params = [p for p in self.search_space if p.get_search_type() == "bounded"]

        # Check if any bounded parameter lacks a step size
        contains_stepless_bounded = any(p.get_step_size() is None for p in bounded_params)

        suggest_grid_search = False
        grid_search_space = {}

        if not contains_stepless_bounded:
            # Process categorical parameters
            for p in categorical_params:
                choices = p.get_choices()
                param_name = p.get_name()
                grid_search_space[param_name] = choices

            # Process bounded parameters
            for p in bounded_params:
                if p.get_dtype() == "int":
                    bounds = p.get_bounds()
                    step_size = p.get_step_size()
                    param_range = list(range(bounds[0], bounds[1], step_size))
                elif p.get_dtype() == "float":
                    bounds = p.get_bounds()
                    step_size = p.get_step_size()
                    param_range = list(np.round(np.arange(bounds[0], bounds[1], step_size), 5))
                else:
                    raise ValueError(f"Unsupported dtype: {p.get_dtype()}")

                # Ensure the max value is included if divisible by step size
                if (bounds[1] - bounds[0]) % step_size == 0:
                    param_range.append(bounds[1])

                param_name = p.get_name()
                grid_search_space[param_name] = param_range

            # Calculate the total size of the grid space
            grid_space_size = np.prod([len(values) for values in grid_search_space.values()])

            # Suggest grid search if the total combinations are within the limit
            if grid_space_size <= max_combinations:
                suggest_grid_search = True

        return suggest_grid_search, grid_search_space
 
    def _get_pruning_callback(self, trial):
        """
        Get model-specific pruning callback for tree-based methods.

        Parameters:
            trial (optuna.Trial): The current Optuna trial.
        """
        if self.model.get_name() == "xgboost":
            return optuna.integration.XGBoostPruningCallback(trial, observation_key="validation_0-rmse")
        elif self.model.get_name() == "lightgbm":
            return optuna.integration.LightGBMPruningCallback(trial, metric="l2", valid_name="train")
        return None
    

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Executes the objective function for a given trial.

        Parameters:
            trial (optuna.Trial): The current Optuna trial.

        Returns:
            float: The evaluation metric value for the trial.
        """
        self.log_debug(f"Starting trial {trial.number} for model {self.model.get_name()}...")
        start_time = time.time()
        try:
            optuna_search_space = self._build_optuna_search_space(trial)
            self.log_debug(f"Trial parameters: {optuna_search_space}")

            # Check cache for repeat trials
            trial.set_user_attr("repeat_trial", False)
            repeat_metric = self._pull_repeat_metrics(trial)
            if repeat_metric is not None:
                self.log_info(f"Using cached metric for trial: {repeat_metric}")
                trial.set_user_attr("repeat_trial", True)
                return repeat_metric
            
            # # Add pruning callback for supported models
            # if self.model.get_name() in ["xgboost", "lightgbm"]:
            #     pruning_callback = self._get_pruning_callback(trial)
            #     if pruning_callback:
            #         optuna_search_space["callbacks"] = [pruning_callback]

            desired_metrics = list(set(["wmape","mae","rmse"] + [self.config.eval_metric]))

            cv_result = cross_validate(data = self.data, 
                                        date_col = self.model.date_col, 
                                        target_col = self.model.target_y, 
                                        model_class = type(self.model), 
                                        model_param_dict = optuna_search_space, 
                                        cv_dates = self.cv_dates, 
                                        metrics = desired_metrics)
            
            trial.set_user_attr("full_cv_results", cv_result)

            # saving this is slightly redundant with full_cv_results
            # but this format is much easier to work with
            all_metrics = {mt.upper():[cv_round["metrics"][mt.upper()] for cv_round in cv_result] for mt in desired_metrics}
            self.log_info(f"trial cv metrics: {all_metrics}")
            trial.set_user_attr("cv_metrics", all_metrics)

            all_cv_rounds_metrics = all_metrics[self.config.eval_metric.upper()]
            aggregated_eval_metric = np.mean(all_cv_rounds_metrics) + np.std(all_cv_rounds_metrics)
            metric_value = aggregated_eval_metric

            # Prune if taking too long
            elapsed_time = time.time() - start_time
            self.log_info(f"Trial {trial.number} completed. Aggregated (mean + std) CV Metric ({self.config.eval_metric}): {metric_value}. Elapsed Time: ({elapsed_time:.2f}s).")
            
            # This doesn't do anything because it is only run AFTER the training is complete?
            # # Prune if taking too long
            # if elapsed_time > (self.config.timeout_sec / self.config.n_trials):
            #     self.log_warning(f"Trial taking too long ({elapsed_time:.2f}s). Pruning.")
            #     raise optuna.TrialPruned()
                
            if metric_value <= 0 or np.isclose(metric_value, 0):
                self.log_warning(f"Invalid metric value: {metric_value}. Assigning penalty.")
                return 1e10
            
            # Cache the result
            self.evaluated_trials[self._get_trial_hash(trial.params)] = metric_value
            return metric_value
        
        except optuna.exceptions.TrialPruned:
            # Handle pruning specifically
            # TO DO: include the below piece but make it not error when we do not yet have a best value
            # Current best value: {trial.study.best_value}
            self.log_warning(f"Trial {trial.number} was pruned.")
            raise
        except Exception as e:
            # Handle all other exceptions
            self.log_error(f"Trial {trial.number} failed with error: {str(e)}")
            raise


def run_tuned_fit(model, 
                  category: str,
                  data: pd.DataFrame, 
                  cv_dates,
                  param_search_space: Optional[Sequence[SearchSpaceParam]] = None, # TYPE: SEQUENCE OF SearchSpaceParam TYPES
                  config: Optional[OptunaConfig] = None,
                  is_test: bool = False,
                  study_name: Optional[str] = None,
                  logger = None,
                  tuning_only: bool = False,
                  ) -> Optional[Any]:
    """
    Runs Optuna-based hyperparameter optimization for a given model.

    Parameters:
        model: The model to be optimized.
        train_data (pd.DataFrame): The training dataset.
        val_data (pd.DataFrame): The validation dataset.
        param_search_space (Optional[Sequence[SearchSpaceParam]]): 
            The hyperparameter search space. If None, default search parameters are used.
        config (dict): A dictionary of optuna configuration settings.
        is_test (bool): Whether the optimization is running in test mode.
        study_name (Optional[str]): Name of the Optuna study.

    Returns:
        None: The function trains the model with the best parameters found and prints the results.
    """
    if logger is None: # Use standard logging
        from tempo_forecasting.utils.logging_utils import logger as default_logger
        logger = default_logger
        log_func_info = lambda msg: logger.info(msg)
        log_func_warn = lambda msg: logger.warning(msg)
        log_func_debug = lambda msg: logger.debug(msg)
        log_func_error = lambda msg: logger.error(msg)
    elif logger == "print":
        log_func_info = lambda msg: print(msg)
        log_func_warn = lambda msg: print(msg)
        log_func_debug = lambda msg: print(msg)
        log_func_error = lambda msg: print(msg)
    else:
        # Use the WorkerLogger's category-aware logging if passed in
        log_func_info = lambda msg, details="Run Tuned Fit": logger.info(message=msg, category=category, details=details)
        log_func_warn = lambda msg, details="Run Tuned Fit": logger.warning(message=msg, category=category, details=details)
        log_func_debug = lambda msg, details="Run Tuned Fit": logger.debug(message=msg, category=category, details=details)
        log_func_error = lambda msg, details="Run Tuned Fit": logger.error(message=msg, category=category, details=details)

    # Define callback for detailed logging
    def detailed_logging_callback(study, trial):
        """Callback to log detailed trial information"""
        if trial.state == TrialState.COMPLETE:
            log_func_info(f"Trial {trial.number} completed with value: {trial.value:.4f}; Parameters: {json.dumps(trial.params, default=str)}")
        elif trial.state == TrialState.PRUNED:
            log_func_info(f"Trial {trial.number} pruned; Current best value: {study.best_value if study.best_value is not None else 'None'}")
        elif trial.state == TrialState.FAIL:
            log_func_error(f"Trial {trial.number} failed; Error: {getattr(trial, 'error', 'Unknown error')}")
    
    # Set study attributes for callbacks
    def set_study_attributes(study, attributes):
        for key, value in attributes.items():
            study.set_user_attr(key, value)

    log_func_info(f"Starting hyperparameter optimization for {model.get_name()}")
    
    config = config or OptunaConfig()

    data = enforce_datetime_index(df=data.copy(), date_col=model.date_col)
    objective = OptunaObjective(
        model=model,
        data=data, 
        cv_dates=cv_dates, 
        param_search_space=param_search_space,
        config=config,
        is_test=is_test,
        logger=logger,
        category=category
    )
    
    grid_suggested, grid_search_space = objective.suggest_grid_search(max_combinations=config.n_trials)
    if grid_suggested:
        sampler = GridSampler(grid_search_space)
        # log_func_info(f"grid suggested for {model.get_name()} with search space: {grid_search_space}")
    else:
        sampler = TPESampler(seed=42, 
                                n_startup_trials=config.n_startup_trials,
                                multivariate=True,
                                n_ei_candidates=24
                                )

    # Create and configure study
    log_func_info(f"Creating Optuna study: {study_name}")
    study = optuna.create_study(
        direction="minimize", 
        study_name=study_name,
        sampler=sampler
        )

    set_study_attributes(study, {
        "start_time": time.time(),
        "timeout": config.timeout_sec, # user attr timeout for time_limit_callback 
        "config": config
    })
    
    # Attempt to run study
    try:        
        log_func_info(f"Starting optimization for {study_name}")
        optimization_start = time.time()
        study.optimize(
            objective, 
            n_trials=config.n_trials, 
            timeout=config.timeout_sec, # .optimize timeout for native optuna time limiting
            callbacks=[
                OptunaCallbacks.time_limit_callback,
                OptunaCallbacks.performance_plateau_callback
            ]
        )

        optimization_time = time.time() - optimization_start
        log_func_info(f"Optimization completed in {optimization_time:.2f}s; Ran {len(study.trials)} trials")

        # Log study statistics
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        n_pruned = len([t for t in study.trials if t.state == TrialState.PRUNED])
        n_fail = len([t for t in study.trials if t.state == TrialState.FAIL])
        
        log_func_info(f"Study {study_name} statistics; Complete: {n_complete}, Pruned: {n_pruned}, Failed: {n_fail}")

        if n_complete == 0:
            log_func_warn(f"No successful trials for {study_name}.")
            # raise
        elif tuning_only:
            model._set_model_params(study.best_trial.params)
        else:
            # Apply best parameters and fit final model
            model._set_model_params(study.best_trial.params)
            final_min_date, final_cutoff_date, final_max_date = args["cv_dates"][-1]
            train, test = select_train_and_test(modeling_data = data, 
                                                date_col = model.date_col, 
                                                min_date_str = final_min_date,
                                                cutoff_date_str = final_cutoff_date, 
                                                max_date_str = final_max_date)
            model.fit(train, model_param_dict=study.best_trial.params)

        return logger, study

    except Exception as e:
        log_func_error(f"Error during hyperparameter optimization: {str(e)}; {traceback.format_exc()}")
        return logger, None