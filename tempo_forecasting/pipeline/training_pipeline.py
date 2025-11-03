import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from tempo_forecasting.utils.training_utils import calculate_metric, enforce_datetime_index, select_train_and_test
from tempo_forecasting.optuna_opt.run_optuna import OptunaConfig, run_tuned_fit
import traceback


def train_pipeline(category: str, 
                   models: Dict[str, Any], 
                   modeling_data: pd.DataFrame, 
                   args: Dict[str, Any], 
                   optuna_config: OptunaConfig,
                   override_cv_dates = None,
                   logger = None
                   ) -> Tuple[Dict[str, Any], Optional[Any]]:
    """
    Trains models for a given category using Optuna for hyperparameter tuning.
    
    Ensures that:
        - Models have enough training data (as determined in preprocessing).
        - Models receive the correct forecast horizon.
        - Predictions and metrics are collected for evaluation.

    Parameters:
        - category (str): The category name (e.g., machine type), for logging purposes.
        - train_data (DataFrame): The training dataset.
        - test_data (DataFrame): The testing dataset.
        - args (Dict[str, Any]): A dictionary containing at least the following keys:
            - 'date_col' (str): The name of the column containing datetime information.
            - 'target_y' (str): The name of the target variable column.
        - optuna_config (OptunaConfig): Configurations for Optuna tuning.

    Returns:
        dict: A dictionary containing results for each trained model.
    """
    if logger is None:
        from tempo_forecasting.utils.logging_utils import logger as default_logger
        logger = default_logger
        log_func_info = lambda msg: logger.info(msg)
        log_func_error = lambda msg: logger.error(msg)
    elif logger == "print":
        log_func_info = lambda msg: print(msg)
        log_func_error = lambda msg: print(msg)
    else:
        # Use the WorkerLogger's category-aware logging 
        log_func_info = lambda msg, details="Training Pipeline": logger.info(message=msg, category=category, details=details)
        log_func_error = lambda msg, details="Training Pipeline": logger.error(message=msg, category=category, details=details)

    modeling_data = enforce_datetime_index(df=modeling_data.copy(), date_col=args["date_col"])
    cv_dates = override_cv_dates or args["cv_dates"]

    model_results = {}
    for model_name, model_class in models.items():
        try:
            # Copy of data that will be used for ALL model types
            # so as to not accidentally change the original pandas df
            data = modeling_data.copy()

            # Instantiate the model
            model = model_class(target_y=args['target_y'], date_col=args['date_col'])
            
            # Train model
            optuna_logger, study = run_tuned_fit(model, 
                                                category,
                                                data, 
                                                cv_dates = cv_dates, 
                                                config = optuna_config,
                                                study_name = f"{category} - {model_name} model",
                                                logger=logger,
                                                tuning_only = True
                                                )
            
            if optuna_logger is not None:
                logger = optuna_logger
                
            best_params = study.best_trial.params
            log_func_info(f"Best parameters for {model_name} model: {best_params}, best value: {study.best_value}")

            # Generate train and test from final cv window
            #   will call simple_train_test_dates later
            #   but also want vars to make it clear what each component of the date window triple is
            simple_train_test_dates = cv_dates[-1] 
            final_min_date, final_cutoff_date, final_max_date = simple_train_test_dates
            train, test = select_train_and_test(modeling_data = data, 
                                                date_col = args["date_col"], 
                                                min_date_str = final_min_date,
                                                cutoff_date_str = final_cutoff_date, 
                                                max_date_str = final_max_date)

            # Train model using the best parameters
            model.fit(train, model_param_dict=best_params)

            # Make predictions for test data
            predictions = model.predict(test) 
            train_preds = model.fitted_vals # these should be fit.fittedvalues
            
            # Compile cv metrics
            cv_all_trial_metrics = {}
            for trial in study.trials:
                if (not trial.user_attrs["repeat_trial"]) & (trial.state == 1):
                    # captures only distinct trials that completed
                    for m in trial.user_attrs["cv_metrics"]:
                            existing_metrics = cv_all_trial_metrics.get(m,[])
                            cv_all_trial_metrics[m] = existing_metrics + [trial.user_attrs["cv_metrics"][m]]


            cv_metrics = {
                "cv_best_tuning_metric": study.best_trial.values[0],
                "cv_best_mean_all_metrics": {k:np.mean(v) for k,v in study.best_trial.user_attrs["cv_metrics"].items()},
                "cv_best_full_all_metrics": study.best_trial.user_attrs["cv_metrics"],
                "cv_all_trials_all_metrics": cv_all_trial_metrics
            }

            # Compile Date Info
            split_dates = {
                "cv_dates": cv_dates,
                "simple_train_test_dates": simple_train_test_dates
            }

            # Store results
            model_results[model_name] = {
                "cv_metrics": cv_metrics,
                "train_preds": np.array(train_preds.squeeze()),
                "test_preds": predictions,
                "model_params": best_params
            }
                        
        except Exception as e:
            log_func_error(f"Error training {model_name} model for {category}: {str(e)}; Traceback: {traceback.format_exc()}")
            continue

    category_results = {
        "models": model_results,
        "data": {
            "data_vals": np.array(data[args['target_y']]),
            "data_dates": np.array(data.index.strftime("%Y-%m-%d")),
            "train_test_split_dates": split_dates,
        }
    }

    return category_results, logger