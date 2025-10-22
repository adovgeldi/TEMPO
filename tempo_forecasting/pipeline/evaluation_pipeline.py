import pandas as pd
import numpy as np
from tempo_forecasting.utils.training_utils import calculate_metric, select_train_and_test
from tempo_forecasting.utils.config_utils import get_models
from typing import Dict, Any, Optional, Tuple

def evaluation_pipeline(
        category: str, 
        category_results: Dict[str, Any],
        args,
        target_metric: str ="WMAPE",
        forecast_horizon: int = 365,
        logger = None
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Any]]:
    """
    Evaluates trained models, selects the best model for each category, and stores results.

    Parameters:
        category (str): The category name (e.g., machine type).
        category_results (dict): Dictionary of trained models and their metrics.
        target_metric (str): The metric to determine the best model (default is WMAPE).
        logger (WorkerLogger, optional): Logger instance for recording progress.

    Returns:
        dict: A dictionary containing the best model for the category and its performance.
    """
    if logger is None: # Use standard logging
        from tempo_forecasting.utils.logging_utils import logger as default_logger
        logger = default_logger
        log_func_info = lambda msg: logger.info(msg)
    elif logger == "print":
        log_func_info = lambda msg: print(msg)
    else:
        # Use the WorkerLogger's category-aware logging if passed in
        log_func_info = lambda msg, details="Evaluation Pipeline": logger.info(message=msg, category=category, details=details)
    
    # Ensure we have results to evaluate
    if not category_results:
        log_func_info(f"No valid models trained for category {category}. Skipping evaluation.")
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df, empty_df, logger

    log_func_info(f"Starting evaluation for category: {category}")
    log_func_info(f"Available models: {list(category_results['models'].keys())}")

    # Select the best model based on the lowest avg target metric (e.g., WMAPE)
    best_model_name = min(category_results["models"], 
                          key=lambda x: category_results['models'][x]['cv_metrics']['cv_best_mean_all_metrics'][target_metric])
    cv_best_avg_metric = category_results['models'][best_model_name]['cv_metrics']['cv_best_mean_all_metrics'][target_metric]
    cv_best_metric_all = category_results['models'][best_model_name]['cv_metrics']['cv_best_full_all_metrics'][target_metric]
    log_func_info(f"Selected best model: {best_model_name}, with mean cv {target_metric}: {cv_best_avg_metric}")

    # Extract best model's results and model
    best_model_data = category_results['models'][best_model_name]
    best_params = best_model_data["model_params"]

    models = get_models(how="all")
    best_model_class = models[best_model_name]

    # Log parameter details
    log_func_info(f"Best model parameters: {best_params}")

    # Re-Process Data
    full_data = pd.DataFrame({
        args["date_col"]: category_results["data"]["data_dates"],
        args["target_y"]: category_results["data"]["data_vals"]
    })

    # Combine eval data
    eval_split_dates = category_results["data"]["train_test_split_dates"]['simple_train_test_dates']

    log_func_info(f"Combining evaluation train and test data")
    eval_train_data, eval_test_data = select_train_and_test(modeling_data = full_data, 
                                                date_col = args["date_col"], 
                                                min_date_str = eval_split_dates[0],
                                                cutoff_date_str = eval_split_dates[1],
                                                max_date_str = eval_split_dates[2])
    
    # Calculate simple train and test metrics
    eval_train_vals = np.array(eval_train_data[args["target_y"]])
    eval_train_metric = calculate_metric(eval_train_vals, best_model_data["train_preds"],metric=target_metric)
    log_func_info(f"Train {target_metric}: {eval_train_metric}")

    eval_test_vals = np.array(eval_test_data[args["target_y"]])
    eval_test_metric = calculate_metric(eval_test_vals, best_model_data["test_preds"],metric=target_metric)
    log_func_info(f"Test {target_metric}: {eval_test_metric}")

    # Select data for final modeling
    log_func_info(f"Combining train and test data for full retraining")
    final_min_date, final_max_date = args["retrain_dates"]
    final_train_data, _ = select_train_and_test(modeling_data = full_data, 
                                                date_col = args["date_col"], 
                                                min_date_str = final_min_date,
                                                cutoff_date_str = final_max_date,
                                                max_date_str = final_max_date)

    # Retrain best model on final train set
    log_func_info(f"Retraining {best_model_name} on final training dataset")
    model = best_model_class(target_y=args["target_y"], date_col=args["date_col"])

    try:
        model.fit(final_train_data.copy(), model_param_dict=best_params)
        log_func_info(f"Successfully retrained model on full dataset")
    except Exception as e:
        log_func_info(f"Error retraining model: {str(e)}", details=str(e))
        raise

    final_train_preds = model.fitted_vals # Predict on full dataset
    final_train_preds = np.maximum(np.ceil(final_train_preds - 0.3).astype(int), 0)
    final_train_metric = calculate_metric(final_train_data[args["target_y"]], final_train_preds, metric=target_metric)  

    # Generate future dates for prediction
    # Create placeholder DataFrame for future forecast
    log_func_info(f"Last date in training data: {final_max_date}")
    future_dates = pd.date_range(start=pd.Timestamp(final_max_date) + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
    future_df = pd.DataFrame(index=future_dates)
    future_df[args["target_y"]] = None  # Placeholder for predictions

    log_func_info(f"Predicting {forecast_horizon} days ahead")
    try:
        future_predictions = model.predict(future_df)
        future_predictions = np.maximum(np.ceil(future_predictions - 0.3).astype(int), 0)
        log_func_info(f"Successfully generated {len(future_predictions)} future predictions")
    except Exception as e:
        log_func_info(f"Error generating future predictions: {str(e)}", details=str(e))
        raise
    
    log_func_info(f"Preparing output DataFrames")

    final_param_df = pd.DataFrame({
            "category": [category],
            "model_name": [best_model_name],
            "best_params": [str(best_model_data["model_params"])],
            "metric_type": [target_metric],
            "cv_avg_metric": [cv_best_avg_metric],
            "cv_all_metrics": [cv_best_metric_all],
            "eval_train_metric": [eval_train_metric],
            "eval_test_metric": [eval_test_metric],
            "final_train_metric": [final_train_metric]
        })
    
    final_vals = full_data.copy()
    final_vals["category"] = category
    final_vals["date"] = final_vals[args["date_col"]]
    final_vals["true_vals"] = final_vals[args["target_y"]]

    drop_cols = [c for c in [args["date_col"],args["target_y"]] if c not in ["category","date","true_vals"]]
    final_vals = final_vals.drop(drop_cols, axis=1)

    eval_train_df = pd.DataFrame({
            "date": np.array(pd.to_datetime(eval_train_data.index).strftime("%Y-%m-%d")).squeeze(),
            "eval_train_preds": best_model_data["train_preds"].squeeze()
        })

    eval_test_df = pd.DataFrame({
            "date": np.array(pd.to_datetime(eval_test_data.index).strftime("%Y-%m-%d")).squeeze(),
            "eval_test_preds": best_model_data["test_preds"].squeeze()
        })
    
    final_fitted_df = pd.DataFrame({
            "date": np.array(pd.to_datetime(final_train_data.index).strftime("%Y-%m-%d")).squeeze(),
            "final_train_preds": final_train_preds.squeeze()
        })

    future_forecast_df = pd.DataFrame({
            "date": pd.to_datetime(future_dates).strftime("%Y-%m-%d"),
            "forecast": future_predictions.squeeze()
        })
    
    final_vals = final_vals.merge(eval_train_df, how="outer", left_on="date", right_on="date")\
                           .merge(eval_test_df, how="outer", left_on="date", right_on="date")\
                           .merge(final_fitted_df, how="outer", left_on="date", right_on="date")\
                           .merge(future_forecast_df, how="outer", left_on="date", right_on="date")
    final_vals["category"] = category

    log_func_info(f"Evaluation complete for {category}")
    
    # Log some summary statistics
    log_func_info(f"Summary: {best_model_name} model with train {target_metric}={final_train_metric:.4f}, forecast horizon={forecast_horizon} days")

    # return future_forecast_df, final_param_df, final_pred_df, final_fitted_df, final_vals, logger
    return final_param_df, final_vals, logger
