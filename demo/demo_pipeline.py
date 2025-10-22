"""
Simplified Demo Pipeline for TEMPO Forecasting Application

This module provides a streamlined interface to the TEMPO forecasting library,
optimized for demo applications with simplified configuration and faster execution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import traceback
from datetime import datetime, timedelta

from tempo_forecasting.utils.training_utils import calculate_time_periods, calculate_metric
from tempo_forecasting.pipeline.preprocessing_pipeline import preprocess_pipeline
from tempo_forecasting.pipeline.training_pipeline import train_pipeline  
from tempo_forecasting.pipeline.evaluation_pipeline import evaluation_pipeline
from tempo_forecasting.optuna_opt.run_optuna import OptunaConfig


class DemoPipeline:
    """Simplified pipeline wrapper for TEMPO demo applications"""
    
    def __init__(self, 
                 target_col: str = 'n_rented',
                 date_col: str = 'date',
                 category_col: str = 'category',
                 freq: str = 'D'):
        """
        Initialize the demo pipeline
        
        Args:
            target_col: Name of the target variable column
            date_col: Name of the date column
            category_col: Name of the category column  
            freq: Frequency of the time series ('D' for daily, 'M' for monthly, etc.)
        """
        self.target_col = target_col
        self.date_col = date_col
        self.category_col = category_col
        self.freq = freq
        self.results = {}
        self.processed_data = None
        
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate input data format and content
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required columns
        required_cols = [self.date_col, self.category_col, self.target_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            
        if errors:
            return False, errors
            
        # Check data types and content
        try:
            data[self.date_col] = pd.to_datetime(data[self.date_col])
        except:
            errors.append(f"Cannot convert {self.date_col} to datetime")
            
        try:
            data[self.target_col] = pd.to_numeric(data[self.target_col])
        except:
            errors.append(f"Cannot convert {self.target_col} to numeric")
            
        # Check for sufficient data
        if len(data) < 100:
            errors.append("Insufficient data: need at least 100 observations")
            
        # Check categories
        categories = data[self.category_col].unique()
        if len(categories) < 1:
            errors.append("No categories found in data")
            
        # Check for missing values in key columns
        key_missing = data[required_cols].isnull().sum()
        if key_missing.sum() > 0:
            errors.append(f"Missing values found: {key_missing.to_dict()}")
            
        return len(errors) == 0, errors
    
    def prepare_demo_config(self, 
                           data: pd.DataFrame,
                           test_periods: int = 6,
                           n_trials: int = 10,
                           timeout_minutes: int = 5) -> Dict[str, Any]:
        """
        Prepare configuration optimized for demo purposes
        
        Args:
            data: Input DataFrame
            test_periods: Number of periods to hold out for testing
            n_trials: Number of Optuna trials per model
            timeout_minutes: Timeout for optimization in minutes
            
        Returns:
            Configuration dictionary
        """
        # Get data date range
        data_sorted = data.sort_values(self.date_col)
        data_sorted[self.date_col] = pd.to_datetime(data_sorted[self.date_col])
        max_date = data_sorted[self.date_col].max()
        min_date = data_sorted[self.date_col].min()
        
        # Calculate time periods for demo (simplified)
        total_periods = len(data_sorted[self.date_col].unique())
        
        if total_periods < 30:
            raise ValueError(f"Need at least 30 time periods, found {total_periods}")
            
        # Configure cross-validation windows (simplified for demo)
        # Very forgiving configuration for demo to ensure models have enough training data
        available_train_periods = total_periods - test_periods
        
        # Ensure we have at least 2+ years of training data for the strictest model (expsmooth needs 730 days)
        if available_train_periods < 800:  # Less than ~2.2 years
            # Use minimal CV with maximum training data
            n_train_months = max(24, available_train_periods // 30)  # At least 2 years
            n_validation_sets = 1  # Just one CV window
            cv_window_step_days = 30  # Small step
        else:
            # Normal CV setup
            n_train_months = max(24, (available_train_periods - 30) // 30)  # At least 2 years, leave buffer
            n_validation_sets = min(2, available_train_periods // 365)  # Max 2 CV sets
            cv_window_step_days = max(60, available_train_periods // 15)  # Larger steps
        
        # Convert dates to strings safely
        def safe_date_str(date_val):
            if hasattr(date_val, 'strftime'):
                return date_val.strftime('%Y-%m-%d')
            else:
                return str(date_val)[:10]  # Take first 10 chars if already string
        
        # For daily data, convert test_periods directly - don't divide by 30
        n_test_months_calc = test_periods // 30 if test_periods >= 30 else 1
        
        try:
            time_periods = calculate_time_periods(
                max_date_str=safe_date_str(max_date),
                n_test_months=n_test_months_calc,
                n_train_months=n_train_months,
                n_validation_sets=n_validation_sets,
                cv_window_step_days=cv_window_step_days,
                verbose=False
            )
        except Exception as e:
            # Fallback to simple train/test split with much more training data
            cutoff_date = data_sorted[self.date_col].iloc[-(test_periods + 10)]  # Add buffer
            time_periods = {
                'cv_windows': [{
                    'min': safe_date_str(min_date),
                    'cutoff': safe_date_str(cutoff_date), 
                    'max': safe_date_str(max_date)
                }],
                'retrain_range': {
                    'min': safe_date_str(min_date),
                    'max': safe_date_str(max_date)
                }
            }
        
        # Prepare arguments
        cv_dates_list = [[d["min"], d["cutoff"], d["max"]] for d in time_periods["cv_windows"]]
        
        args = {
            'date_col': self.date_col,
            'target_y': self.target_col,
            'freq': self.freq,
            'cv_dates': cv_dates_list,
            'retrain_dates': [time_periods["retrain_range"]["min"], time_periods["retrain_range"]["max"]],
            'use_parallel': False
        }
        
        # Optuna configuration optimized for demo speed
        optuna_config = OptunaConfig(
            n_trials=n_trials,
            n_startup_trials=min(5, n_trials//2),
            timeout_sec=timeout_minutes * 60,
            eval_metric="wmape"
        )
        
        return {
            'args': args,
            'optuna_config': optuna_config,
            'time_periods': time_periods,
            'test_periods': test_periods
        }
    
    def run_forecasting(self, 
                       data: pd.DataFrame,
                       test_periods: int = 6,
                       selected_models: Optional[List[str]] = None,
                       n_trials: int = 10,
                       progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline
        
        Args:
            data: Input DataFrame
            test_periods: Number of periods to hold out for testing
            selected_models: List of model names to use (None for default subset)
            n_trials: Number of Optuna trials
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary containing results, metrics, and visualizations data
        """
        if progress_callback:
            progress_callback("Validating data...", 0.1)
            
        # Validate data
        is_valid, errors = self.validate_data(data.copy())
        if not is_valid:
            raise ValueError(f"Data validation failed: {'; '.join(errors)}")
        
        if progress_callback:
            progress_callback("Preparing configuration...", 0.2)
            
        # Prepare configuration
        config = self.prepare_demo_config(data, test_periods, n_trials)
        
        # Get categories
        categories = data[self.category_col].unique()
        if progress_callback:
            progress_callback(f"Processing {len(categories)} categories...", 0.3)
        
        # Run pipeline for each category
        results = {}
        param_dfs = []
        output_data_dfs = []
        processing_log = []  # Capture detailed logs
        
        for i, category in enumerate(categories):
            try:
                if progress_callback:
                    progress = 0.3 + (i / len(categories)) * 0.6
                    progress_callback(f"Processing category: {category}", progress)
                
                # Filter data for category
                category_data = data[data[self.category_col] == category].drop(self.category_col, axis=1).copy()
                category_data = category_data.sort_values(self.date_col).reset_index(drop=True)
                
                log_msg = f"Processing category '{category}' with {len(category_data)} data points"
                print(log_msg)
                processing_log.append(log_msg)
                
                log_msg = f"Date range: {category_data[self.date_col].min()} to {category_data[self.date_col].max()}"
                print(log_msg)
                processing_log.append(log_msg)
                
                # Step 1: Preprocessing
                try:
                    preproc_output = preprocess_pipeline(
                        category=category,
                        category_data=category_data,
                        args=config['args'],
                        logger=None
                    )
                    
                    # Handle variable return values from preprocessing
                    if len(preproc_output) == 4:
                        model_set, cv_dates, _, _ = preproc_output
                        log_msg = f"Preprocessing returned 4 values (error case) for {category}"
                    elif len(preproc_output) == 3:
                        model_set, cv_dates, _ = preproc_output
                        log_msg = f"Preprocessing returned 3 values (normal case) for {category}"
                    else:
                        raise ValueError(f"Unexpected number of return values from preprocess_pipeline: {len(preproc_output)}")
                    
                    print(log_msg)
                    processing_log.append(log_msg)
                    
                    log_msg = f"Preprocessing successful for {category}: {len(model_set)} models available"
                    if len(model_set) == 0:
                        log_msg += f" (No models passed data requirements - may need more training data or different CV setup)"
                    print(log_msg)
                    processing_log.append(log_msg)
                except Exception as e:
                    log_msg = f"Preprocessing failed for category {category}: {str(e)}"
                    print(log_msg)
                    processing_log.append(log_msg)
                    continue
                
                # Filter models if specified (for demo speed)
                if selected_models:
                    model_set = {k: v for k, v in model_set.items() if k in selected_models}
                else:
                    # Use a fast subset for demo
                    demo_models = ['prophet', 'xgboost', 'expsmooth']
                    model_set = {k: v for k, v in model_set.items() if k in demo_models}
                
                log_msg = f"Selected models for {category}: {list(model_set.keys())}"
                print(log_msg)
                processing_log.append(log_msg)
                
                if not model_set:
                    log_msg = f"No models available for category {category}"
                    print(log_msg)
                    processing_log.append(log_msg)
                    continue
                
                # Step 2: Training
                try:
                    train_output = train_pipeline(
                        category=category,
                        models=model_set,
                        modeling_data=category_data,
                        args=config['args'],
                        optuna_config=config['optuna_config'],
                        override_cv_dates=cv_dates,
                        logger="print"
                    )
                    
                    category_results, _ = train_output
                    results[category] = category_results
                    log_msg = f"Training successful for category {category}"
                    print(log_msg)
                    processing_log.append(log_msg)
                except Exception as e:
                    log_msg = f"Training failed for category {category}: {str(e)}"
                    print(log_msg)
                    processing_log.append(log_msg)
                    import traceback
                    error_trace = traceback.format_exc()
                    processing_log.append(f"Training error trace for {category}:\n{error_trace}")
                    continue
                
                # Step 3: Evaluation (modified for demo to show all models)
                try:
                    category_param_df, category_output_data_df = self._evaluate_all_models(
                        category=category,
                        category_results=category_results,
                        config=config
                    )
                    
                    param_dfs.append(category_param_df)
                    output_data_dfs.append(category_output_data_df)
                    log_msg = f"Evaluation successful for category {category} - {len(category_param_df)} models"
                    print(log_msg)
                    processing_log.append(log_msg)
                except Exception as e:
                    log_msg = f"Evaluation failed for category {category}: {str(e)}"
                    print(log_msg)
                    processing_log.append(log_msg)
                    import traceback
                    error_trace = traceback.format_exc()
                    processing_log.append(f"Evaluation error trace for {category}:\n{error_trace}")
                    continue
                
            except Exception as e:
                log_msg = f"Unexpected error processing category {category}: {str(e)}"
                print(log_msg)
                processing_log.append(log_msg)
                import traceback
                error_trace = traceback.format_exc()
                processing_log.append(f"Unexpected error trace for {category}:\n{error_trace}")
                continue
        
        if progress_callback:
            progress_callback("Finalizing results...", 0.9)
        
        # Combine results
        if param_dfs:
            combined_param_df = pd.concat(param_dfs, ignore_index=True)
            combined_output_df = pd.concat(output_data_dfs, ignore_index=True)
        else:
            error_msg = f"""No categories were successfully processed.
            
            Total categories attempted: {len(categories)}
            Categories: {list(categories)}
            
            This usually happens due to:
            1. Insufficient data per category (need at least 30+ time periods)
            2. Date format issues
            3. Cross-validation configuration problems
            4. Model initialization failures
            
            Detailed processing log:
            {chr(10).join(processing_log)}"""
            raise ValueError(error_msg)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(combined_param_df, combined_output_df)
        
        if progress_callback:
            progress_callback("Complete!", 1.0)
        
        # Store results
        self.results = {
            'param_df': combined_param_df,
            'output_df': combined_output_df,
            'raw_results': results,
            'config': config,
            'summary_metrics': summary_metrics,
            'categories': list(categories),
            'test_periods': test_periods,
            'processing_log': processing_log
        }
        
        return self.results
    
    def _evaluate_all_models(self, category: str, category_results: Dict[str, Any], config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate all models for a category (demo version that keeps all models, not just the best)
        
        Returns:
            Tuple of (param_df, output_df) containing results for ALL models
        """
        from tempo_forecasting.utils.training_utils import calculate_metric
        
        param_rows = []
        output_rows = []
        
        # Process each model that was trained
        for model_name, model_data in category_results['models'].items():
            # Extract model results
            cv_metrics = model_data['cv_metrics']
            best_params = model_data['model_params']
            
            # Get CV metrics
            cv_avg_wmape = cv_metrics['cv_best_mean_all_metrics']['WMAPE']
            cv_all_wmape = cv_metrics['cv_best_full_all_metrics']['WMAPE']
            
            # Calculate simple train/test metrics
            data_vals = category_results['data']['data_vals']
            train_preds = model_data['train_preds']
            test_preds = model_data['test_preds']
            
            # For demo purposes, use available data to calculate metrics
            train_wmape = calculate_metric(data_vals[:len(train_preds)], train_preds, metric='WMAPE')
            test_wmape = calculate_metric(data_vals[-len(test_preds):], test_preds, metric='WMAPE')
            
            # Create parameter row
            param_row = {
                'category': category,
                'model_name': model_name,
                'best_params': str(best_params),
                'metric_type': 'WMAPE',
                'cv_avg_metric': cv_avg_wmape,
                'cv_all_metrics': cv_all_wmape,
                'eval_train_metric': train_wmape,
                'eval_test_metric': test_wmape,
                'future_forecast_metric': 0.0,  # Placeholder for demo
                'forecast_horizon': config['test_periods']
            }
            param_rows.append(param_row)
            
            # Create output data rows matching expected structure
            data_dates = category_results['data']['data_dates']
            
            # Combine all data into single rows (matching original evaluation pipeline structure)
            for i, date in enumerate(data_dates):
                actual_val = data_vals[i] if i < len(data_vals) else None
                train_pred = train_preds[i] if i < len(train_preds) else None
                
                # Determine if this is in test period
                test_start_idx = len(data_vals) - len(test_preds)
                test_pred = None
                if i >= test_start_idx and (i - test_start_idx) < len(test_preds):
                    test_pred = test_preds[i - test_start_idx]
                
                output_row = {
                    'category': category,
                    'model_name': model_name,
                    'date': date,
                    'true_vals': actual_val,
                    'eval_train_preds': train_pred,
                    'eval_test_preds': test_pred,
                    'final_train_preds': train_pred,  # For compatibility
                    'forecast': None  # No future forecasts in demo for simplicity
                }
                output_rows.append(output_row)
        
        return pd.DataFrame(param_rows), pd.DataFrame(output_rows)
    
    def _calculate_summary_metrics(self, param_df: pd.DataFrame, output_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary metrics across all categories"""
        
        summary = {
            'total_categories': len(param_df),
            'models_used': param_df['model_name'].unique().tolist(),
            'avg_wmape': param_df['cv_avg_metric'].mean(),
            'best_wmape': param_df['cv_avg_metric'].min(),
            'worst_wmape': param_df['cv_avg_metric'].max(),
        }
        
        # Model performance summary
        model_performance = param_df.groupby('model_name').agg({
            'cv_avg_metric': ['mean', 'std', 'count']
        }).round(3)
        model_performance.columns = ['avg_wmape', 'std_wmape', 'count']
        summary['model_performance'] = model_performance.to_dict('index')
        
        # Best model by category
        summary['best_models'] = param_df[['category', 'model_name', 'cv_avg_metric']].to_dict('records')
        
        return summary
    
    def get_forecast_vs_actual_data(self) -> pd.DataFrame:
        """
        Extract forecast vs actual data for the test periods
        
        Returns:
            DataFrame with actual vs predicted values for visualization
        """
        if not self.results:
            raise ValueError("No results available. Run forecasting first.")
        
        output_df = self.results['output_df']
        
        # Get test period data (where we have both actuals and predictions)
        test_data = output_df[
            (~output_df['true_vals'].isna()) & 
            (~output_df['eval_test_preds'].isna())
        ].copy()
        
        if len(test_data) == 0:
            # Fallback to final forecast data
            test_data = output_df[
                (~output_df['true_vals'].isna()) & 
                (~output_df['final_train_preds'].isna())
            ].copy()
            test_data = test_data.tail(self.results['test_periods'] * len(self.results['categories']))
        
        return test_data
    
    def get_model_comparison_data(self) -> pd.DataFrame:
        """Get data for model performance comparison visualization"""
        if not self.results:
            raise ValueError("No results available. Run forecasting first.")
            
        param_df = self.results['param_df']
        
        # Create comparison dataframe
        comparison_data = []
        for _, row in param_df.iterrows():
            comparison_data.append({
                'category': row['category'],
                'model_type': row['model_name'],
                'wmape': row['cv_avg_metric'],
                'mae': row.get('eval_test_metric', row['cv_avg_metric']),  # Fallback to WMAPE if MAE not available
                'is_best_model': True  # All are best for their category in this simplified version
            })
        
        return pd.DataFrame(comparison_data)


def run_demo_pipeline(data: pd.DataFrame, 
                     test_periods: int = 6,
                     progress_callback: Optional[callable] = None) -> Dict[str, Any]:
    """
    Convenience function to run the demo pipeline with default settings
    
    Args:
        data: Input DataFrame
        test_periods: Number of periods to hold out
        progress_callback: Progress callback function
        
    Returns:
        Results dictionary
    """
    pipeline = DemoPipeline()
    return pipeline.run_forecasting(data, test_periods=test_periods, progress_callback=progress_callback)


if __name__ == "__main__":
    # Demo usage
    from synthetic_data_generator import SyntheticDataGenerator
    
    # Generate sample data
    generator = SyntheticDataGenerator()
    data = generator.generate_demo_dataset('retail_sales')
    
    print("Running demo pipeline...")
    
    def progress_callback(message, progress):
        print(f"Progress {progress:.1%}: {message}")
    
    # Run pipeline
    pipeline = DemoPipeline()
    results = pipeline.run_forecasting(data, test_periods=30, progress_callback=progress_callback)
    
    print("\nResults summary:")
    print(f"Processed {results['summary_metrics']['total_categories']} categories")
    print(f"Average WMAPE: {results['summary_metrics']['avg_wmape']:.3f}")
    print(f"Models used: {results['summary_metrics']['models_used']}")