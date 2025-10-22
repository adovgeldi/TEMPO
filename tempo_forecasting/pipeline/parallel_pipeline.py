import pyspark.sql as psql
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

import json
import pandas as pd

import sys
import os
import traceback

from typing import Dict, Any, Tuple, Optional, Callable

from tempo_forecasting.optuna_opt.run_optuna import OptunaConfig
from tempo_forecasting.pipeline.preprocessing_pipeline import preprocess_pipeline 
from tempo_forecasting.pipeline.training_pipeline import train_pipeline
from tempo_forecasting.pipeline.evaluation_pipeline import evaluation_pipeline

from tempo_forecasting.logging.log_manager import log_batch_to_delta
from tempo_forecasting.logging.worker_logger import WorkerLogger
from tempo_forecasting.config.logging_config import SCHEMA_NAME, LOG_TABLE_NAME

class ParallelPipeline:
    """
    A class to handle parallel processing of multiple categories through the complete pipeline.
    This class provides functionality to process data across multiple categories in parallel, 
    performing tasks such as data extraction, preprocessing, model training, and evaluation.

    Attributes:
        - n_processes (int): The number of processes to use for parallel computation. If not specified,
                        it defaults to the number of CPU cores minus one.
        - chunk_size (int): The number of categories to process per chunk. Defaults to 1.

    Methods:
        - process_category: Processes a single category through the complete pipeline, returning results or None if processing fails.
        - run: Executes the complete pipeline in parallel across all unique categories in the provided DataFrame.
    """
    def __init__(self,
                 run_id: str,
                 group_col: str,
                 args: Dict[str, Any],
                 optuna_config: OptunaConfig,
                 catalog_name: str,
                 target_metric: str = "WMAPE"):
        print("Parallel Pipeline Initialized on Databricks")

        self.run_id = run_id
        self.group_col = group_col
        self.args = args
        self.optuna_config = optuna_config
        self.target_metric = target_metric
        self.catalog_name = catalog_name

        self.result_schema = T.StructType([
            T.StructField("category", T.StringType(), True),
            T.StructField("model_parameters", T.StringType(), True),
            T.StructField("model_data", T.StringType(), True),
            T.StructField("full_results", T.StringType(), True),
            T.StructField("worker_logs", T.StringType(), True)
        ])

        self.logger = WorkerLogger(run_id=self.run_id, component="parallel_process", catalog_name=self.catalog_name)

    def create_forecast_function(self) -> Callable:
        def process_category(pdf: pd.DataFrame) -> pd.DataFrame:
            category = str(pdf[self.group_col].iloc[0])
            category_logger = WorkerLogger(run_id=self.run_id, component=f"process_{category}", catalog_name=self.catalog_name)
            category_logger.info(category=category, message=f"Processing category: {category}\n")

            try:
                results = {}
                    
                # Filter the DataFrame for the current category
                category_data = pdf.drop(self.group_col, axis=1).copy()

                # Step 1: Preprocessing
                category_logger.info(category=category, message=f"Running preprocessing for {category}")

                preproc_output = preprocess_pipeline(category = category,
                                                     category_data = category_data, 
                                                     args = self.args,
                                                     logger = category_logger)
                model_set, cv_dates, preproc_logger = preproc_output

                # Update logger if a new one was returned
                if preproc_logger is not None:
                    category_logger = preproc_logger
                category_logger.info(category=category, 
                                     message=f"Preprocessing complete. Models: {model_set}, CV dates: {cv_dates}")
                
                # Step 2: Training
                category_logger.info(category=category, 
                                     message=f"Running training for {category}")
     
                train_output = train_pipeline(category = category, 
                                               models = model_set, 
                                               modeling_data = category_data, 
                                               args = self.args, 
                                               optuna_config = self.optuna_config,
                                               override_cv_dates = cv_dates,
                                               logger = category_logger)   
                category_results, training_logger = train_output
                results[category] = category_results
                
                # Update logger if a new one was returned
                if training_logger is not None:
                    category_logger = training_logger
                category_logger.info(category=category, 
                                     message=f"Training complete. Results type: {type(category_results)}")

                # Step 3: Evaluation
                category_logger.info(category=category, 
                                     message=f"Running evaluation for {category}")

                eval_output = evaluation_pipeline(category = category,
                                                  category_results = category_results,
                                                  args = self.args,
                                                  target_metric = self.target_metric,
                                                  forecast_horizon = 365,
                                                  logger = category_logger)
                param_df, output_data_df, eval_logger = eval_output

                # Update logger if a new one was returned
                if eval_logger is not None:
                    category_logger = eval_logger
                
                # Build param and output data jsons to pass out of parallel pipeline
                params_json = param_df.to_json(orient="records")
                output_data_json = output_data_df.to_json(orient="records")

                # Build results json, slightly more complicated but same idea
                results_cols = ["category","model","cv_best_avg_metrics",
                                "cv_best_all_metrics","cv_all_trials_all_metrics","best_params"]
                results_df = pd.DataFrame(columns=results_cols)

                for cat in results:
                    for model in results[cat]["models"]:
                        cm_results = results[cat]["models"][model]

                        new_row = {'category': cat, 
                                    'model': model,
                                    'cv_best_avg_metrics': cm_results["cv_metrics"]["cv_best_mean_all_metrics"],
                                    'cv_best_all_metrics': cm_results["cv_metrics"]["cv_best_full_all_metrics"],
                                    'cv_all_trials_all_metrics': cm_results["cv_metrics"]["cv_all_trials_all_metrics"],
                                    'best_params': cm_results["model_params"],
                                    }
                        
                        results_df = pd.concat([results_df,pd.DataFrame([new_row])],ignore_index=True)

                results_json = results_df.to_json(orient="records")

                # Serialize logs for return
                logs_json = category_logger.serialize_logs()
            
                # Create return DataFrame
                output =  pd.DataFrame([{
                    "category": category,
                    "model_parameters": params_json,
                    "model_data": output_data_json,
                    "full_results": results_json,
                    "worker_logs": logs_json
                }])

                return output
            
            except Exception as e:
                # Serialize exception details
                error_message = f"Error processing category {category}: {str(e)}"
                category_logger.error(category=category, message=error_message, details=str(e))
                
                # Return error result with logs
                output = pd.DataFrame([{
                    "category": category,
                    "model_parameters": "{}",
                    "model_data": "{}",
                    "full_results": "{}",
                    "worker_logs": category_logger.serialize_logs()
                }])
                
                return output

        return process_category

    def run_parallel_forecasting(
        self,
        spark_df, 
        num_partitions=None):
        """
        Run forecasting in parallel across all categories
        """
        self.logger.info(message=f"Starting parallel forecasting", category="pipeline")
        self.logger.info(message=f"Total rows: {spark_df.count()}", category="pipeline")
        self.logger.info(message=f"Columns: {spark_df.columns}", category="pipeline")
        
        spark = SparkSession.builder.getOrCreate()
        
        # Calculate partitions
        if num_partitions is None:
            num_partitions = spark_df.select(self.group_col).distinct().count()

        print(f"Setting number of partitions to: {num_partitions}")
        
        # Repartition
        partitioned_df = spark_df.repartition(num_partitions, self.group_col)
        partitioned_df.cache().count()
        
        partition_info = (
            partitioned_df.groupBy(F.spark_partition_id()).count().collect()
        )
        print(f"Partition info: {partition_info}")

        process_function = self.create_forecast_function()
        result_df = partitioned_df.groupBy(self.group_col).applyInPandas(
            process_function,
            schema = self.result_schema
        )
        result_df.cache().count()

        # Collect and process logs from all workers
        self.logger.info(message="Processing logs from workers", category="pipeline")
        worker_logs = result_df.select("category", "worker_logs").collect()
        
        # Merge all worker logs into the main logger
        for row in worker_logs:
            category = row.category
            logs = row.worker_logs
            
            if logs and logs.strip():
                self.logger.info(message=f"Merging logs for category: {category}", category="pipeline")
                self.logger.merge_logs(logs)
            else:
                self.logger.warning(message=f"No logs received for category: {category}", category="pipeline")
        
        # Write all logs to Delta
        self.logger.info(message="Writing all logs to Delta table", category="pipeline")
        self.logger.write_logs_to_delta()
        
        # Return the result DataFrame without the logs column to save memory
        result_data_df = result_df.drop("worker_logs")
        
        return result_data_df
    
    def postprocess_results(
        self,
        result_df: psql.DataFrame
        ) -> Tuple[psql.DataFrame, psql.DataFrame, psql.DataFrame, psql.DataFrame]:

        final_param_schema = T.ArrayType(T.StructType([
            T.StructField("category", T.StringType(), True),
            T.StructField("model_name", T.StringType(), True),
            T.StructField("best_params", T.StringType(), True),
            T.StructField("metric_type", T.StringType(), True),
            T.StructField("cv_avg_metric", T.DoubleType(), True),
            T.StructField("cv_all_metrics", T.ArrayType(T.DoubleType()), True),
            T.StructField("eval_train_metric", T.DoubleType(), True),
            T.StructField("eval_test_metric", T.DoubleType(), True),
            T.StructField("final_train_metric", T.DoubleType(), True)
        ]))

        final_data_schema = T.ArrayType(T.StructType([
            T.StructField("category", T.StringType(), True),
            T.StructField("date", T.StringType(), True),
            T.StructField("true_vals", T.DoubleType(), True),
            T.StructField("eval_train_preds", T.DoubleType(), True),
            T.StructField("eval_test_preds", T.DoubleType(), True),
            T.StructField("final_train_preds", T.DoubleType(), True),
            T.StructField("forecast", T.DoubleType(), True),
        ]))

        full_results_schema = T.ArrayType(T.StructType([
            T.StructField("category", T.StringType(), True),
            T.StructField("model", T.StringType(), True),
            T.StructField("cv_best_avg_metrics", 
                          T.MapType(T.StringType(),T.DoubleType()), True),
            T.StructField("cv_best_all_metrics", 
                          T.MapType(T.StringType(),T.ArrayType(T.DoubleType())), True),
            T.StructField("cv_all_trials_all_metrics", 
                          T.MapType(T.StringType(),T.ArrayType(T.ArrayType(T.DoubleType()))), True),
            T.StructField("best_params", T.StringType(), True)
        ]))

        # Parse each JSON column into a proper Spark struct
        parsed_df = (result_df
            .withColumn("model_parameters", 
                        F.from_json(F.col("model_parameters"), final_param_schema))
            .withColumn("model_data", 
                        F.from_json(F.col("model_data"), final_data_schema))
            .withColumn("full_results", 
                        F.from_json(F.col("full_results"),schema=full_results_schema))
        )

        # Explode each column into its own DataFrame
        model_parameters_df = (parsed_df
            .select(F.explode(F.col("model_parameters")).alias("model_parameters"))
            .selectExpr("model_parameters.*")
        )

        model_data_df = (parsed_df
            .select(F.explode(F.col("model_data")).alias("model_data"))
            .selectExpr("model_data.*")
        )

        full_results_df = (parsed_df
            .select(F.explode(F.col("full_results")).alias("full_results"))
            .selectExpr("full_results.*")
        )

        # Cache results for performance
        model_parameters_df.cache().count()
        model_data_df.cache().count()
        full_results_df.cache().count()

        return model_parameters_df, model_data_df, full_results_df