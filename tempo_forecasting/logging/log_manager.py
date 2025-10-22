from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import pandas as pd


def create_log_table(catalog_name, schema_name, table_name):
    """Create or confirm the centralized logging table"""
    spark = SparkSession.builder.getOrCreate()

    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {full_table_name} (
        run_id STRING,
        timestamp TIMESTAMP,
        worker_id STRING,
        level STRING,
        component STRING,
        category STRING,
        message STRING,
        details STRING
    )
    USING DELTA
    PARTITIONED BY (run_id)
    """
    
    try:
        spark.sql(create_table_sql)
        return True
    except Exception as e:
        print(f"Error creating log table: {str(e)}")
        return False


def log_batch_to_delta(spark: SparkSession, catalog_name: str, schema_name: str, table_name: str, log_entries: list):
    """
    Write a batch of log entries to a Delta table
    
    Args:
        spark: SparkSession
        catalog_name: The catalog name for the Delta table
        schema_name: The schema name for the Delta table
        table_name: The table name for the Delta table
        log_entries: List of log entry dictionaries
    """
    if not log_entries:
        print("No log entries to write")
        return
    
    # Convert datetime objects to strings if needed
    processed_entries = []
    for entry in log_entries:
        processed_entry = entry.copy()
        if isinstance(processed_entry.get("timestamp"), datetime):
            processed_entry["timestamp"] = processed_entry["timestamp"].isoformat()
        processed_entries.append(processed_entry)
    
    # Create a DataFrame from the log entries
    log_df = pd.DataFrame(processed_entries)
    
    # Convert to Spark DataFrame
    log_schema = StructType([
        StructField("run_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("worker_id", StringType(), True),
        StructField("level", StringType(), True),
        StructField("component", StringType(), True),
        StructField("category", StringType(), True),
        StructField("message", StringType(), True),
        StructField("details", StringType(), True)
    ])
    
    spark_log_df = spark.createDataFrame(log_df, schema=log_schema)
    
    # Convert timestamp string to timestamp
    spark_log_df = spark_log_df.withColumn("timestamp", spark_log_df["timestamp"].cast(TimestampType()))
    
    # Write to Delta table
    table_path = f"{catalog_name}.{schema_name}.{table_name}"
    print(f"Writing {len(log_entries)} log entries to {table_path}")
    
    spark_log_df.write.format("delta").mode("append").saveAsTable(table_path)
    
    print(f"Successfully wrote {len(log_entries)} log entries to {table_path}")


def query_logs(catalog_name, schema_name, table_name, run_id=None, 
               component=None, level=None, limit=100):
    """Query logs with optional filtering"""
    spark = SparkSession.builder.getOrCreate()

    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    
    query = f"SELECT * FROM {full_table_name} WHERE 1=1"
    
    if run_id:
        query += f" AND run_id = '{run_id}'"
    if component:
        query += f" AND component = '{component}'"
    if level:
        query += f" AND level = '{level}'"
    
    query += " ORDER BY timestamp DESC"
    
    if limit:
        query += f" LIMIT {limit}"
    
    return spark.sql(query)


def cleanup_old_logs(catalog_name, schema_name, table_name, days_to_keep=30):
    """Delete logs older than the specified number of days"""
    spark = SparkSession.builder.getOrCreate()
    
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    
    from datetime import datetime, timedelta
    cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime("%Y-%m-%d")
    
    spark.sql(f"DELETE FROM {full_table_name} WHERE timestamp < '{cutoff_date}'")