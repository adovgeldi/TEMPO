import os
import socket
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple

from pyspark.sql import SparkSession
from pyspark.sql.functions import spark_partition_id
from tempo_forecasting.logging.log_manager import log_batch_to_delta
from tempo_forecasting.config.logging_config import SCHEMA_NAME, LOG_TABLE_NAME


class WorkerLogger:
    """Logger for distributed worker nodes"""
    def __init__(self, run_id, catalog_name: str, component="default"):
        self.worker_id = f"{socket.gethostname()}_{os.getpid()}"
        self.run_id = run_id
        self.component = component
        self.log_buffer = []
        self.catalog_name = catalog_name
    
    def log(self, level, category, message, details=""):
        """Add a log entry to the buffer"""
        log_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "worker_id": self.worker_id,
            "level": level,
            "component": self.component,
            "category": category,
            "message": message,
            "details": details
        }
        self.log_buffer.append(log_entry)
        
        # Print to stdout for immediate feedback
        print(f"[{level}] {self.worker_id} - {message}")
        return log_entry
    
    def info(self, message, category="", details=""):
        """Log an info message"""
        return self.log("INFO", category, message, details)
    
    def warning(self, message, category="", details=""):
        """Log a warning message"""
        return self.log("WARNING", category, message, details)
    
    def error(self, message, category="", details=""):
        """Log an error message"""
        return self.log("ERROR", category, message, details)
    
    def debug(self, message, category="", details=""):
        """Log a debug message"""
        return self.log("DEBUG", category, message, details)
    
    def flush(self):
        """Flush the log buffer and return the entries"""
        entries = self.log_buffer.copy()
        self.log_buffer = []
        return entries
    
    def serialize_logs(self):
        """Serialize logs for transmission from workers"""
        logs = self.flush()
        return json.dumps([
            {
                "level": log["level"],
                "category": log["category"],
                "message": log["message"],
                "details": log["details"],
                "timestamp": log["timestamp"],
                "worker_id": log["worker_id"],
                "run_id": log["run_id"],
                "component": log["component"]
            }
            for log in logs
        ])
    
    @staticmethod
    def deserialize_logs(serialized_logs: str) -> List[Dict[str, Any]]:
        """Deserialize logs received from workers"""
        if not serialized_logs or serialized_logs == "":
            return []
            
        log_data = json.loads(serialized_logs)
        return [
            {
                "level": item["level"],
                "category": item["category"],
                "message": item["message"],
                "details": item["details"],
                "timestamp": item["timestamp"],
                "worker_id": item["worker_id"],
                "run_id": item["run_id"],
                "component": item["component"]
            }
            for item in log_data
        ]
    
    def merge_logs(self, serialized_logs: str):
        """Merge serialized logs from workers into this logger's buffer"""
        deserialized_logs = self.deserialize_logs(serialized_logs)
        self.log_buffer.extend(deserialized_logs)
    
    def write_logs_to_delta(self):
        """Write the current log buffer to Delta table"""
        if not self.log_buffer:
            print("No logs to write")
            return
            
        spark = SparkSession.builder.getOrCreate()
        log_batch_to_delta(spark, self.catalog_name, SCHEMA_NAME, LOG_TABLE_NAME, self.log_buffer)
        print(f"Wrote {len(self.log_buffer)} logs to Delta table")
        self.log_buffer = []