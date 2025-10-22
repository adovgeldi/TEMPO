import json
from datetime import datetime

def deserialize_logs(serialized_logs, run_id, component="worker"):
    """Convert serialized logs back to log entry dictionaries"""
    logs = json.loads(serialized_logs)
    
    return [{
        "run_id": run_id,
        "timestamp": datetime.now(),
        "worker_id": "deserialized",
        "level": level,
        "component": component,
        "category": category,
        "message": message,
        "details": details
    } for level, category, message, details in logs]

def extract_logs_from_results(results, log_separator="|LOGSEP|"):
    """Extract logs from UDF results"""
    log_entries = []
    
    for row in results:
        result_parts = row.result.split(log_separator)
        if len(result_parts) >= 3:
            result, run_id, serialized_logs = result_parts
            logs = deserialize_logs(serialized_logs, run_id)
            log_entries.extend(logs)
    
    return log_entries