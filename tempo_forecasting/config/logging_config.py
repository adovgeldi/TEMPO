# Centralized logging configuration

# Unity Catalog settings
SCHEMA_NAME = "logs"
LOG_TABLE_NAME = "ml_training_logs"

# Logging parameters
LOG_RETENTION_DAYS = 30
DEFAULT_BUFFER_SIZE = 10
DEFAULT_COMPONENT = "ml_training"

# Log levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# Minimum log level to record
MIN_LOG_LEVEL = "INFO"