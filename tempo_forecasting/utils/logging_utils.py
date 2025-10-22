import logging
import json
import os
from typing import Optional
import importlib.resources

"""
Log levels let you decide how important a message is. From least to most important, the levels are:
    DEBUG: Detailed information, mostly for developers (e.g., "Starting loop, index = 0").
    INFO: General information about the program's progress (e.g., "User logged in").
    WARNING: Something unexpected happened, but the program can continue (e.g., "Disk space is low").
    ERROR: Something went wrong, and the program might not work properly (e.g., "File not found").
    CRITICAL: A serious error that may cause the program to stop (e.g., "System crash").

** If setting level below to INFO, DEBUG messages won't appear since they are at a lower level, etc.
"""

# Set Matplotlib's logging level to WARNING cause it's annoying...
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Load global configuration
def load_global_config(filename: str = "config/global_config.json") -> dict:
    """
    Loads the global configuration file.

    Parameters:
        filename (str): Path to the global configuration file.

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with importlib.resources.files("tempo_forecasting").joinpath(filename).open("r") as f:
        return json.load(f)


def shutdown_logging(logger_name: str = None):
    """
    Cleans up and shuts down logging handlers for a specific logger or all loggers.
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:  # Copy list to avoid modifying while iterating
            logger.removeHandler(handler)
            if hasattr(handler, "close"):
                try:
                    handler.close()
                except OSError as e:
                    print(f"Warning: Failed to close handler: {e}")
    else:
        logging.shutdown()


# Load the global configuration
global_config = load_global_config()
global_log_level = global_config.get("logging", {}).get("log_level") or None
log_name = global_config.get("logging", {}).get("log_name", "global_logger")
log_file = global_config.get("logging", {}).get("log_file", "logs/app_log.log")

def configure_logger(
    name: str = "global_logger", 
    log_file: str = "tempo_forecasting/logs/pipeline.log", 
    global_log_level: Optional[str] = None # If not provided, files will logged at DEBUG level, and streamhandler at INFO
    ) -> logging.Logger:
    """
    Configures and returns a global logger with separate levels for stream and file handlers.

    Parameters:
        name (str): Name of the logger.
        log_file (str): Path to the log file.
        global_log_level (str): Level of logging granularity. Options: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create or retrieve the logger
    logger = logging.getLogger(name)

    # Default log level for the logger
    logger.setLevel(logging.DEBUG)

    # Map string log levels to actual logging constants
    log_level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Validate global_log_level
    if global_log_level:
        if global_log_level.upper() not in log_level_mapping:
            raise ValueError(f"Invalid log level: {global_log_level}. Valid options are: {', '.join(log_level_mapping.keys())}")
        global_level = log_level_mapping[global_log_level.upper()]
    else:
        global_level = None  # Use defaults for handlers


    # # Ensure the log file directory exists
    # log_dir = os.path.dirname(log_file)
    # if log_dir and not os.path.exists(log_dir):
    #     os.makedirs(log_dir)


    # Avoid adding duplicate handlers
    if not logger.handlers:
        try:
            # # Fix: File handler now only closes at shutdown, not during setup
            # file_handler = logging.FileHandler(log_file, mode="a")
            # file_handler.setLevel(global_level if global_level else logging.DEBUG)
            # file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            # Stream handler for logging in Databricks console
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(global_level if global_level else logging.INFO)
            stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            # logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

            logger.info("Logging setup completed successfully.")

        except Exception as e:
            print(f"Failed to set up logging: {e}")
            raise e

    return logger


# Initialize the logger with custom or default settings depending on required usage
logger = configure_logger(name=log_name, log_file=log_file, global_log_level=None)