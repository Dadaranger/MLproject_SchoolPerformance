# Import required libraries
import logging  # For logging functionality
import os      # For operating system related operations
from datetime import datetime  # For timestamp generation

# Create a log file name with current timestamp (format: month_day_year_hour_minute_second.log)
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create the full path for the logs directory
# The logs will be stored in a 'logs' subdirectory of the current working directory
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)

# Create the logs directory if it doesn't exist
# exist_ok=True prevents error if directory already exists
os.makedirs(logs_path,exist_ok=True)

# Define the complete path for the log file
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

# Configure the logging system with the following settings:
logging.basicConfig(
    filename=LOG_FILE_PATH,    # Specify the log file location
    # Define log message format:
    # [timestamp] line_number logger_name - log_level - log_message
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,        # Set logging level to INFO (captures INFO, WARNING, ERROR, CRITICAL)
)

if __name__ == "__main__":
    logging.info("Logging system initialized.")