## Enhanced Scheduler Script with Comprehensive Retry Mechanisms

import os
import json
import logging
import requests
import subprocess
import argparse
import yaml
import shutil
import smtplib
from datetime import datetime, timedelta, time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time as time_module
from typing import List, Dict, Optional, Tuple

# Constants
SCHEDULED_TIMES = [
    time(hour=7, minute=55),
    time(hour=13, minute=12),
    time(hour=19, minute=55),
    time(hour=1, minute=12)
]

FOLDER_MAPPING = {
    '00z': (7, 55),
    '06z': (13, 12),
    '12z': (19, 55),
    '18z': (1, 12)
}

EMAIL_RECIPIENT = "anterezic22@gmail.com"

class EmailConfig:
    """Configuration for email notifications."""
    SENDER_EMAIL = "azureforecast@gmail.com"
    PASSWORD = os.environ.get('EMAIL_PASSWORD', "rasc qmyy yqlo jopc")  # Use environment variable for security
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587

def send_email(subject: str, body: str, recipient: str) -> bool:
    """
    Send an email with the specified subject and body to the recipient.

    Args:
        subject (str): Email subject.
        body (str): Email body.
        recipient (str): Recipient email address.

    Returns:
        bool: True if email was sent successfully, False otherwise.
    """
    message = MIMEMultipart()
    message["From"] = EmailConfig.SENDER_EMAIL
    message["To"] = recipient
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(EmailConfig.SMTP_SERVER, EmailConfig.SMTP_PORT) as server:
            server.starttls()
            server.login(EmailConfig.SENDER_EMAIL, EmailConfig.PASSWORD)
            server.send_message(message)
        return True
    except Exception as e:
        logging.error(f"Failed to send email: {e}")
        return False

def setup_logging(log_dir: str) -> logging.Logger:
    """
    Set up logging with both file and console handlers.

    Args:
        log_dir (str): Directory where log files will be stored.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'master.log')

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Logger setup
    logger = logging.getLogger('master')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def check_folders(date: str, logger: logging.Logger, base_url: str = "https://data.ecmwf.int/forecasts/") -> List[str]:
    """
    Check available folders for a given date.

    Args:
        date (str): Date in YYYYMMDD format.
        logger (logging.Logger): Logger instance for logging messages.
        base_url (str): Base URL to check for folders.

    Returns:
        List[str]: List of available folders.
    """
    url = f"{base_url}{date}/"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        available_folders = ["00z", "06z", "12z", "18z"]
        return [folder for folder in available_folders if folder in response.text]
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to access URL: {url}. Error: {e}")
        return []

def load_state(state_file: str) -> Dict:
    """
    Load the state from a JSON file.

    Args:
        state_file (str): Path to the state file.

    Returns:
        Dict: Loaded state data.
    """
    try:
        if os.path.exists(state_file):
            with open(state_file, 'r') as file:
                return json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to load state file: {e}")
    return {}

def save_state(state: Dict, state_file: str) -> None:
    """
    Save the current state to a JSON file.

    Args:
        state (Dict): State data to save.
        state_file (str): Path to the state file.
    """
    try:
        with open(state_file, 'w') as file:
            json.dump(state, file)
    except Exception as e:
        logging.error(f"Failed to save state: {e}")

def count_files_in_subfolder(folder_path: str, subfolder_name: str) -> int:
    """
    Count the number of files in a specified subfolder.

    Args:
        folder_path (str): Path to the main folder.
        subfolder_name (str): Name of the subfolder.

    Returns:
        int: Number of files in the subfolder.
    """
    subfolder_path = os.path.join(folder_path, subfolder_name)
    if not os.path.exists(subfolder_path):
        return 0
    return len([f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))])

def get_next_run_time(current_utc_time: datetime) -> datetime:
    """
    Calculate the next scheduled run time based on current UTC time.

    Args:
        current_utc_time (datetime): Current UTC datetime.

    Returns:
        datetime: Next scheduled run datetime.
    """
    sorted_times = sorted(SCHEDULED_TIMES)
    today = current_utc_time.date()
    current_time = current_utc_time.time()

    for scheduled_time in sorted_times:
        if scheduled_time > current_time:
            return datetime.combine(today, scheduled_time)

    # If no remaining times today, schedule for the first time tomorrow
    return datetime.combine(today + timedelta(days=1), sorted_times[0])

def get_date_and_run(current_utc_time: datetime) -> Tuple[str, str]:
    """
    Determine the date to query and the corresponding EC run based on current UTC time.

    Args:
        current_utc_time (datetime): Current UTC datetime.

    Returns:
        Tuple[str, str]: Date in YYYYMMDD format and EC run identifier.
    """
    current_minutes = current_utc_time.hour * 60 + current_utc_time.minute

    # Define time boundaries in minutes since midnight
    time_boundaries = {
        'boundary1': 1 * 60 + 12,    # 01:12
        'boundary2': 7 * 60 + 55,    # 07:55
        'boundary3': 13 * 60 + 12,   # 13:12
        'boundary4': 19 * 60 + 55    # 19:55
    }

    today = current_utc_time.strftime('%Y%m%d')
    yesterday = (current_utc_time - timedelta(days=1)).strftime('%Y%m%d')

    if time_boundaries['boundary1'] <= current_minutes < time_boundaries['boundary2']:
        return yesterday, '18z'
    elif time_boundaries['boundary2'] <= current_minutes < time_boundaries['boundary3']:
        return today, '00z'
    elif time_boundaries['boundary3'] <= current_minutes < time_boundaries['boundary4']:
        return today, '06z'
    elif current_minutes >= time_boundaries['boundary4']:
        return today, '12z'
    else:  # current_minutes < boundary1
        return yesterday, '12z'

def get_missing_runs(state: Dict, current_utc_time: datetime, n_runs: int) -> List[Tuple[str, str]]:
    """
    Get a list of missing date-run pairs from the last N runs.

    Args:
        state (Dict): Current state dictionary
        current_utc_time (datetime): Current UTC datetime
        n_runs (int): Number of runs to check

    Returns:
        List[Tuple[str, str]]: List of missing date-run pairs
    """
    # Get the date and run we should be at based on current time
    current_date, current_run = get_date_and_run(current_utc_time)

    # All possible runs
    all_runs = ['00z', '06z', '12z', '18z']

    # Generate all expected date-run pairs for the last n_runs
    expected_pairs = []
    current_dt = datetime.strptime(current_date, '%Y%m%d')

    # Map runs to hours for comparison
    run_hours = {'00z': 0, '06z': 6, '12z': 12, '18z': 18}
    current_run_hour = run_hours[current_run]

    days_to_check = (n_runs // 4) + 1  # Number of days to look back

    for days_back in range(days_to_check):
        check_date = (current_dt - timedelta(days=days_back)).strftime('%Y%m%d')

        for run in all_runs:
            # For the current date, only include runs up to the current run
            if days_back == 0:
                if run_hours[run] > current_run_hour:
                    continue

            expected_pairs.append((check_date, run))

    # Take only the last n_runs pairs
    expected_pairs = expected_pairs[:n_runs]

    # Find missing pairs
    missing_pairs = []
    for date, run in expected_pairs:
        if date not in state or run not in state[date]:
            missing_pairs.append((date, run))

    return missing_pairs

def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration data.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

    # Validate required fields
    required_fields = [
        'log_dir', 'n_steps', 'sql_checker_script_path', 'data_download_script_path',
        'download_dir_path', 'preprocess_script_path', 'temp_preprocessing_folder_path',
        'wimdy_inference_path', 'plot_script_path', 'data_injection_script_path',
        'scaler_path', 'models_path', "plot_injection_script_path",
        'data_output_path', 'plot_output_path',
        # Added retry configurations
        'max_download_retries', 'initial_retry_delay',
        'max_subprocess_retries', 'initial_subprocess_delay',
        'max_global_retries', 'global_retry_delay'
    ]

    missing_fields = [field for field in required_fields if field not in config_data]
    if missing_fields:
        raise KeyError(f"Missing required config fields: {', '.join(missing_fields)}")

    return config_data

def run_subprocess(script_path: str, args: List[str], logger: logging.Logger, description: str) -> bool:
    """
    Run a subprocess with the given script and arguments.

    Args:
        script_path (str): Path to the Python script to execute.
        args (List[str]): List of arguments to pass to the script.
        logger (logging.Logger): Logger instance for logging messages.
        description (str): Description of the subprocess for logging.

    Returns:
        bool: True if subprocess executed successfully, False otherwise.
    """
    try:
        logger.info(f"Running {description} ({script_path}) with arguments: {' '.join(args)}")
        subprocess.run(["python3", script_path] + args, check=True)
        logger.info(f"{description} executed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run {description}: {e}")
        return False

def perform_cleanup(paths: List[str], logger: logging.Logger) -> None:
    """
    Clean up specified directories by deleting their contents.

    Args:
        paths (List[str]): List of directory paths to clean.
        logger (logging.Logger): Logger instance for logging messages.
    """
    for path in paths:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                logger.info(f"Deleted directory: {path}")
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

def retry_operation(max_retries: int, initial_delay: int, operation, *args, **kwargs) -> bool:
    """
    Attempt to perform an operation with retries and exponential backoff.

    Args:
        max_retries (int): Maximum number of retry attempts.
        initial_delay (int): Initial delay between retries in seconds.
        operation (callable): The operation to perform.
        *args: Arguments to pass to the operation.
        **kwargs: Keyword arguments to pass to the operation.

    Returns:
        bool: True if operation succeeds, False otherwise.
    """
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        success = operation(*args, **kwargs)
        if success:
            return True
        else:
            logging.warning(f"Attempt {attempt} failed. Retrying in {delay} seconds...")
            time_module.sleep(delay)
            delay *= 2  # Exponential backoff
    return False

def global_retry(logger: logging.Logger, config: dict, current_pipeline_state: Dict, state_file: str) -> bool:
    """
    Perform a global retry by cleaning temporary folders and restarting the pipeline.

    Args:
        logger (logging.Logger): Logger instance for logging messages.
        config (dict): Configuration data.
        current_pipeline_state (Dict): Current state before retry.
        state_file (str): Path to the state file.

    Returns:
        bool: True if global retry succeeds, False otherwise.
    """
    logger.info("Initiating global retry: Cleaning temporary folders and restarting pipeline.")

    # Clean temporary folders
    temp_folders = [
        config['download_dir_path'],
        config['temp_preprocessing_folder_path'],
        config['plot_output_path'],
        config['data_output_path']
    ]
    perform_cleanup(temp_folders, logger)

    # Reset state
    save_state({}, state_file)
    current_pipeline_state.clear()

    logger.info("Temporary folders cleaned and state reset. Restarting pipeline.")

    return True  # Indicate that global retry has been initiated

def main(config: dict):
    """
    Main function to orchestrate the forecast processing system.

    Args:
        config (dict): Configuration data.
    """
    logger = setup_logging(config['log_dir'])

    # Configuration Parameters
    max_download_retries = config.get('max_download_retries', 5)
    initial_retry_delay = config.get('initial_retry_delay', 10)  # in seconds

    max_subprocess_retries = config.get('max_subprocess_retries', 3)
    initial_subprocess_delay = config.get('initial_subprocess_delay', 5)  # in seconds

    max_global_retries = config.get('max_global_retries', 1)
    global_retry_delay = config.get('global_retry_delay', 20)  # in seconds

    global_retry_count = 0

    temp_folders = [
        config['download_dir_path'],
        config['temp_preprocessing_folder_path'],
        config['plot_output_path'],
        config['data_output_path']
    ]
    perform_cleanup(temp_folders, logger)

    while True:
        current_utc_time = datetime.utcnow()

        run_subprocess(
            config['sql_checker_script_path'],
            ["-o", config['log_dir'], "--n", "10"],
            logger,
            "SQL Checker"
        )

        state_file = os.path.join(config['log_dir'], 'state.json')
        state = load_state(state_file)

        missing_runs = get_missing_runs(state, current_utc_time, 8)

        if missing_runs:
            logger.info(f"Missing runs detected: {missing_runs}")
            for date, ec_run in missing_runs:
                logger.info(f"Processing missing run: Date={date}, Run={ec_run}")

                def download_data():
                    available_folders = check_folders(date, logger)
                    if ec_run in available_folders:
                        success = run_subprocess(
                            config['data_download_script_path'],
                            ["--n_steps", str(config['n_steps']),
                             "--output_dir", config['download_dir_path'],
                             "--log_dir", config['log_dir'],
                             "--ec_date", date,
                             "--ec_run", ec_run],
                            logger,
                            "Raw Data Download"
                        )
                        if success:
                            logger.info(f"Successfully downloaded data for {ec_run} on {date}.")
                            # Update state
                            if date not in state:
                                state[date] = []
                            state[date].append(ec_run)
                            save_state(state, state_file)
                        return success
                    else:
                        logger.warning(f"Folder '{ec_run}' not available for date {date}.")
                        return False

                # Attempt to download data with retries
                download_success = retry_operation(
                    max_retries=max_download_retries,
                    initial_delay=initial_retry_delay,
                    operation=download_data
                )

                if not download_success:
                    logger.error(f"Failed to download data for {ec_run} on {date} after {max_download_retries} retries.")
                    send_email(
                        subject="ERROR - Python Forecast Scheduler",
                        body=f"Failed to download data for run '{ec_run}' on date '{date}' after {max_download_retries} retries.",
                        recipient=EMAIL_RECIPIENT
                    )
                    # Proceed to global retry
                    if global_retry_count < max_global_retries:
                        global_retry_count += 1
                        retry_success = global_retry(logger, config, state, state_file)
                        if retry_success:
                            time_module.sleep(global_retry_delay)
                            break  # Restart the while loop
                    else:
                        logger.error(f"Exceeded maximum global retries ({max_global_retries}). Terminating scheduler.")
                        send_email(
                            subject="CRITICAL ERROR - Python Forecast Scheduler",
                            body=f"Scheduler has failed after {max_global_retries} global retries.",
                            recipient=EMAIL_RECIPIENT
                        )
                        exit(1)
                else:
                    # Reset global retry count upon successful operation
                    global_retry_count = 0

                # Additional processing can be added here (e.g., preprocessing, inference, etc.)
                # Implementing retries for each subprocess step

                subprocess_steps = [
                    {
                        'script': config['preprocess_script_path'],
                        'args': ["--data_dir", config['download_dir_path'],
                                 "--temp_dir", config['temp_preprocessing_folder_path'],
                                 "--output_dir", config['data_output_path'],
                                 "--logs_dir", config['log_dir']],
                        'description': 'Preprocessing'
                    },
                    {
                        'script': config['wimdy_inference_path'],
                        'args': ["--data_input_dir", config['temp_preprocessing_folder_path'],
                                 "--scaler_dir", config['scaler_path'],
                                 "--model_dir", config['models_path'],
                                 "--data_output_dir", config['data_output_path'],
                                 "--n_inference_steps_ifs", str(count_files_in_subfolder(config['download_dir_path'], 'ifs')),
                                 "--n_inference_steps_aifs", str(count_files_in_subfolder(config['download_dir_path'], 'aifs')),
                                 "--logs_dir", config['log_dir']],
                        'description': 'WIMDY Inference'
                    },
                    {
                        'script': config['plot_script_path'],
                        'args': ["--ec_run", ec_run,
                                 "--output_dir", config['plot_output_path'],
                                 "--logs_dir", config['log_dir'],
                                 "--input_dir", config['data_output_path']],
                        'description': 'Plot Generation'
                    }#,
                    # {
                    #     'script': config['data_injection_script_path'],
                    #     'args': ["--ec_run", ec_run,
                    #              "--ec_publish_time", f"{date} {FOLDER_MAPPING.get(ec_run, (0,0))[0]:02}:{FOLDER_MAPPING.get(ec_run, (0,0))[1]:02}",
                    #              "--plots_dir", config['plot_output_path'],
                    #              "--data_dir", config['data_output_path']],
                    #     'description': 'Data Injection'
                    # },
                    # {
                    #     'script': config['plot_injection_script_path'],
                    #     'args': ["--ec_run", ec_run,
                    #              "--ec_publish_time", f"{date} {FOLDER_MAPPING.get(ec_run, (0,0))[0]:02}:{FOLDER_MAPPING.get(ec_run, (0,0))[1]:02}",
                    #              "--input_dir", config['plot_output_path']],
                    #     'description': 'Plot Injection'
                    # }
                ]

                for step in subprocess_steps:
                    step_success = retry_operation(
                        max_retries=max_subprocess_retries,
                        initial_delay=initial_subprocess_delay,
                        operation=lambda: run_subprocess(
                            step['script'],
                            step['args'],
                            logger,
                            step['description']
                        )
                    )

                    if not step_success:
                        logger.error(f"Failed to execute {step['description']} after {max_subprocess_retries} retries.")
                        send_email(
                            subject=f"ERROR - {step['description']} Failed",
                            body=f"The {step['description']} step failed for run '{ec_run}' on date '{date}' after {max_subprocess_retries} retries.",
                            recipient=EMAIL_RECIPIENT
                        )
                        # Proceed to global retry
                        if global_retry_count < max_global_retries:
                            global_retry_count += 1
                            retry_success = global_retry(logger, config, state, state_file)
                            if retry_success:
                                time_module.sleep(global_retry_delay)
                                break  # Restart the while loop
                        else:
                            logger.error(f"Exceeded maximum global retries ({max_global_retries}). Terminating scheduler.")
                            send_email(
                                subject="CRITICAL ERROR - Python Forecast Scheduler",
                                body=f"Scheduler has failed after {max_global_retries} global retries.",
                                recipient=EMAIL_RECIPIENT
                            )
                            exit(1)
                    else:
                        # Reset global retry count upon successful step
                        global_retry_count = 0

            else:
                logger.info("No missing runs detected at this time.")

            # Calculate next run time
            next_run_time = get_next_run_time(datetime.utcnow())
            sleep_duration = (next_run_time - datetime.utcnow()).total_seconds()

            # Ensure positive sleep duration
            if sleep_duration <= 0:
                sleep_duration = 1  # Minimum sleep duration

            logger.info(f"Sleeping for {sleep_duration} seconds until next run at {next_run_time}.")
            time_module.sleep(sleep_duration)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast Processing System")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    try:
        configuration = load_config(args.config)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"Configuration Error: {e}")
        exit(1)

    main(configuration)