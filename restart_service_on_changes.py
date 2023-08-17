import os
import time
import subprocess
import atexit
import signal
import sys
from datetime import datetime
from loguru import logger


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.add(f"logs/logs_{timestamp}.log", rotation="23:59", compression="zip")

process = None  # Global variable to track the subprocess


def start_script():
    """Start the target Python script as a subprocess."""
    return subprocess.Popen(["python3", "chat.py"])


def get_modification_time(path):
    """Get the last modification time of a file or directory."""
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return None


def restart_script(process):
    """Restart the target script by terminating the previous subprocess and starting a new one."""
    if process is not None:
        process.terminate()
    return start_script()


def cleanup_and_exit():
    """Cleanup function to ensure subprocess termination on script exit."""
    global process
    if process is not None:
        process.terminate()


def handle_sigterm(signum, frame):
    """Signal handler for handling SIGTERM signal."""
    cleanup_and_exit()
    sys.exit(0)


if __name__ == "__main__":
    directory_path = "/chat_llm/input_docs"
    logger.info(f"The document directory path: {directory_path}")

    last_modification_time = get_modification_time(directory_path)
    logger.info(f"The last modification time in the directory is: {last_modification_time}")
    process = start_script()

    atexit.register(cleanup_and_exit)  # Register the cleanup function
    signal.signal(signal.SIGTERM, handle_sigterm)  # Handle SIGTERM signal

    try:

        while True:
            current_modification_time = get_modification_time(directory_path)

            if current_modification_time != last_modification_time:
                logger.info(f"Changes detected in '{directory_path}'. Restarting script...")
                process = restart_script(process)
                last_modification_time = current_modification_time

            time.sleep(20)  # Check the changes in the directory every 20 seconds
    except KeyboardInterrupt:
        pass

