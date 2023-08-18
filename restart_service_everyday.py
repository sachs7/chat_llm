import os
import time
import subprocess
import atexit
import signal
import sys
from datetime import datetime, timedelta
from loguru import logger


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.add(f"logs/logs_{timestamp}.log", rotation="23:59", compression="zip")

process = None  # Global variable to track the subprocess


def start_script():
    """Start the target Python script as a subprocess."""
    return subprocess.Popen(["python3", "resqchat.py"])


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
    logger.info("Staring the script...")
    process = start_script()

    atexit.register(cleanup_and_exit)  # Register the cleanup function
    signal.signal(signal.SIGTERM, handle_sigterm)  # Handle SIGTERM signal

    try:

        while True:
            now = datetime.now()
            next_restart_time = datetime(
                now.year, now.month, now.day, 23, 59
            ) + timedelta(days=1)
            time_until_restart = (next_restart_time - now).total_seconds()

            logger.info(f"Next restart at: {next_restart_time}")
            time.sleep(time_until_restart)

            logger.info("Restarting script...")
            process = restart_script(process)
    except KeyboardInterrupt:
        pass
