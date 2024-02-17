import logging


class CustomFormatter(logging.Formatter):
    """Custom formatter to add color to log levels."""

    # ANSI escape codes for various colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    # Assign different colors to different log levels
    LOG_COLORS = {
        logging.ERROR: RED,
        logging.WARNING: YELLOW,
        logging.INFO: GREEN,
        logging.DEBUG: RESET,
    }

    def format(self, record):
        color = self.LOG_COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_global_logger():
    """Set up the global logging configuration with a custom formatter."""
    formatter = CustomFormatter("%(levelname)s: %(message)s")

    # Configure the root logger
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
