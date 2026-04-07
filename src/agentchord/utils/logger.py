import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import warnings

LOG_PATH = "logs"

class Logger:
    logger_names = list()
    handlers = dict()

    def __init__(self, name: str, file_name: str = ""):
        if name in self.logger_names:
            warnings.warn(f"Logger {name} is already registered!")
        else:
            self.logger_names.append(name)
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        )
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.file_path = os.path.join(LOG_PATH, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") if not file_name else os.path.join(LOG_PATH, file_name)
        if self.file_path in self.handlers:
            self.file_handler = self.handlers[self.file_path]
        else:
            self.file_handler = RotatingFileHandler(self.file_path, maxBytes=10485760, backupCount=5)
            self.handlers[self.file_path] = self.file_handler
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)
        self.set_debug_level(False)

    def set_log_redirection(self, file_name: str):
        self.logger.removeHandler(self.file_handler)
        file_path = self.file_path if not file_name else os.path.join(LOG_PATH, file_name)
        self.file_handler = RotatingFileHandler(file_path, maxBytes=10485760, backupCount=5)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

    def set_debug_level(self, debug: bool):
        if debug:
            self.stream_handler.setLevel(logging.DEBUG)
        else:
            self.stream_handler.setLevel(logging.INFO)

    def debug(self, message: str):
        self.logger.debug(message)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)