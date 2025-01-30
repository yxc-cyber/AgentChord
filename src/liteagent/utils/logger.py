import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

LOG_PATH = "logs"

class Logger:
    logger_names = list()

    def __init__(self, name: str, file_name: str = ""):
        if name in Logger.logger_names:
            raise Exception(f"Logger {name} is already registered!")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
        )
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        file_path = os.path.join(LOG_PATH, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") if not file_name else os.path.join(LOG_PATH, file_name)
        self.file_handler = RotatingFileHandler(file_path, maxBytes=10485760, backupCount=5)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.stream_handler)
        self.logger.addHandler(self.file_handler)

    def set_log_redirection(self, file_name: str):
        self.logger.removeHandler(self.file_handler)
        self.file_handler = RotatingFileHandler(file_name, maxBytes=10485760, backupCount=5)
        self.file_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.file_handler)

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