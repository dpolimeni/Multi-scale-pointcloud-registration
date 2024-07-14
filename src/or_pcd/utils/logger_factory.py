import inspect
import logging
import os
import sys
from logging import Logger, FileHandler, Formatter, StreamHandler

from or_pcd.utils.constants import __LOG_FOLDER__, __DEFAULT_LOG_FORMAT__


class LoggerFactory:
    """
    Use this class to generate custom loggers!
    """

    def __init__(self):
        pass

    @staticmethod
    def get_logger(
        log_name: str = None,
        log_on_file: bool = False,
        file_name: str = None,
        folder_name: str = None,
        log_file_format: str = None,
        log_console_format: str = None,
        file_handler_level: int = logging.DEBUG,
        console_handler_level: int = logging.DEBUG,
    ) -> Logger:
        """
        Creates customized logger

        :param log_name: name of the logger. If not provided, the file from which the method is called will be used
        :param log_on_file: set this to True if you wish to log on file
        :param file_name: name of the log file. If not provided, the name of the logger will be used
        :param folder_name: name of the folder in which log file is placed. If not provided, __LOG_FOLDER__ is used
        :param log_file_format: log format for file. If None, __DEFAULT_LOG_FORMAT__ will be used
        :param log_console_format: log format for console. If None, __DEFAULT_LOG_FORMAT__ will be used
        :param file_handler_level: only messages above this level will be logged on file
        :param console_handler_level: only messages above this level will be logged on console
        """

        # GET LOGGER NAME (USE CALLER FILE IF NOT PROVIDED)
        _log_name = (
            log_name
            if log_name is not None
            else os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0]
        )

        # INITIALIZE LOGGER
        logger = logging.getLogger(name=_log_name)
        logger.setLevel(level=logging.DEBUG)

        if log_on_file:

            # GET LOG FOLDER NAME (USE DEFAULT IF NOT PROVIDED)
            _folder_name = folder_name if folder_name is not None else __LOG_FOLDER__

            # CREATE DIRECTORY (IF NOT ALREADY AVAILABLE)
            if not os.path.isdir(s=_folder_name):
                os.makedirs(name=_folder_name, exist_ok=True)

            # GET LOG FILE NAME (USE CALLER FILE IF NOT PROVIDED)
            _file_name = file_name if file_name is not None else _log_name

            # CREATE FILE HANDLER
            file_handler = FileHandler(filename=os.path.join(_folder_name, _file_name))

            # SET LEVEL TO HANDLER
            file_handler.setLevel(level=file_handler_level)

            # CREATE AND SET FORMATTER FOR FILE HANDLER
            _log_file_format = (
                log_file_format
                if log_file_format is not None
                else __DEFAULT_LOG_FORMAT__
            )
            file_handler.setFormatter(fmt=Formatter(fmt=_log_file_format))

            # ADD FILE HANDLER TO LOGGER
            logger.addHandler(hdlr=file_handler)

        # INITIALIZE CONSOLE HANDLER
        console_handler = StreamHandler(stream=sys.stdout)
        console_handler.flush = sys.stdout.flush

        # SET LEVEL TO HANDLER
        console_handler.setLevel(level=console_handler_level)

        # CREATE AND SET FORMATTER FOR STREAM HANDLER
        _log_console_format = (
            log_console_format
            if log_console_format is not None
            else __DEFAULT_LOG_FORMAT__
        )
        console_handler.setFormatter(fmt=Formatter(fmt=_log_console_format))

        # ADD CONSOLE HANDLER TO LOGGER
        logger.addHandler(hdlr=console_handler)

        return logger
