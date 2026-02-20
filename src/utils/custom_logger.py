import logging

CMD_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger() -> logging.Logger:
    """
    Configure common logger with console handler, file logger, level and format.
    :return: configured logger
    """
    log_level = logging.DEBUG
    custom_logger = logging.getLogger(__name__)
    custom_logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    formatter = logging.Formatter(CMD_LOG_FORMAT)
    console_handler.setFormatter(formatter)

    fh = logging.FileHandler("training.log")
    fh.setLevel(logging.DEBUG)

    custom_logger.addHandler(fh)
    custom_logger.addHandler(console_handler)
    return custom_logger


logger = get_logger()
