import logging

from rich.logging import RichHandler


def setup_logger() -> logging.Logger:
    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%x %H:%M:%S]",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger("rich")
    logger.setLevel(logging.INFO)
    return logger
