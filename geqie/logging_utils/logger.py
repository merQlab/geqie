import logging as logging
from geqie.logging_utils import levels

LOG_FORMAT = "%(levelname)s\t %(asctime)s --- %(message)s (geqie.%(module)s:%(lineno)d)"
LOGGER = None


class GEQIELogger(logging.Logger):
    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(levels.TRACE):
            self._log(levels.TRACE, message, args, **kwargs)

    def state(self, message, *args, **kwargs):
        if self.isEnabledFor(levels.STATE):
            self._log(levels.STATE, message, args, **kwargs)

    def math(self, message, *args, **kwargs):
        if self.isEnabledFor(levels.MATH):
            self._log(levels.MATH, message, args, **kwargs)


def get_logger(name: str) -> GEQIELogger:
    logging.setLoggerClass(GEQIELogger)
    return logging.getLogger(name)


def setup_logger(verbosity_level: int | None = None, reset: bool = False) -> GEQIELogger:
    global LOGGER
    verbosity_level = verbosity_level or levels.ERROR
    
    if LOGGER and not reset:
        return LOGGER
    
    if LOGGER and reset:
        for handler in LOGGER.handlers[:]:
            LOGGER.removeHandler(handler)

    logging.basicConfig(level=logging.CRITICAL)
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = get_logger("geqie")
    logger.addHandler(handler)
    logger.setLevel(verbosity_level)
    logger.propagate = False
    LOGGER = logger
    return logger
