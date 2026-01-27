import logging as logging

ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
MATH = 9
STATE = 8
TRACE = 5

LOGGING_LEVELS_MAPPING = {
    "ERROR": ERROR,
    "WARNING": WARNING,
    "INFO": INFO,
    "DEBUG": DEBUG,
    "MATH": MATH,
    "STATE": STATE,
    "TRACE": TRACE,
}

CLI_VERBOSITY_LEVELS = {
    0: ERROR,  # Critical and Error as a default level
    1: WARNING,
    2: INFO,
    3: DEBUG,
    4: MATH,
    5: STATE,
    6: TRACE,
}


def cli_verbosity_to_logging_level(verbosity_level: int | str) -> int:
    if isinstance(verbosity_level, str):
        return LOGGING_LEVELS_MAPPING.get(verbosity_level.upper(), ERROR)
    return CLI_VERBOSITY_LEVELS.get(verbosity_level, ERROR)