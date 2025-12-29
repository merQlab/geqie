import logging

ERROR = logging.ERROR
WARNING = logging.WARNING
INFO = logging.INFO
DEBUG = logging.DEBUG
MATH = 9
STATE = 8
TRACE = 5

CLI_VERBOSITY_LEVELS = {
    0: ERROR,  # Critical and Error as a default level
    1: WARNING,
    2: INFO,
    3: DEBUG,
    4: MATH,
    5: STATE,
    6: TRACE,
}


def cli_verbosity_to_logging_level(verbosity_level: int) -> int:
    return CLI_VERBOSITY_LEVELS.get(verbosity_level, ERROR)