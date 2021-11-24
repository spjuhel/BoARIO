import sys
import logging
from logging.config import dictConfig

logging_config = dict(
    version=1,
    formatters={
        'verbose': {
            'format': ("[%(asctime)s] %(levelname)s "
                       "[%(name)s:%(lineno)s] %(message)s"),
            'datefmt': "%d/%b/%Y %H:%M:%S",
        },
        'simple': {
            'format': ("[%(asctime)s] %(levelname)s "
                       "[%(name)s:%(lineno)s] %(message)s"),
        },
    },
    handlers={
        'debug-logger': {'class': 'logging.handlers.RotatingFileHandler',
                           'formatter': 'verbose',
                           'level': logging.DEBUG,
                           'filename': 'logs/execution_debug.log',
                           'maxBytes': 52428800,
                           'backupCount': 7},
        'run-logger': {'class': 'logging.handlers.RotatingFileHandler',
                             'formatter': 'verbose',
                             'level': logging.INFO,
                             'filename': 'logs/batch.log',
                             'maxBytes': 52428800,
                             'backupCount': 7},
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': sys.stdout,
        },
    },
    loggers={
        'debug-logger': {
            'handlers': ['debug-logger'],
            'level': logging.DEBUG
        },
        'run-logger': {
            'handlers': ['run-logger', 'console'],
            'level': logging.INFO
        }
    }
)

dictConfig(logging_config)

debug_logger = logging.getLogger('debug-logger')
run_logger = logging.getLogger('run-logger')
