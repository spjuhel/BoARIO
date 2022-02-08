import logging

DEBUGFORMATTER = logging.Formatter(fmt='%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s',datefmt='%H:%M:%S',)
"""Debug file formatter."""

INFOFORMATTER = logging.Formatter(fmt='%(asctime)s [%(levelname)s] - %(message)s',datefmt='%H:%M:%S',)
"""Log file and stream output formatter."""
