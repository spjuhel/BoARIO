import logging

from ario3.logging_conf import INFOFORMATTER

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# defines the stream handler
_ch = logging.StreamHandler()  # creates the handler
_ch.setLevel(logging.INFO)  # sets the handler info
_ch.setFormatter(logging.Formatter(INFOFORMATTER))  # sets the handler formatting

# adds the handler to the global variable: log
logger.addHandler(_ch)
