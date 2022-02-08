import coloredlogs, logging
from ario3.logging_conf import INFOFORMATTER, DEBUGFORMATTER

# Create a logger object.
logger = logging.getLogger(__name__)

fieldstyle = {'asctime': {'color': 'green'},
              'levelname': {'bold': True, 'color': 'black'},
              'filename':{'color':'cyan'},
              'funcName':{'color':'blue'}}

levelstyles = {'critical': {'bold': True, 'color': 'red'},
               'debug': {'color': 'green'},
               'error': {'color': 'red'},
               'info': {'color':'magenta'},
               'warning': {'color': 'yellow'}}

coloredlogs.install(level=logging.DEBUG,
                    logger=logger,
                    fmt='%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s',
                    datefmt='%H:%M:%S',
                    field_styles=fieldstyle,
                    level_styles=levelstyles)

logger.setLevel(logging.DEBUG)

# defines the stream handler
_ch = logging.StreamHandler()  # creates the handler
_ch.setLevel(logging.INFO)  # sets the handler info
_ch.setFormatter(DEBUGFORMATTER)  # sets the handler formatting

# adds the handler to the global variable: log
logger.addHandler(_ch)
