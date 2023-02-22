# BoARIO : The Adaptative Regional Input Output model in python.
# Copyright (C) 2022  Samuel Juhel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import coloredlogs
import logging
from functools import lru_cache

__version__ = "v0.4.1b0"

# Create a logger object.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEBUGFORMATTER = logging.Formatter(fmt='%(asctime)s [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s',datefmt='%H:%M:%S',)
"""Debug file formatter."""

INFOFORMATTER = logging.Formatter(fmt='%(asctime)s [%(levelname)s] - %(message)s',datefmt='%H:%M:%S',)


@lru_cache(10)
def warn_once(logger, msg: str):
    logger.warning(msg)

"""Log file and stream output formatter."""

#logger.setLevel(logging.DEBUG)
#print("c")
#print(logger.handlers)

# defines the stream handler
#_ch = logging.StreamHandler()  # creates the handler
#_ch.setLevel(logging.DEBUG)  # sets the handler info
#_ch.setFormatter(DEBUGFORMATTER)  # sets the handler formatting

# adds the handler to the global variable: log
#logger.addHandler(_ch)
logger.propagate = False
#print(logger.handlers)
#print("a")
