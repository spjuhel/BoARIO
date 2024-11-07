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

"""BoARIO : The Adaptative Regional Input Output model in python."""

import importlib.metadata
import logging

__version__ = importlib.metadata.version("boario")
__author__ = "sjuhel <pro@sjuhel.org>"

DEBUG_TRACE = False
DEBUGFORMATTER = logging.Formatter(
    fmt="%(asctime)s - [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s",
    datefmt="%H:%M:%S",
)
"""Debug file formatter."""

# INFOFORMATTER = logging.Formatter(
#     fmt="%(asctime)s - [%(levelname)s] - %(message)s",
#     datefmt="%H:%M:%S",
# )
# """Info file formatter."""

# Create a logger object.
logger = logging.getLogger(__name__)

try:
    import pygit2

    try:
        __git_branch__ = pygit2.Repository(__file__).head.name
        logger.info("You are using boario from branch %s", __git_branch__)
    except pygit2.GitError:
        logger.info(
            "Could not find git branch, this is normal if you installed boario from pip/conda."
        )
except ModuleNotFoundError:
    logger.info("Unable to tell git branch as pygit2 was not found.")
