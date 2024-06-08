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

try:
    import coloredlogs as coloredlogs  # noqa: F401
except ImportError:
    _has_coloredlogs = False
else:
    _has_coloredlogs = True

import importlib.metadata
import logging
from functools import lru_cache

__version__ = importlib.metadata.version("boario")
__author__ = "sjuhel <pro@sjuhel.org>"

DEBUGFORMATTER = logging.Formatter(
    fmt="%(asctime)s - boario - [%(levelname)s] - [%(filename)s > %(funcName)s() > %(lineno)s] - %(message)s",
    datefmt="%H:%M:%S",
)
"""Debug file formatter."""

INFOFORMATTER = logging.Formatter(
    fmt="%(asctime)s - boario - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)
"""Info file formatter."""


# Create a logger object.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console logger
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(INFOFORMATTER)

# Avoid adding multiple handlers in case of repeated imports
if not logger.handlers:
    logger.addHandler(ch)


# Functions to activate/deactivate logging
def deactivate_logging():
    """Deactivate logging for the package."""
    logger.disabled = True


def activate_logging():
    """Activate logging for the package."""
    logger.disabled = False


# Functions to disable/enable console logging
def disable_console_logging():
    """Disable console logging for the package."""
    logger.info(
        "Disabling logging. You can reenable it with `boario.enable_console_logging()`"
    )
    logger.removeHandler(ch)


def enable_console_logging():
    """Enable console logging for the package."""
    if ch not in logger.handlers:
        logger.addHandler(ch)


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


logger.info(
    "Loaded boario module. You can disable logging in console with `boario.disable_console_logging()`."
)


@lru_cache(10)
def warn_once(logger, msg: str):
    logger.warning(msg)


logger.propagate = False
