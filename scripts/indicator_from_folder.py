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

import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from boario.indicators import Indicators
import pathlib
import argparse
import logging
import re

parser = argparse.ArgumentParser(description="Produce indicators from one run folder")
parser.add_argument('folder', type=str, help='The str path to the folder')
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("indicators_from_folder")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False

def produce_indicator(folder):
    regex = re.compile(r"(?P<region>[A-Z]{2})_type_(?P<type>RoW|Full)_qdmg_(?:raw_|int_)?(?P<gdp_dmg>(?:[\d_]+%?)|(?:max|min))_Psi_(?P<psi>(?:0_\d+)|1_0)_inv_tau_(?P<inv_tau>\d+)(?:_inv_time_(?:\d+))?")
    m = regex.match(folder.name)
    if m is None:
        scriptLogger.warning("Directory {} didn't match regex".format(folder.name))
    else:
        indic = Indicators.from_folder(folder, folder/"indexes.json")
        indic.update_indicators()
        indic.indicators.copy()
        indic.write_indicators()

if __name__ == '__main__':
    scriptLogger.info('Starting Script')
    args = parser.parse_args()
    folder = pathlib.Path(args.folder).resolve()
    produce_indicator(folder)
