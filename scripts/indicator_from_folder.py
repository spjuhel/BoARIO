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
    regex = re.compile(r"(?P<region>[A-Z]{2})_type_(?P<type>RoW|Full)_qdmg_(?P<gdp_dmg>(?:[\d_]+%)|(?:max|min))_Psi_(?P<psi>0_\d+)_inv_tau_(?P<inv_tau>\d+)")
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
