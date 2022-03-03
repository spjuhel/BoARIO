import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from ario3.simulation import Simulation
from ario3.indicators import Indicators
from ario3.logging_conf import DEBUGFORMATTER
import json
import pandas as pd
import numpy as np
import pathlib
import csv
import logging
import coloredlogs
import numpy as np
import pickle
from datetime import datetime

parser = argparse.ArgumentParser(description='Aggregate an exio3 MRIO sectors')
parser.add_argument('main_folder', type=str, help='A folder which subfolders are results of run')
parser.add_argument('-o', "--output", type=str, help='The str path to save the pickled mrio to', nargs='?', default='../')
args = parser.parse_args()
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("indicators_batch")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False


res = [{
            "tot_fd_unmet": "unset",
            "aff_fd_unmet": "unset",
            "rebuild_durations": "unset",
            "shortage_b": False,
            "shortage_date_start": "unset",
            "shortage_date_end": "unset",
            "shortage_date_max": "unset",
            "shortage_ind_max": "unset",
            "shortage_ind_mean": "unset",
            "10_first_shortages": "unset",
            "prod_gain_tot": "unset",
            "prod_lost_tot": "unset",
            "prod_gain_unaff": "unset",
            "prod_lost_unaff": "unset",
            "region":"unset",
            "q_dmg":"unset",
            "pib": "unset",
            "psi": "unset",
            "inv_tau": "unset",
            "flood_int":"unset",
            "n_timesteps":"unset",
            "has_crashed":"unset"
            }]

for v in dmgs:
         qdmg = str(round(v, 3))
         dir = (list(wd.glob(region+'_RoW_qdmg_'+qdmg+'*_inv_tau_6'))[0])
         indic = Indicators.from_folder(dir, dir/"indexes.json")
         indic.update_indicators()
         res_a = indic.indicators.copy()
         indic.write_indicators()
         res_a['pib'] = v
         res_a['region'] = region
         res_a['flood_int'] = qdmg
         res_a['psi'] = 0.85
         res_a['inv_tau'] = 6
         res.append(res_a)

with (wd/".."/("indicators_"+datetime.now().strftime("%Y%m%d-%H%M")+".csv")).open('w') as f:
        w = csv.DictWriter(f, res[0].keys())
        w.writeheader()
        w.writerows(res)

if __name__ == '__main__':
    args = parser.parse_args()
    wd = pathlib.Path(args.main_folder).resolve()
    out = pathlib.Path(args.output).resolve()
    
