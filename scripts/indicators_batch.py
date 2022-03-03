import os
import re
import sys

from pandas.core.accessor import register_index_accessor
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
import argparse
from glob import glob

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
    "gdp": "unset",
    "type":"unset",
    "psi": "unset",
    "inv_tau": "unset",
    "flood_int":"unset",
    "n_timesteps":"unset",
    "top_5_sector_loss":"unset",
    "has_crashed":"unset"
}]

def batch(wd:pathlib.Path, out):
    subdirs = [d for d in wd.iterdir() if d.is_dir()]
    scriptLogger.info("Subdirs found : {}".format(subdirs))
    for d in subdirs:
        regex = re.compile(r"(?P<region>[A-Z]{2})_(?P<type>RoW|Full)_qdmg_(?:(?P<gdp_dmg>[\d.]+)|(?P<flood_int>[\d.]+%))_Psi_(?P<psi>0_\d+)_inv_tau_(?P<inv_tau>\d+)")
        m = regex.match(d.name)
        if m is None :
            scriptLogger.warning("Directory {} didn't match regex".format(d.name))
        else:
            regex_res = m.groupdict()
            region = regex_res['region']
            if regex_res['gdp_dmg'] is None:
                sce_type = "flood_intensity"
            else:
                sce_type = "gdp_share"
            qdmg = regex_res['gdp_dmg']
            flood_int = regex_res['flood_int']
            psi = regex_res['psi']
            inv_tau = regex_res['inv_tau']
            indic = Indicators.from_folder(d, d/"indexes.json")
            indic.update_indicators()
            res_a = indic.indicators.copy()
            indic.write_indicators()
            res_a['gdp'] = qdmg
            res_a['region'] = region
            res_a['flood_int'] = flood_int
            res_a['psi'] = psi
            res_a['inv_tau'] = inv_tau
            res_a['type'] = sce_type
            res.append(res_a)

    with (out/("indicators_"+datetime.now().strftime("%Y%m%d-%H%M")+".csv")).open('w') as f:
        w = csv.DictWriter(f, res[0].keys())
        w.writeheader()
        w.writerows(res)

if __name__ == '__main__':
    scriptLogger.info('Starting Script')
    args = parser.parse_args()
    wd = pathlib.Path(args.main_folder).resolve()
    if args.output == '../':
        out = pathlib.Path(args.main_folder+'/'+args.output).resolve()
    else:
        out = pathlib.Path(args.output).resolve()
    batch(wd, out)
