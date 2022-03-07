import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

import json
import pandas as pd
import numpy as np
import pathlib
import csv
import argparse
import logging
import re

parser = argparse.ArgumentParser(description="Produce csv from json indicators")
parser.add_argument('folder', type=str, help='The str path to the main folder')
parser.add_argument('indicator_file', type=str, help='The name of the indicator file to load (indicators.json or prod_indicators.json)')
parser.add_argument('-o', "--output", type=str, help='Path where to save csv')
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("indicators_batch")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False

def produce_csv(folder,save_path,indicator_file):
    future_df = []
    for ind in folder.glob('**/'+indicator_file):
        with ind.open('r') as f:
            dico = json.load(f)
        dico['run_name'] = ind.parent.name
        if isinstance(dico['region'],list) and len(dico['region'])==1:
            dico['region'] = dico['region'][0]
        future_df.append(dico)
    pd.DataFrame(future_df).to_csv(save_path)



if __name__ == '__main__':
    scriptLogger.info('Starting Script')
    args = parser.parse_args()
    folder = pathlib.Path(args.folder).resolve()
    produce_csv(folder,save_path=args.output,indicator_file=args.indicator_file)
