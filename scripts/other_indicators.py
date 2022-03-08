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

parser = argparse.ArgumentParser(description="Produce json indicators from feather df")
parser.add_argument('folder', type=str, help='The str path to the folder')
parser.add_argument('-o', "--output", type=str, help='Path where to save json')
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("indicators_other")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False

def prod_change_region(df):
    df2=df[df.variable=="production"].set_index(['step','region','sector']).drop(['variable'],axis=1).unstack([1,2])
    df3=df[df.variable=="rebuild_production"].set_index(['step','region','sector']).drop(['variable'],axis=1).unstack([1,2])
    df4 = df2-df3
    df4.columns=df4.columns.droplevel(0)
    prod_chg = df4 - df4.iloc[1,:]
    prod_chg = prod_chg.round(6)
    return prod_chg.sum().groupby('region').sum()

def fd_loss_region(df_loss):
    df2 = df_loss.set_index(['step','region','fd_cat']).unstack([1,2])
    df2 = df2.round(6)
    return df2.sum().groupby('region').sum()

def produce_json(folder,save_path):
    if isinstance(folder,str):
        folder = pathlib.Path(folder)
    assert isinstance(folder,pathlib.Path)
    df = pd.read_feather(folder/"treated_df.feather")
    df_loss = pd.read_feather(folder/"treated_df_loss.feather")
    # Production loss
    prod_chg_region = prod_change_region(df)
    prod_chg_region = pd.DataFrame({folder.name:prod_chg_region}).T
    prod_chg_region.to_json(save_path+"/prod_chg.json")
    # FD loss
    fd_loss_region_df = fd_loss_region(df_loss)
    fd_loss_region_df = pd.DataFrame({folder.name:fd_loss_region_df}).T
    fd_loss_region_df.to_json(save_path+"/fd_loss.json")


if __name__ == '__main__':
    scriptLogger.info('Starting Script')
    args = parser.parse_args()
    folder = pathlib.Path(args.folder).resolve()
    produce_json(folder,save_path=args.output)
