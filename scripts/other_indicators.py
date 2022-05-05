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

import pandas as pd
import pathlib
import argparse
import logging

parser = argparse.ArgumentParser(description="Produce json indicators from parquet df")
parser.add_argument('folder', type=str, help='The str path to the folder')
parser.add_argument('-o', "--output", type=str, help='Path where to save json')

def prod_change_region(df):
    df2=df[df.variable=="production"].set_index(['step','region','sector']).drop(['variable'],axis=1).unstack([1,2])
    #df3=df[df.variable=="rebuild_production"].set_index(['step','region','sector']).drop(['variable'],axis=1).unstack([1,2])
    #df4 = df2-df3
    df2.columns=df2.columns.droplevel(0)
    prod_chg = df2 - df2.iloc[0,:]
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
    df = pd.read_parquet(folder/"treated_df.parquet")
    df_loss = pd.read_parquet(folder/"treated_df_loss.parquet")
    # Production loss
    prod_chg_region = prod_change_region(df)
    prod_chg_region = pd.DataFrame({folder.name:prod_chg_region}).T
    prod_chg_region.to_json(save_path+"/prod_chg.json", indent=4)
    # FD loss
    fd_loss_region_df = fd_loss_region(df_loss)
    fd_loss_region_df = pd.DataFrame({folder.name:fd_loss_region_df}).T
    fd_loss_region_df.to_json(save_path+"/fd_loss.json", indent=4)


if __name__ == '__main__':
    args = parser.parse_args()
    folder = pathlib.Path(args.folder).resolve()
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
    scriptLogger = logging.getLogger("other_indicator - "+folder.name)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    scriptLogger.addHandler(consoleHandler)
    scriptLogger.setLevel(logging.INFO)
    scriptLogger.propagate = False
    scriptLogger.info('Starting Script')
    produce_json(folder,save_path=args.output)
