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

import json
import pandas as pd
import pathlib
import argparse
import logging

parser = argparse.ArgumentParser(description="Produce csv from json indicators")
parser.add_argument("folder", type=str, help="The str path to the main folder")
# parser.add_argument('indicator_file', type=str, help='The name of the indicator file to load (indicators.json or prod_indicators.json)')
parser.add_argument("-o", "--output", type=str, help="Path where to save csv")


def produce_general_csv(folder, save_path):
    future_df = []
    for ind in folder.glob("**/indicators.json"):
        with ind.open("r") as f:
            dico = json.load(f)

        dico["run_name"] = ind.parent.name
        if isinstance(dico["region"], list) and len(dico["region"]) == 1:
            dico["region"] = dico["region"][0]

        future_df.append(dico)

    pd.DataFrame(future_df).to_csv(save_path)


def produce_region_prod_loss_csv(folder, save_path):
    future_df = None
    for ind in folder.glob("**/prod_chg.json"):
        if "RoW" in ind.parent.name:
            pass
        else:
            with ind.open("r") as f:
                dico = json.load(f)

            df = pd.DataFrame(dico)
            if future_df is None:
                future_df = df.copy()
            else:
                future_df = pd.concat([future_df, df])

    future_df.to_csv(save_path)


def produce_region_fd_loss_csv(folder, save_path):
    future_df = None
    for ind in folder.glob("**/fd_loss.json"):
        if "RoW" in ind.parent.name:
            pass
        else:
            with ind.open("r") as f:
                dico = json.load(f)

            df = pd.DataFrame(dico)
            if future_df is None:
                future_df = df.copy()
            else:
                future_df = pd.concat([future_df, df])

    future_df.to_csv(save_path)


if __name__ == "__main__":
    args = parser.parse_args()
    folder = pathlib.Path(args.folder).resolve()
    logFormatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S"
    )
    scriptLogger = logging.getLogger("indicators_batch")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)

    scriptLogger.addHandler(consoleHandler)
    scriptLogger.setLevel(logging.INFO)
    scriptLogger.propagate = False
    scriptLogger.info("Starting Script")
    produce_general_csv(folder, save_path=args.output + "general.csv")
    produce_region_prod_loss_csv(folder, save_path=args.output + "prodloss.csv")
    produce_region_fd_loss_csv(folder, save_path=args.output + "fdloss.csv")
