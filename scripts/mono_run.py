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
print(os.getcwd())
module_path = os.path.abspath(os.path.join("../"))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join("./"))
if module_path not in sys.path:
    sys.path.append(module_path)

from boario.simulation import Simulation
import json
import pathlib
import logging
import pickle
import argparse

parser = argparse.ArgumentParser(description="Produce indicators from one run folder")
parser.add_argument("region", type=str, help="The region to run")
parser.add_argument("params", type=str, help="The params file")
parser.add_argument("psi", type=str, help="The psi parameter")
parser.add_argument("inv_tau", type=str, help="The inventory restoration parameter")
parser.add_argument("stype", type=str, help="The type (RoW or Full) simulation to run")
parser.add_argument("rtype", type=str, help="The damage type (raw or int)")
parser.add_argument("flood_dmg", type=str, help="The flood damage/intensity to run")
parser.add_argument("mrios_path", type=str, help="The mrios path")
parser.add_argument("output_dir", type=str, help="The output directory")
parser.add_argument("flood_gdp_file", type=str, help="The share of gdp impacted according to flood distribution file")
parser.add_argument("event_file", type=str, help="The event template file")
parser.add_argument("mrio_params", type=str, help="The mrio parameters file")
parser.add_argument("alt_inv_dur", type=str, help="The optional alternative main inventory duration", nargs="?", default=None)

def run(region, params, psi, inv_tau, stype, rtype, flood_dmg, mrios_path, output_dir, flood_gdp_file, event_file, mrio_params, alt_inv_dur=None):
    with open(params) as f:
        params_template = json.load(f)
    params_template["output_dir"] = output_dir
    params_template["mrio_params_file"] = mrio_params
    with open(flood_gdp_file) as f:
        flood_gdp_share = json.load(f)

    with open(event_file) as f:
        event_template = json.load(f)

    if stype == "RoW":
        mrio_path = list(pathlib.Path(mrios_path).glob("mrio_"+region+"*.pkl"))
        scriptLogger.info("Trying to load {}".format(mrio_path))
        assert len(mrio_path)==1
        mrio_path = list(mrio_path)[0]
    else:
        mrio_path = pathlib.Path(mrios_path)/"mrio_full.pkl"

    with mrio_path.open("rb") as f:
        mrio = pickle.load(f)

    event = event_template.copy()
    sim_params = params_template.copy()
    sim_params["psi_param"] = float(psi.replace("_","."))
    sim_params["inventory_restoration_time"] = inv_tau
    value_added = (mrio.x.T - mrio.Z.sum(axis=0))
    value_added = value_added.reindex(sorted(value_added.index), axis=0) #type: ignore
    value_added = value_added.reindex(sorted(value_added.columns), axis=1)
    value_added[value_added < 0] = 0.0
    gdp_df = value_added.groupby("region",axis=1).sum().T["indout"]
    # TODO : Change this hard coded value !
    gdp_df = gdp_df*1000000
    del mrio
    scriptLogger.info("Done !")
    scriptLogger.info("Main storage dir is : {}".format(pathlib.Path(params_template["output_dir"]).resolve()))
    if rtype == "int":
        v = flood_gdp_share[region][flood_dmg]
        dmg = gdp_df[region] * v
        event["r_dmg"] = v
        event["q_dmg"] = dmg
    elif rtype == "raw":
        dmg = flood_dmg
        event["r_dmg"] = flood_dmg / gdp_df
        event["q_dmg"] = dmg

    event["aff-regions"] = region
    sim_params["output_dir"] = output_dir
    if alt_inv_dur:
        sim_params["results_storage"] = region+"_type_"+stype+"_qdmg_"+rtype+"_"+flood_dmg+"_Psi_"+psi+"_inv_tau_"+str(sim_params["inventory_restoration_time"])+"_inv_time_"+str(int(alt_inv_dur))
    else:
        sim_params["results_storage"] = region+"_type_"+stype+"_qdmg_"+rtype+"_"+flood_dmg+"_Psi_"+psi+"_inv_tau_"+str(sim_params["inventory_restoration_time"])
    model = Simulation(sim_params, mrio_path)
    if alt_inv_dur:
        model.mrio.change_inv_duration(alt_inv_dur)
    model.read_events_from_list([event])
    try:
        scriptLogger.info("Model ready, looping")
        model.loop(progress=False)
    except Exception:
        scriptLogger.exception("There was a problem:")

if __name__ == "__main__":
    args = parser.parse_args()
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
    scriptLogger = logging.getLogger("generic_run - {}_{}_{}_{}_{}_{}".format(args.region, args.psi, args.inv_tau, args.stype, args.rtype, args.flood_dmg))
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    scriptLogger.addHandler(consoleHandler)
    scriptLogger.setLevel(logging.INFO)
    scriptLogger.propagate = False

    scriptLogger.info("=============== STARTING RUN ================")
    run(args.region, args.params, args.psi, int(args.inv_tau), args.stype, args.rtype, args.flood_dmg, args.mrios_path, args.output_dir, args.flood_gdp_file, args.event_file, args.mrio_params, args.alt_inv_dur)
