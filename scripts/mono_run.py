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
import re
import sys

import pandas as pd
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
        scriptLogger.info("Loading simulation params template from {}".format(params))
        params_template = json.load(f)
    params_template["output_dir"] = output_dir
    params_template["mrio_params_file"] = mrio_params
    #with open(flood_gdp_file) as f:
    #    flood_gdp_share = json.load(f)
    flood_gdp_df = pd.read_parquet(flood_gdp_file)

    with open(event_file) as f:
        scriptLogger.info("Loading event template from {}".format(event_file))
        event_template = json.load(f)

    mrio_path = pathlib.Path(mrios_path)

    with mrio_path.open("rb") as f:
        mrio = pickle.load(f)

    event = event_template.copy()
    sim_params = params_template.copy()
    scriptLogger.info("Setting psi parameter to {}".format(float(psi.replace("_","."))))
    sim_params["psi_param"] = float(psi.replace("_","."))
    sim_params["inventory_restoration_tau"] = inv_tau
    scriptLogger.info("Setting inventory restoration time to {}".format(inv_tau))

    #TODO remove this ?
    value_added = (mrio.x.T - mrio.Z.sum(axis=0))
    value_added = value_added.reindex(sorted(value_added.index), axis=0) #type: ignore
    value_added = value_added.reindex(sorted(value_added.columns), axis=1)
    value_added[value_added < 0] = 0.0
    gdp_df = value_added.groupby("region",axis=1).sum().T["indout"]
    if mrio.unit.unit.unique()[0] != "M.EUR" :
        scriptLogger.warning("MRIO unit appears to not be 'M.EUR'; but {} instead, which is not yet implemented. Contact the dev !".format(mrio.unit.unit.unique()[0]))
    else:
        gdp_df = gdp_df*(10**6)

    #TODO : Finish this
    if stype == "Subregions":
        scriptLogger.info("Subregions run detected !")
        if "sliced" not in str(mrio_path):
            raise ValueError("mrio {} seems not to contain subregions (sliced is not present in its name)".format(str(mrio_path)))
        else:
            mrio_rgxp = re.compile(r"(?P<region>[A-Z]{2})_sliced_in_(?P<split_number>[0-9]+)")
            if (mrio_match := mrio_rgxp.match(mrio_path.stem)) is None:
                raise ValueError("MRIO ({}) is not valid in this context".format(str(mrio_path.stem)))
            else:
                splited_region = mrio_match['region']
                split_number = int(mrio_match['split_number'])
            region_rgxp = re.compile(r"(?P<main_region>[A-Z]{2,3})-(?P<subregion>(?P<sregionname>[A-Z]{2,3})_?(?P<n>\d+)|all|one)")
            if (match := region_rgxp.match(region)) is None:
                raise ValueError("Impacted region ({}) is not valid in this context".format(str(region)))
            else:
                if match['main_region'] != splited_region:
                    raise ValueError("Impacted region ({}) is different from the splited region ({})".format(str(region, splited_region)))
                event["main_region"] = match['main_region']
                if match['subregion'] == "all":
                    event["aff_regions"] = [event["main_region"]+"_"+str(i) for i in range(split_number)]
                elif match['subregion'] == "one":
                    event["aff_regions"] = event["main_region"]+"_1"
                else:
                    event["aff_regions"] = match['sregionname']+"_"+match['n']
    elif stype == "RoW":
        pass
    elif stype== "Full":
        pass
    else:
        raise ValueError("Simulation type {} is incorrect".format(stype))

    scriptLogger.info("Done !")
    scriptLogger.info("Main storage dir is : {}".format(pathlib.Path(params_template["output_dir"]).resolve()))
    if rtype == "int":
        event_row = flood_gdp_df.loc[(flood_gdp_df['class'] == flood_dmg) & (flood_gdp_df['EXIO3_region'] == region)]
        if event_row.empty:
            raise ValueError("This tuple of region / flood class ({},{}) is does not have a representative event (it is likely a duplicate of another class)".format(region,flood_dmg))
        dmg_as_gdp_share = float(event_row['dmg_as_gdp_share'])
        total_direct_dmg = dmg_as_gdp_share * gdp_df[region] #float(event_row['total_dmg'])
        duration = int(event_row['duration'])
        scriptLogger.info("Setting flood duration to {}".format(duration))
        event["duration"] = duration
        event["r_dmg"] = dmg_as_gdp_share
        event["q_dmg"] = total_direct_dmg
    elif rtype == "raw":
        dmg = flood_dmg
        event["r_dmg"] = float(flood_dmg) / float(gdp_df[region])
        scriptLogger.info("Damages represent : {}/{} = {} of the region GDP".format(flood_dmg, gdp_df[region], event['r_dmg']))
        event["q_dmg"] = float(dmg)
    else:
        raise ValueError("Run damage type {} is incorrect".format(rtype))

    event["aff_regions"] = region
    scriptLogger.info("Setting aff_regions to {}".format(region))
    sim_params["output_dir"] = output_dir
    if alt_inv_dur:
        sim_params["results_storage"] = region+"_type_"+stype+"_qdmg_"+rtype+"_"+flood_dmg+"_Psi_"+psi+"_inv_tau_"+str(sim_params["inventory_restoration_tau"])+"_inv_time_"+str(int(alt_inv_dur))
    else:
        sim_params["results_storage"] = region+"_type_"+stype+"_qdmg_"+rtype+"_"+flood_dmg+"_Psi_"+psi+"_inv_tau_"+str(sim_params["inventory_restoration_tau"])
    sim = Simulation(sim_params, mrio_path, modeltype=sim_params['model_type'])
    if alt_inv_dur:
        sim.model.change_inv_duration(alt_inv_dur)
    sim.read_events_from_list([event])
    try:
        scriptLogger.info("Model ready, looping")
        sim.loop(progress=False)
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
