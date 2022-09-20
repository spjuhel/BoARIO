import os
import sys
import pathlib
import pandas as pd
import pymrio as pym
import pickle as pkl
import logging
import argparse
import json
import re
from pymrio.core.mriosystem import IOSystem

SEC_AGG_ODS_FILENAME = "exiobase3_aggregate_to_7_sectors.ods"
PARAMS_ODS_FILENAME ="exiobase3_7_sectors_params.ods"
EXIO3_MONETARY = 1000000
MAIN_INVENTORY_DURATION = 90
PARAMS = {
    # The directory to use to store results (relative to output_dir)
    "results_storage": "results",
    # This tells the model to register the evolution of the stocks
    # of every industry (the file can be quite large (2Gbytes+ for
    # a 365 days simulation with exiobase))
    "register_stocks": True,
    # Parameters of the model (we detail these in the documentation)
   "psi_param": 0.85,
   "order_type": "alt",
   # Time division of a year in the model (365 == day, 52 == week, ...)
   "year_to_temporal_unit_factor": 365,
    # Number of day|week|... of one step of the model (ie time sampling)
   "temporal_units_by_step": 1,
    # Charateristic time of inventory restoration
   "inventory_restoration_tau": 60,
    # Base overproduction factor
   "alpha_base": 1.0,
    # Maximum overproduction factor
   "alpha_max": 1.25,
    # Charateristic time of overproduction
   "alpha_tau": 365,
    # Charateristic time of rebuilding
   "rebuild_tau": 60,
    # Number of day|week|... to simulate
   "n_temporal_units_to_sim": 700,
    # Unused
   "min_duration": 700
}

def lexico_reindex(mrio: pym.IOSystem) -> pym.IOSystem:
    """Reindex IOSystem lexicographicaly

    Sort indexes and columns of the dataframe of a :ref:`pymrio.IOSystem` by
    lexical order.

    Parameters
    ----------
    mrio : pym.IOSystem
        The IOSystem to sort

    Returns
    -------
    pym.IOSystem
        The sorted IOSystem

    """

    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.index), axis=0)
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.columns), axis=1)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0) #type: ignore
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)

    return mrio

def full_mrio_pickle(exio3, save_path=None):
    scriptLogger.info("Removing IOSystem attributes deemed unnecessary")
    attr = ['Z', 'Y', 'x', 'A', 'L', 'unit', 'population', 'meta', '__non_agg_attributes__', '__coefficients__', '__basic__']
    tmp = list(exio3.__dict__.keys())
    for at in tmp:
        if at not in attr:
            delattr(exio3,at)
    assert isinstance(exio3, IOSystem)
    scriptLogger.info("Done")
    scriptLogger.info("Computing the missing IO components")
    exio3.calc_all()
    scriptLogger.info("Done")
    scriptLogger.info("Reindexing the dataframes lexicographicaly")
    exio3 = lexico_reindex(exio3)
    scriptLogger.info("Done")
    scriptLogger.info("Saving Full mrio pickle file to {}".format(pathlib.Path(save_path).absolute()))
    exio3 = lexico_reindex(exio3)
    with open(save_path, 'wb') as f:
        pkl.dump(exio3, f)

def aggreg_mrio_pickle(full_exio_path, sector_aggregator_path, save_path=None):
    exio_path = pathlib.Path(full_exio_path)
    if not exio_path.exists():
        raise FileNotFoundError("Exiobase file not found - {}".format(exio_path))
    with exio_path.open('rb') as f:
        scriptLogger.info("Loading EXIOBASE3 from {}".format(exio_path.resolve()))
        exio3 = pkl.load(f)
    assert isinstance(exio3, IOSystem)
    sec_agg_vec = pd.read_excel(sector_aggregator_path, sheet_name="aggreg_input", engine="odf")
    sec_agg_newnames = pd.read_excel(sector_aggregator_path, sheet_name="name_input", engine="odf", index_col=0, squeeze=True)
    sec_agg_vec = sec_agg_vec.sort_values(by="sector")
    scriptLogger.info("Reading aggregation matrix from sheet 'input' in file {}".format(pathlib.Path(sector_aggregator_path).absolute()))
    scriptLogger.info("Aggregating from {} to {} sectors".format(len(exio3.get_sectors()), len(sec_agg_vec.group.unique()))) #type:ignore
    sec_agg_vec['new_sectors'] = sec_agg_vec.group.map(sec_agg_newnames.to_dict())
    exio3.aggregate(sector_agg=sec_agg_vec.new_sectors.values)
    exio3.calc_all()
    scriptLogger.info("Done")
    scriptLogger.info("Saving to {}".format(pathlib.Path(save_path).absolute()))
    exio3 = lexico_reindex(exio3)
    with open(save_path, 'wb') as f:
        pkl.dump(exio3, f)

def params_from_ods(ods_file,monetary,main_inv_dur):
    mrio_params = {}
    mrio_params["monetary_unit"] = monetary
    mrio_params["main_inv_dur"] = main_inv_dur
    df = pd.read_excel(ods_file) #type: ignore
    mrio_params["capital_ratio_dict"] = df[["Aggregated version sector", "Capital to VA ratio"]].set_index("Aggregated version sector").to_dict()['Capital to VA ratio']
    mrio_params["inventories_dict"] = df[["Aggregated version sector", "Inventory size (days)"]].set_index("Aggregated version sector").to_dict()['Inventory size (days)']
    return mrio_params

def event_tmpl_from_ods(ods_file):
    event_params = {}
    event_params["aff_regions"] = ["FR"]
    event_params["dmg_distrib_regions"] = [1]
    event_params["dmg_distrib_sectors_type"] = "gdp"
    event_params["dmg_distrib_sectors"] = []
    event_params["duration"] = 5
    event_params["name"] = "Test-event"
    event_params["occur"] = 7
    event_params["q_dmg"] = 1000000
    df = pd.read_excel(ods_file) #type: ignore
    event_params["aff_sectors"] = df.loc[(df.Affected=="yes"),"Aggregated version sector"].to_list()
    event_params["rebuilding_sectors"] = df.loc[(df["Rebuilding factor"] > 0),["Aggregated version sector", "Rebuilding factor"]].set_index("Aggregated version sector").to_dict()['Rebuilding factor']
    return event_params

parser = argparse.ArgumentParser(description="Build a minimal example for BoARIO, from EXIOBASE3 MRIO table zip file")
parser.add_argument('source_path', type=str, help='The str path to the directory with source materials')
parser.add_argument('-o', "--output", type=str, help='The path to the example directory to create', nargs='?', default='./testing-directory/')

args = parser.parse_args()
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("EXIOBASE3_Minimal_example_generator")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False

if __name__ == '__main__':
    args = parser.parse_args()
    scriptLogger.info("Make sure you use the same python environment when you use the minimal example as now.")
    scriptLogger.info("Your current environment is: {}".format(sys.executable))

    sec_agg_ods = pathlib.Path(args.source_path)/SEC_AGG_ODS_FILENAME
    params_ods = pathlib.Path(args.source_path)/PARAMS_ODS_FILENAME
    output_dir = pathlib.Path(args.output)
    # Create full mrio pickle file

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    full_exio_pickle_name = "exiobase3_full.pkl"
    minimal_exio_name = "exiobase3_minimal.pkl"
    params_file_name = "params.json"
    mrio_params_file_name = "mrio_params.json"
    event_file_name = "event.json"
    scriptLogger.info("This will create the following directory, with all required files for the minimal example : {}".format(output_dir.resolve()))

    if not sec_agg_ods.exists():
        raise FileNotFoundError("Sector aggregator ods file not found - {}".format(sec_agg_ods))

    if not params_ods.exists():
        raise FileNotFoundError("Params ods file not found - {}".format(params_ods))

    if not (output_dir/full_exio_pickle_name).exists():
        regex = re.compile(r"(IOT_\d\d\d\d_ixi.zip)")
        exio_path = None
        for root, dirs, files in os.walk(args.source_path):
            scriptLogger.info("Looking for Exiobase3 file here {}".format(args.source_path))
            for f in files:
                if regex.match(f):
                    exio_path = (pathlib.Path(args.source_path)/f).resolve()
                    scriptLogger.info("Found Exiobase3 file here {}".format(exio_path))
                    break

        if exio_path is None:
            raise FileNotFoundError("Exiobase file not found in given source directory - {}".format(args.source_path))

        scriptLogger.info("Parsing EXIOBASE3 from {} - Note that this takes a few minutes on a powerful laptop. ".format(exio_path.resolve()))
        exio3 = pym.parse_exiobase3(path=exio_path)
        full_mrio_pickle(exio3, save_path=output_dir/full_exio_pickle_name)

    # create minimal mrio pickle file
    if not (output_dir/minimal_exio_name).exists():
        aggreg_mrio_pickle(output_dir/full_exio_pickle_name, sector_aggregator_path=sec_agg_ods, save_path=output_dir/minimal_exio_name)

    # create params file
    if not (output_dir/params_file_name).exists():
        scriptLogger.info("Generating simulation parameters file : {}".format((output_dir/params_file_name).resolve()))
        params = PARAMS
        params["output_dir"] = str(output_dir.resolve())
        with (output_dir/params_file_name).open("w") as f:
            json.dump(params, f, indent=4)

    # create mrio_params_file
    if not (output_dir/mrio_params_file_name).exists():
        scriptLogger.info("Generating mrio parameters file : {}".format((output_dir/mrio_params_file_name).resolve()))
        mrio_params = params_from_ods(params_ods, EXIO3_MONETARY, MAIN_INVENTORY_DURATION)
        with (output_dir/mrio_params_file_name).open("w") as f:
            json.dump(mrio_params, f, indent=4)

    # create mrio_params_file
    if not (output_dir/event_file_name).exists():
        scriptLogger.info("Generating event file : {}".format((output_dir/event_file_name).resolve()))
        event_params = event_tmpl_from_ods(params_ods)
        with (output_dir/event_file_name).open("w") as f:
            json.dump(event_params, f, indent=4)

    scriptLogger.info("Done !")
