import pathlib
import pymrio as pym
import pandas as pd
import pickle as pkl
import logging
import argparse
import json
import country_converter as coco
from pymrio.core.mriosystem import IOSystem

parser = argparse.ArgumentParser(description='Aggregate an EXIOBASE3 MRIO table in less sectors (an optionally less regions)')
parser.add_argument('exio_path', type=str, help='The str path to the exio3 zip file')
parser.add_argument('aggreg_path', type=str, help='The str path to the ods aggregation matrix file')
parser.add_argument('sector_names_json_path', type=str, help='The str path to the json file with the new names of the sectors')
parser.add_argument('regions_aggregator_json_path', type=str, help='The str path to the json file with the regions aggregation', nargs='?', default=None)
parser.add_argument('-o', "--output", type=str, help='The str path to save the pickled mrio to', nargs='?', default='./mrio_dump')

args = parser.parse_args()
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("aggreg_exio3")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False

def aggreg(exio_path, sector_aggregator_path, new_sectors_name_path, regions_aggregator_json_path=None, save_path=None):
    with pathlib.Path(new_sectors_name_path).open('r') as f:
        a = json.load(f)
    if regions_aggregator_json_path is not None:
        with pathlib.Path(regions_aggregator_json_path).open('r') as f:
            b = json.load(f)
    sec_agg_matrix = pd.read_excel(sector_aggregator_path, sheet_name="input", engine="odf", header=None).to_numpy()
    scriptLogger.info("Parsing exiobase3 from {}".format(pathlib.Path(exio_path).resolve()))
    exio3 = pym.parse_exiobase3(path=exio_path)
    # gain some diskspace and RAM by removing unused attributes
    attr = ['Z', 'Y', 'x', 'A', 'L', 'unit', 'population', 'meta', '__non_agg_attributes__', '__coefficients__', '__basic__']
    tmp = list(exio3.__dict__.keys())
    for at in tmp:
        if at not in attr:
            delattr(exio3,at)
    assert isinstance(exio3, IOSystem)
    scriptLogger.info("Done")
    scriptLogger.info("Computing the IO components")
    exio3.calc_all()
    scriptLogger.info("Done")
    scriptLogger.info("Reading aggregation matrix from sheet 'input' in file {}".format(pathlib.Path(sector_aggregator_path).absolute()))
    scriptLogger.info("Aggregating from {} to {} sectors".format(len(exio3.get_sectors()), sec_agg_matrix.shape[0]))
    exio3.aggregate(sector_agg=sec_agg_matrix)
    scriptLogger.info("Done")
    scriptLogger.info("Renaming sectors from {}".format(pathlib.Path(new_sectors_name_path).absolute()))
    exio3.rename_sectors(a)
    scriptLogger.info("Done")
    if regions_aggregator_json_path is not None:
        regions_aggregator = coco.agg_conc(original_countries=exio3.get_regions(),
                                           aggregates=b['aggregates'],
                                           missing_countries=b['missing'])
        exio3.aggregate(region_agg=regions_aggregator)
    exio3.calc_all()
    name = save_path
    scriptLogger.info("Saving to {}".format(pathlib.Path(name).absolute()))

    with open(name, 'wb') as f:
        pkl.dump(exio3, f)

if __name__ == '__main__':
    args = parser.parse_args()
    name = pathlib.Path(args.exio_path).stem
    aggreg(args.exio_path, args.aggreg_path, args.sector_names_json_path, regions_aggregator_json_path=args.regions_aggregator_json_path, save_path=args.output)
