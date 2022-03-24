import pathlib
import pymrio as pym
import pandas as pd
import pickle as pkl
import logging
import datetime
import argparse
import json
import country_converter as coco

parser = argparse.ArgumentParser(description='Aggregate an exio3 MRIO sectors')
parser.add_argument('exio_path', type=str, help='The str path to the exio3 zip file')
parser.add_argument('aggreg_path', type=str, help='The str path to the ods aggregation matrix file')
parser.add_argument('sector_names_json_path', type=str, help='The str path to the json file with the new names of the sectors')
parser.add_argument('-r', "--region_agg", type=str, help='The str path to the json file with the regions aggregation', nargs='?', default=None)
parser.add_argument('-o', "--output", type=str, help='The str path to save the pickled mrio to', nargs='?', default='./mrio_dump')

args = parser.parse_args()
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
scriptLogger = logging.getLogger("aggreg_exio3")
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)

scriptLogger.addHandler(consoleHandler)
scriptLogger.setLevel(logging.INFO)
scriptLogger.propagate = False

agg_dicts_default = [{'aggregates': {'MX': 'MX'}, 'missing': 'Rest of the world'},
 {'aggregates': {'NO': 'NO'}, 'missing': 'Rest of the world'},
 {'aggregates': {'NL': 'NL'}, 'missing': 'Rest of the world'},
 {'aggregates': {'RU': 'RU'}, 'missing': 'Rest of the world'},
 {'aggregates': {'EE': 'EE'}, 'missing': 'Rest of the world'},
 {'aggregates': {'ID': 'ID'}, 'missing': 'Rest of the world'},
 {'aggregates': {'FR': 'FR'}, 'missing': 'Rest of the world'},
 {'aggregates': {'SE': 'SE'}, 'missing': 'Rest of the world'},
 {'aggregates': {'CN': 'CN'}, 'missing': 'Rest of the world'},
 {'aggregates': {'HU': 'HU'}, 'missing': 'Rest of the world'},
 {'aggregates': {'PT': 'PT'}, 'missing': 'Rest of the world'},
 {'aggregates': {'LT': 'LT'}, 'missing': 'Rest of the world'},
 {'aggregates': {'LV': 'LV'}, 'missing': 'Rest of the world'},
 {'aggregates': {'BR': 'BR'}, 'missing': 'Rest of the world'},
 {'aggregates': {'HR': 'HR'}, 'missing': 'Rest of the world'},
 {'aggregates': {'SK': 'SK'}, 'missing': 'Rest of the world'},
 {'aggregates': {'JP': 'JP'}, 'missing': 'Rest of the world'},
 {'aggregates': {'US': 'US'}, 'missing': 'Rest of the world'},
 {'aggregates': {'ZA': 'ZA'}, 'missing': 'Rest of the world'},
 {'aggregates': {'CH': 'CH'}, 'missing': 'Rest of the world'},
 {'aggregates': {'BE': 'BE'}, 'missing': 'Rest of the world'},
 {'aggregates': {'PL': 'PL'}, 'missing': 'Rest of the world'},
 {'aggregates': {'CA': 'CA'}, 'missing': 'Rest of the world'},
 {'aggregates': {'CZ': 'CZ'}, 'missing': 'Rest of the world'},
 {'aggregates': {'RO': 'RO'}, 'missing': 'Rest of the world'},
 {'aggregates': {'IT': 'IT'}, 'missing': 'Rest of the world'},
 {'aggregates': {'TR': 'TR'}, 'missing': 'Rest of the world'},
 {'aggregates': {'GB': 'GB'}, 'missing': 'Rest of the world'},
 {'aggregates': {'SI': 'SI'}, 'missing': 'Rest of the world'},
 {'aggregates': {'GR': 'GR'}, 'missing': 'Rest of the world'},
 {'aggregates': {'FI': 'FI'}, 'missing': 'Rest of the world'},
 {'aggregates': {'AU': 'AU'}, 'missing': 'Rest of the world'},
 {'aggregates': {'DE': 'DE'}, 'missing': 'Rest of the world'},
 {'aggregates': {'KR': 'KR'}, 'missing': 'Rest of the world'},
 {'aggregates': {'IE': 'IE'}, 'missing': 'Rest of the world'},
 {'aggregates': {'ES': 'ES'}, 'missing': 'Rest of the world'},
 {'aggregates': {'BG': 'BG'}, 'missing': 'Rest of the world'},
 {'aggregates': {'IN': 'IN'}, 'missing': 'Rest of the world'},
 {'aggregates': {'AT': 'AT'}, 'missing': 'Rest of the world'}]

def aggreg(exio_path, sector_aggregator_path, new_sectors_name_path, regions_aggregator_dicts, save_path=None):
    with pathlib.Path(new_sectors_name_path).open('r') as f:
        a = json.load(f)
    sec_agg_matrix = pd.read_excel(sector_aggregator_path, sheet_name="input", engine="odf", header=None).to_numpy()
    scriptLogger.info("Parsing exiobase3 from {}".format(pathlib.Path(exio_path).resolve()))
    exio3 = pym.parse_exiobase3(path=exio_path)
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
    for dico in regions_aggregator_dicts:
        exio3_c = exio3.copy()
        regions_aggregator = coco.agg_conc(original_countries=exio3.get_regions(),
                                           aggregates=dico['aggregates'],
                                           missing_countries=dico['missing'])
        exio3_c.aggregate(region_agg=regions_aggregator)
        exio3_c.calc_all()
        name = save_path+list(dico['aggregates'].keys())[0]+"_"+datetime.datetime.now().strftime("%m-%d-%H%M")+".pkl"
        scriptLogger.info("Saving to {}".format(pathlib.Path(name).resolve()))
        with open(name, 'wb') as f:
            pkl.dump(exio3_c, f)

if __name__ == '__main__':
    args = parser.parse_args()
    name = pathlib.Path(args.exio_path).stem
    if args.region_agg is None:
        agg_dicts = agg_dicts_default
    else :
        with pathlib.Path(args.region_agg).open('r') as f:
            agg_dicts = json.load(f)
    aggreg(args.exio_path, args.aggreg_path, args.sector_names_json_path, regions_aggregator_dicts=agg_dicts, save_path=args.output)
