#import ario3.mriosystem as mrio_sys
import pathlib
import pymrio as pym
import pandas as pd
import pickle as pkl
import logging
import sys
import datetime
import argparse
import json

parser = argparse.ArgumentParser(description='Aggregate an exio3 MRIO sectors')
parser.add_argument('exio_path', type=str, help='The str path to the exio3 zip file')
parser.add_argument('aggreg_path', type=str, help='The str path to the ods aggregation matrix file')
parser.add_argument('sector_names_json_path', type=str, help='The str path to the json file with the new names of the sectors')
parser.add_argument('save_path', type=str, help='The str path to save the pickled mrio to', nargs='?', default='./mrio_dump')

args = parser.parse_args()

def aggreg(exio_path, aggreg_path, sectors_name_path, save_path=None):
    rootLogger.info("Parsing exiobase3 from {}".format(pathlib.Path(exio_path).absolute()))
    exio3 = pym.parse_exiobase3(path=exio_path)
    rootLogger.info("Done")
    rootLogger.info("Reading aggregation matrix from sheet 'input' in file {}".format(pathlib.Path(aggreg_path).absolute()))
    sec_agg_matrix = pd.read_excel(aggreg_path, sheet_name="input", engine="odf", header=None).to_numpy()
    rootLogger.info("Aggregating from {} to {} sectors".format(len(exio3.get_sectors()), sec_agg_matrix.shape[0]))
    exio3.aggregate(sector_agg=sec_agg_matrix)
    rootLogger.info("Done")
    rootLogger.info("Renaming sectors from {}".format(pathlib.Path(sectors_name_path).absolute()))
    with pathlib.Path(sectors_name_path).open('r') as f:
        a = json.load(f)
    exio3.rename_sectors(a)
    rootLogger.info("Done")
    rootLogger.info("Computing the IO components")
    exio3.calc_all()
    rootLogger.info("Done")
    name = save_path+"_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".pkl"
    rootLogger.info("Saving to {}".format(pathlib.Path(name).absolute()))
    with open(name, 'wb') as f:
        pkl.dump(exio3, f)
        
if __name__ == '__main__':  
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
    rootLogger = logging.getLogger(__name__)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    args = parser.parse_args()
    aggreg(args.exio_path, args.aggreg_path, args.sector_names_json_path, args.save_path)