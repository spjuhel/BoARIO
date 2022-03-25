import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)

module_path = os.path.abspath(os.path.join('./'))
if module_path not in sys.path:
    sys.path.append(module_path)

from boario.simulation import Simulation
from boario.indicators import Indicators
import json
import numpy as np
import pathlib
import argparse
import logging

parser = argparse.ArgumentParser(description="Make a 'quick' run of the model to check if it runs correctly")
parser.add_argument('exio_path', type=str, help='The str path to the exio3 pickle file')
parser.add_argument('params_file', type=str, help='The str path to the params to use')
parser.add_argument('event_file', type=str, help='The str path to the event to simulate')
parser.add_argument('save_path', type=str, help='The str path to save the results')

args = parser.parse_args()

VA_idx = np.array(['Taxes less subsidies on products purchased: Total',
       'Other net taxes on production',
       "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
       "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
       "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
       'Operating surplus: Consumption of fixed capital',
       'Operating surplus: Rents on land',
       'Operating surplus: Royalties on resources',
       'Operating surplus: Remaining net operating surplus'], dtype=object)

def query_yes_no(question, default="no"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")

def quick_test(params, event_template, mrio_path):
    event = event_template.copy()
    sim_params = params.copy()
    model = Simulation(sim_params, mrio_path)
    model.read_events_from_list([event])
    model.loop()
    indic = Indicators.from_storage_path(pathlib.Path(sim_params['storage_dir']), params=sim_params)
    indic.update_indicators()
    print(indic.indicators)

if __name__ == '__main__':
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(name)s %(message)s", datefmt="%H:%M:%S")
    rootLogger = logging.getLogger(__name__)
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    args = parser.parse_args()
    with pathlib.Path(args.params_file).open('r') as f:
        params = json.load(f)
    with pathlib.Path(args.event_file).open('r') as f:
        event_template = json.load(f)
    mrio_path = pathlib.Path(args.exio_path)
    params['storage_dir'] = args.save_path
    q = """You are about to run the ARIO3 model with the following inputs :
        - EXIOBASE 3 pickle file : {0}
        - Model parameters :
            Storage directory : {1}
            Steps to simulate : {2}
            MRIO params file  : {3}
        - Event file : {4}

Do you wish to proceed ?
    """.format(mrio_path.resolve(),
               pathlib.Path(params['storage_dir']).resolve(),
               params['n_timesteps'],
               params['mrio_params_file'],
               args.event_file
              )
    hans = query_yes_no(q,"no")
    if hans:
        quick_test(params, event_template, mrio_path)

    else:
        print("Aborting")
        exit()
