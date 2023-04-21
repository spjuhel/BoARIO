import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np

# import the different classes
import boario
from boario.simulation import Simulation         # Simulation wraps the model
from boario.extended_models import ARIOPsiModel  # The core of the model
from boario.indicators import Indicators         # A class computing and storing indicators based on a simulation
from boario.event import EventKapitalRebuild, EventArbitraryProd, EventKapitalRecover  # A class defining a shock on capital
from boario.utils.recovery_functions import *


def minimal_simulation():
    # Load the IOSystem from pymrio
    mrio = pymrio.load_test()#.calc_all()
    mrio.aggregate(region_agg= ["reg1", "reg1", "reg2", "reg2", "reg3", "reg3"], sector_agg=["food","mining","manufactoring","other","construction","other","other","other"])
    mrio.calc_all()
    # Instantiate the model and the simulation
    model = ARIOPsiModel(mrio)
    sim = Simulation(model)
    sim.loop()
    raise SystemExit(0)

def test_minimal_simulation():
    with pytest.raises(SystemExit):
        minimal_simulation()
