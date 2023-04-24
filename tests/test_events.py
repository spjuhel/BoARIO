import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np

# import the different classes
import boario
from boario.model_base import ARIOBaseModel
from boario.simulation import Simulation         # Simulation wraps the model
from boario.event import Event, EventKapitalRebuild, EventArbitraryProd, EventKapitalRecover  # A class defining a shock on capital


@pytest.fixture
def test_mrio():
    mrio = pymrio.load_test()#.calc_all()
    mrio.aggregate(region_agg= ["reg1", "reg1", "reg2", "reg2", "reg3", "reg3"], sector_agg=["food","mining","manufactoring","other","construction","other","other","other"])
    mrio.calc_all()
    return mrio

@pytest.fixture
def test_model(test_mrio):
    model = ARIOPsiModel(test_mrio)
    return model

@pytest.fixture
def test_sim(test_model):
    sim = Simulation(test_model)
    return sim

def test_Event_init():
