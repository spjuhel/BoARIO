import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np

# import the different classes
import boario
from boario.model_base import ARIOBaseModel
from boario.simulation import Simulation  # Simulation wraps the model
from boario.event import (
    Event,
    EventKapitalRebuild,
    EventArbitraryProd,
    EventKapitalRecover,
)  # A class defining a shock on capital


@pytest.fixture
def test_mrio():
    mrio = pymrio.load_test()  # .calc_all()
    mrio.aggregate(
        region_agg=["reg1", "reg1", "reg2", "reg2", "reg3", "reg3"],
        sector_agg=[
            "food",
            "mining",
            "manufactoring",
            "other",
            "construction",
            "other",
            "other",
            "other",
        ],
    )
    mrio.calc_all()
    return mrio


@pytest.fixture
def test_model(test_mrio):
    model = ARIOBaseModel(test_mrio)
    return model


@pytest.fixture
def test_sim(test_model):
    sim = Simulation(test_model)
    return sim


def test_Event_abstract(test_sim):
    with pytest.raises(TypeError):
        ev = Event(
            impact=100000, aff_regions=["reg1"], aff_sectors=["manufactoring", "mining"]
        )


def test_EventKapitalRecover_init_empty():
    with pytest.raises(TypeError):
        ev = EventKapitalRecover()

    with pytest.raises(TypeError):
        ev = EventKapitalRecover(impact=1)


def test_EventKapitalRecover_init_before_model():
    with pytest.raises(AttributeError):
        ev = EventKapitalRecover(
            impact=100000,
            aff_regions=["reg1"],
            aff_sectors=["manufactoring", "mining"],
            recovery_time=30,
        )


def test_EventKapitalRecover_init_before_sim_after_model(test_model):
    with pytest.raises(AttributeError):
        ev = EventKapitalRecover(impact=1)


def test_EventKapitalRecover_wrong_init_after_sim_after_model(test_sim):
    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=1)

    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=pd.Series())

    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=pd.DataFrame())

    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=[])


@pytest.mark.parametrize(
    "impact,regions,sectors,recovery_time",
    [
        pytest.param(-1, "reg1", "manufactoring", 30, id="neg impact"),
        pytest.param(0, "reg1", "manufactoring", 30, id="null impact"),
    ],
)
def test_EventKapitalRecover_incorrect_inputs(
    test_sim, impact, regions, sectors, recovery_time
):
    with pytest.raises(ValueError):
        ev = EventKapitalRecover(
            impact=impact,
            aff_regions=regions,
            aff_sectors=sectors,
            recovery_time=recovery_time,
        )
