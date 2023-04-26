import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np

from numpy.core._exceptions import UFuncTypeError

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
        ev = EventKapitalRecover(
            impact=1,
            aff_regions=["reg1"],
            aff_sectors=["manufactoring", "mining"],
            recovery_time=30,
        )

def test_Event_abstract(test_sim):
    with pytest.raises(TypeError):
        ev = Event(
            impact=100000, aff_regions=["reg1"], aff_sectors=["manufactoring", "mining"]
        )

def test_EventKapitalRecover_wrong_init_after_sim_after_model(test_sim):
    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=1, recovery_time=30)

    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=pd.Series(dtype="float64"), recovery_time=30)

    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=pd.DataFrame(dtype="float64"), recovery_time=30)

    with pytest.raises(ValueError):
        ev = EventKapitalRecover(impact=[], recovery_time=30)


@pytest.mark.parametrize(
    "impact",
    [
        0,
        -1,
        [],
        pd.Series([],dtype="float64"),
        pd.DataFrame([],dtype="float64"),
        np.array([]),
        np.array([1, -1]),
        "str",
    ],
    ids=[
        "null",
        "scal_neg",
        "empty_l",
        "empty_sr",
        "empty_df",
        "empty_np",
        "neg_np",
        "str",
    ],
)
def test_EventKapitalRecover_incorrect_impact(test_sim, impact):
    with pytest.raises((ValueError, UFuncTypeError)):
        ev = EventKapitalRecover(
            impact=impact,
            aff_regions="reg1",
            aff_sectors="manufactoring",
            recovery_time=30,
        )


@pytest.mark.parametrize(
    "regions",
    [
        "non_existing",
        ["non_existing"],
        1,
        [1],
        [],
        pd.DataFrame([],dtype="float64"),
        np.array([]),
        ["reg1", "reg1"],
    ],
    ids=[
        "non_exist_str",
        "non_exist_l_str",
        "non_exist_int",
        "non_exist_l",
        "empty_l",
        "empty_df",
        "empty_np",
        "multiple",
    ],
)
def test_EventKapitalRecover_incorrect_regions(test_sim, regions):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover(
            impact=1000,
            aff_regions=regions,
            aff_sectors="manufactoring",
            recovery_time=30,
        )


@pytest.mark.parametrize(
    "sectors",
    [
        "non_existing",
        ["non_existing"],
        1,
        [1],
        [],
        pd.DataFrame([],dtype="float64"),
        np.array([]),
        ["mining", "mining"],
    ],
    ids=[
        "non_exist_str",
        "non_exist_l_str",
        "non_exist_int",
        "non_exist_l",
        "empty_l",
        "empty_df",
        "empty_np",
        "multiple",
    ],
)
def test_EventKapitalRecover_incorrect_sectors(test_sim, sectors):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover(
            impact=1000,
            aff_regions="reg1",
            aff_sectors=sectors,
            recovery_time=30,
        )


def test_EventKapitalRecover_industries_regions_sectors(test_sim):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover(
            impact=1000,
            aff_regions="reg1",
            aff_sectors="manufactoring",
            aff_industries=[("reg1", "manufactoring")],
            recovery_time=30,
        )


@pytest.mark.parametrize(
    "industries",
    ["non_existing", ["non_existing"], 1, [1], [], pd.DataFrame([],dtype="float64"), np.array([])],
    ids=[
        "non_exist_str",
        "non_exist_l_str",
        "non_exist_int",
        "non_exist_l",
        "empty_l",
        "empty_df",
        "empty_np",
    ],
)
def test_EventKapitalRecover_incorrect_industries(test_sim, industries):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover(
            impact=1000,
            aff_industries=industries,
            recovery_time=30,
        )


@pytest.mark.parametrize(
    "impact, regions, sectors",
    [
        pytest.param([1000, 1000], ["reg1"], ["manufactoring"], id="more_impact"),
        pytest.param([1000], ["reg1", "reg2"], ["manufactoring"], id="less_impact_1"),
        pytest.param([1000], ["reg1"], ["manufactoring", "mining"], id="less_impact_2"),
    ],
)
def test_EventKapitalRecover_invalid_impact_industry_shape(
    test_sim, impact, regions, sectors
):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover(
            impact=impact,
            aff_regions=regions,
            aff_sectors=sectors,
            recovery_time=30,
        )
