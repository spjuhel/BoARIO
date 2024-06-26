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

boario.disable_console_logging()

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
        ev = EventKapitalRecover.from_scalar_regions_sectors(
            impact=100000,
            regions=["reg1"],
            sectors=["manufactoring", "mining"],
            recovery_time=30,
        )


def test_EventKapitalRecover_init_before_sim_after_model(test_model):
    with pytest.raises(AttributeError):
        ev = EventKapitalRecover.from_scalar_regions_sectors(
            impact=1,
            regions=["reg1"],
            sectors=["manufactoring", "mining"],
            recovery_time=30,
        )


def test_Event_abstract(test_sim):
    with pytest.raises(TypeError):
        ev = Event.from_scalar_regions_sectors(
            impact=100000, regions=["reg1"], sectors=["manufactoring", "mining"]
        )


def test_EventKapitalRecover_wrong_init_after_sim_after_model(test_sim):
    with pytest.raises(TypeError):
        ev = EventKapitalRecover.from_scalar_regions_sectors(impact=1, recovery_time=30)

    with pytest.raises(ValueError):
        ev = EventKapitalRecover.from_series(
            impact=pd.Series(dtype="float64"), recovery_time=30
        )

    with pytest.raises(ValueError):
        ev = EventKapitalRecover.from_dataframe(
            impact=pd.DataFrame(dtype="float64"), recovery_time=30
        )


@pytest.mark.parametrize(
    "impact",
    [
        0,
        -1,
        [],
        pd.Series([], dtype="float64"),
        pd.DataFrame([], dtype="float64"),
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
    with pytest.raises((ValueError, UFuncTypeError, TypeError)):
        if isinstance(impact, int):
            ev = EventKapitalRecover.from_scalar_regions_sectors(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                recovery_time=30,
            )
        elif isinstance(impact, pd.Series):
            ev = EventKapitalRecover.from_series(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                recovery_time=30,
            )
        elif isinstance(impact, pd.DataFrame):
            ev = EventKapitalRecover.from_dataframe(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                recovery_time=30,
            )
        else:
            ev = EventKapitalRecover.from_scalar_regions_sectors(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
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
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
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
def test_EventKapitalRecover_incorrect_regions(test_sim, regions):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover.from_scalar_regions_sectors(
            impact=1000,
            regions=regions,
            sectors="manufactoring",
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
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
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
def test_EventKapitalRecover_incorrect_sectors(test_sim, sectors):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRecover.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors=sectors,
            recovery_time=30,
        )


def test_EventKapitalRecover_duplicate_sectors(test_sim):
    with pytest.warns(UserWarning):
        ev = EventKapitalRecover.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors=["mining", "mining"],
            recovery_time=30,
        )


def test_EventKapitalRecover_duplicate_regions(test_sim):
    with pytest.warns(UserWarning):
        ev = EventKapitalRecover.from_scalar_regions_sectors(
            impact=1000,
            regions=["reg1", "reg1"],
            sectors="mining",
            recovery_time=30,
        )


@pytest.mark.parametrize(
    "industries",
    [
        "non_existing",
        ["non_existing"],
        1,
        [1],
        [],
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
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
        ev = EventKapitalRecover.from_scalar_industries(
            impact=1000,
            industries=industries,
            recovery_time=30,
        )


############### REBUILD

def test_EventKapitalRebuild_init_empty():
    with pytest.raises(TypeError):
        ev = EventKapitalRebuild()

    with pytest.raises(TypeError):
        ev = EventKapitalRebuild(impact=1)


def test_EventKapitalRebuild_wrong_init_after_sim_after_model(test_sim):
    with pytest.raises(TypeError):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(impact=1, recovery_time=30)

    with pytest.raises(ValueError):
        ev = EventKapitalRebuild.from_series(
            impact=pd.Series(dtype="float64"),
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )

    with pytest.raises(ValueError):
        ev = EventKapitalRebuild.from_dataframe(
            impact=pd.DataFrame(dtype="float64"),
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )


@pytest.mark.parametrize(
    "impact",
    [
        0,
        -1,
        [],
        pd.Series([], dtype="float64"),
        pd.DataFrame([], dtype="float64"),
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
def test_EventKapitalRebuild_incorrect_impact(test_sim, impact):
    with pytest.raises((ValueError, UFuncTypeError, TypeError)):
        if isinstance(impact, int):
            ev = EventKapitalRebuild.from_scalar_regions_sectors(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                rebuild_tau=30,
                rebuilding_sectors={"construction":1.0}

            )
        elif isinstance(impact, pd.Series):
            ev = EventKapitalRebuild.from_series(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                rebuild_tau=30,
                rebuilding_sectors={"construction":1.0}

            )
        elif isinstance(impact, pd.DataFrame):
            ev = EventKapitalRebuild.from_dataframe(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                rebuild_tau=30,
                rebuilding_sectors={"construction":1.0}

            )
        else:
            ev = EventKapitalRebuild.from_scalar_regions_sectors(
                impact=impact,
                regions="reg1",
                sectors="manufactoring",
                rebuild_tau=30,
                rebuilding_sectors={"construction":1.0}
            )


@pytest.mark.parametrize(
    "regions",
    [
        "non_existing",
        ["non_existing"],
        1,
        [1],
        [],
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
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
def test_EventKapitalRebuild_incorrect_regions(test_sim, regions):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions=regions,
            sectors="manufactoring",
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )


@pytest.mark.parametrize(
    "sectors",
    [
        "non_existing",
        ["non_existing"],
        1,
        [1],
        [],
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
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
def test_EventKapitalRebuild_incorrect_sectors(test_sim, sectors):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors=sectors,
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )


def test_EventKapitalRebuild_duplicate_sectors(test_sim):
    with pytest.warns(UserWarning):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors=["mining", "mining"],
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )


def test_EventKapitalRebuild_duplicate_regions(test_sim):
    with pytest.warns(UserWarning):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions=["reg1", "reg1"],
            sectors="mining",
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )


@pytest.mark.parametrize(
    "industries",
    [
        "non_existing",
        ["non_existing"],
        1,
        [1],
        [],
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
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
def test_EventKapitalRebuild_incorrect_industries(test_sim, industries):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRebuild.from_scalar_industries(
            impact=1000,
            industries=industries,
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.0}
        )

@pytest.mark.parametrize(
    "reb_secs",
    [
        "non_existing",
        ["non_existing"],
        {"construction":"A"},
        {"nonexist":1.0},
        1,
        [1],
        [],
        pd.DataFrame([], dtype="float64"),
        np.array([]),
    ],
    ids=[
        "non_exist_str",
        "non_exist_l_str",
        "exist_non_num",
        "non_exist_dict",
        "non_exist_int",
        "non_exist_l",
        "empty_l",
        "empty_df",
        "empty_np",
    ],
)
def test_EventKapitalRebuild_incorrect_rebuilding_sectors(test_sim, reb_secs):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors="manufactoring",
            rebuild_tau=30,
            rebuilding_sectors=reb_secs
        )

def test_EventKapitalRebuild_incorrect_rebuilding_shares(test_sim):
    with pytest.raises(ValueError):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors="manufactoring",
            rebuild_tau=30,
            rebuilding_sectors={"construction":1.5}
        )

@pytest.mark.parametrize(
    "reb_tau",
    [
        1.5,
        "str",
        -1,
        0,
        [1],
        [],
        pd.DataFrame([1], dtype="float64"),
        np.array([1]),
    ],
    ids=[
        "float",
        "str",
        "neg_int",
        "zero",
        "list",
        "empty_l",
        "df",
        "np",
    ],
)
def test_EventKapitalRebuild_incorrect_reb_tau(test_sim, reb_tau):
    with pytest.raises((ValueError, TypeError, KeyError)):
        ev = EventKapitalRebuild.from_scalar_regions_sectors(
            impact=1000,
            regions="reg1",
            sectors="manufactoring",
            rebuild_tau=reb_tau,
            rebuilding_sectors={"construction":1.0}
        )
