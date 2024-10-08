from pathlib import Path
import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np

# import the different classes
import boario
from boario import event
from boario.model_base import ARIOBaseModel
from boario.simulation import Simulation  # Simulation wraps the model
from boario.extended_models import ARIOPsiModel  # The core of the model
from boario.event import (
    Event,
    EventKapitalRebuild,
    EventArbitraryProd,
    EventKapitalRecover,
)  # A class defining a shock on capital
from boario.utils.recovery_functions import *

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
    model = ARIOPsiModel(test_mrio)
    return model


@pytest.fixture
def test_sim(test_model):
    sim = Simulation(test_model)
    return sim


@pytest.mark.parametrize(
    "region",
    ["reg1", "non_exist"],
    ids=[
        "exist",
        "non_exist",
    ],
)
def test_event_compatibility(test_sim, region):
    # error raise on invalid region

    # error raise on invalid sector

    # error raise on invalid occurrence
    pass


# Tests that a Simulation object can be initialized with a valid model and default parameters.
def test_initialize_simulation_with_valid_model_and_default_parameters(test_model):
    # Act
    sim = Simulation(test_model, register_stocks=True)

    # Assert
    assert isinstance(sim, Simulation)
    assert sim.model == test_model
    assert sim._save_events == False
    assert sim._save_params == False
    assert sim._save_index == False
    assert sim.all_events == []
    assert sim._register_stocks == True
    assert sim.n_temporal_units_to_sim == 365
    # assert sim._files_to_record == [
    #     "_production_evolution",
    #     "_production_cap_evolution",
    #     "_final_demand_evolution",
    #     "_io_demand_evolution",
    #     "_rebuild_demand_evolution",
    #     "_overproduction_evolution",
    #     "_final_demand_unmet_evolution",
    #     "_rebuild_production_evolution",
    #     "_inputs_evolution",
    #     "_limiting_inputs_evolution",
    #     "_regional_sectoral_productive_capital_destroyed_evolution",
    # ]
    assert isinstance(sim.output_dir, Path)

    assert isinstance(sim._production_evolution, np.ndarray)
    assert isinstance(sim._production_cap_evolution, np.ndarray)
    assert isinstance(sim._final_demand_evolution, np.ndarray)
    assert isinstance(sim._io_demand_evolution, np.ndarray)
    assert isinstance(sim._rebuild_demand_evolution, np.ndarray)
    assert isinstance(sim._overproduction_evolution, np.ndarray)
    assert isinstance(sim._final_demand_unmet_evolution, np.ndarray)
    assert isinstance(sim._rebuild_production_evolution, np.ndarray)
    assert isinstance(sim._inputs_evolution, np.ndarray)
    assert isinstance(sim._limiting_inputs_evolution, np.ndarray)
    assert isinstance(
        sim._regional_sectoral_productive_capital_destroyed_evolution, np.ndarray
    )

    assert sim._production_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._production_cap_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._final_demand_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._io_demand_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._rebuild_demand_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._overproduction_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._final_demand_unmet_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._rebuild_production_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._inputs_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._limiting_inputs_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors,
        test_model.n_sectors * test_model.n_regions,
    )
    assert sim._regional_sectoral_productive_capital_destroyed_evolution.shape == (
        sim.n_temporal_units_to_sim,
        test_model.n_sectors * test_model.n_regions,
    )

    # assert sim.currently_happening_events == []
    # assert Event.temporal_unit_range == sim.n_temporal_units_to_sim
    assert sim.n_temporal_units_simulated == 0
    assert sim.current_temporal_unit == 0
    assert sim._n_checks == 0
    assert sim._monotony_checker == 0
    assert sim.scheme == "proportional"
    assert sim.has_crashed == False


# Tests that an error is raised when trying to add events to the simulation with invalid parameters.
def test_add_events_with_invalid_parameters(test_model):
    # Arrange
    sim = Simulation(test_model)

    # Act & Assert
    with pytest.raises(TypeError):
        sim.add_events(16)

    with pytest.raises(TypeError):
        sim.add_event(Event())


# Tests that an error is raised when trying to run the simulation loop with a negative number of temporal units to simulate.
def test_run_simulation_loop_with_negative_number_of_temporal_units(test_model):
    with pytest.raises(ValueError):
        sim = Simulation(test_model, n_temporal_units_to_sim=-1)


# Tests that the simulation results can be saved to memmaps and retrieved as pandas dataframes.
def test_save_simulation_results_to_memmaps_and_retrieve_as_pandas_dataframes(
    test_model,
):
    sim = Simulation(test_model, register_stocks=True)
    sim.loop(progress=False)
    assert isinstance(sim.production_realised, pd.DataFrame)
    assert isinstance(sim.production_capacity, pd.DataFrame)
    assert isinstance(sim.final_demand, pd.DataFrame)
    assert isinstance(sim.intermediate_demand, pd.DataFrame)
    assert isinstance(sim.rebuild_demand, pd.DataFrame)
    assert isinstance(sim.overproduction, pd.DataFrame)
    assert isinstance(sim.final_demand_unmet, pd.DataFrame)
    assert isinstance(sim.rebuild_prod, pd.DataFrame)
    assert isinstance(sim.inputs_stocks, pd.DataFrame)
    assert isinstance(sim.limiting_inputs, pd.DataFrame)
    assert isinstance(sim.productive_capital_to_recover, pd.DataFrame)


# Tests the behavior of the simulation with different combinations of parameters.
def test_of_simulation_with_different_combinations_of_parameters(test_model):
    sim = Simulation(
        test_model,
        register_stocks=True,
        n_temporal_units_to_sim=100,
        save_events=True,
        save_params=True,
        save_index=True,
        save_records=["production_realised"],
    )
    sim.loop(progress=False)
    assert sim.production_realised.shape == (100, test_model.n_industries)


def minimal_simulation(test_model):
    # Instantiate the model and the simulation
    sim = Simulation(test_model)
    sim.loop()
    raise SystemExit(0)


def test_minimal_simulation(test_model):
    with pytest.raises(SystemExit):
        minimal_simulation(test_model)


def test_minor_rec_event(test_sim):
    ev = event.from_scalar_regions_sectors(
        impact=100,
        event_type="recovery",
        affected_regions=["reg1"],
        affected_sectors=["mining"],
        recovery_tau=30,
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop("mining") > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_medium_rec_event(test_sim):
    ev = event.from_scalar_regions_sectors(
        impact=100000,
        event_type="recovery",
        affected_regions=["reg1"],
        affected_sectors=["mining"],
        recovery_tau=30,
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop("mining") < (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_crashing_rec_event(test_sim):
    ev = event.from_scalar_regions_sectors(
        event_type="recovery",
        impact=100000,
        affected_regions=["reg1"],
        affected_sectors=["mining"],
        recovery_tau=30,
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop("mining") < (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_minor_reb_event(test_sim):
    ev = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=100000,
        affected_regions=["reg1"],
        affected_sectors=["manufactoring"],
        rebuilding_sectors={"construction": 1.0},
        duration=350,
        rebuild_tau=100,
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop(["mining", "manufactoring"])
        > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_shortage_reb_event(test_sim):
    ev = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=10000000,
        affected_regions=["reg1"],
        affected_sectors=["manufactoring"],
        rebuilding_sectors={"construction": 1.0},
        duration=90,
        rebuild_tau=100,
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (min_values < (1.0 - 1 / test_sim.model.monetary_factor)).all()
    assert test_sim.model.had_shortage


# def test_crashing_reb_event(test_sim):
#     ev = event.from_scalar_regions_sectors(
#         event_type="rebuild",
#         impact=10000000000,
#         affected_regions=["reg1"],
#         affected_sectors=["manufactoring"],
#         rebuilding_sectors={"construction": 1.0},
#         duration=90,
#         rebuild_tau=100,
#     )
#     test_sim.add_event(ev)
#     test_sim.loop()
#     assert test_sim.has_crashed
