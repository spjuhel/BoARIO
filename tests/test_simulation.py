from pathlib import Path
import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np

# import the different classes
from boario import event
from boario.simulation import (
    Simulation,
    _equal_distribution,
    _normalize_distribution,
)  # Simulation wraps the model
from boario.extended_models import ARIOPsiModel  # The core of the model
from boario.event import Event
from boario.utils.recovery_functions import *


@pytest.fixture
def test_mriot():
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
def test_model(test_mriot):
    model = ARIOPsiModel(test_mriot)
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
    sim.loop()
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
    sim.loop()
    assert sim.production_realised.shape == (100, test_model.n_industries)


def minimal_simulation(test_model):
    # Instantiate the model and the simulation
    sim = Simulation(test_model)
    sim.loop()
    raise SystemExit(0)


def test_minimal_simulation(test_model):
    with pytest.raises(SystemExit):
        minimal_simulation(test_model)


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


def test_equal_distribution():
    # Case 1: Typical case
    affected = pd.Index(["A", "B"])
    addressed_to = pd.Index(["X", "Y", "Z"])
    result = _equal_distribution(affected, addressed_to)

    expected = pd.DataFrame(
        {"A": [1 / 3, 1 / 3, 1 / 3], "B": [1 / 3, 1 / 3, 1 / 3]}, index=["X", "Y", "Z"]
    )
    pd.testing.assert_frame_equal(result, expected)

    # Case 2: Single affected and single addressed_to
    affected = pd.Index(["A"])
    addressed_to = pd.Index(["X"])
    result = _equal_distribution(affected, addressed_to)

    expected = pd.DataFrame({"A": [1.0]}, index=["X"])
    pd.testing.assert_frame_equal(result, expected)

    # Case 3: Empty affected index
    affected = pd.Index([])
    addressed_to = pd.Index(["X", "Y"])
    result = _equal_distribution(affected, addressed_to)

    expected = pd.DataFrame(index=["X", "Y"], columns=[])
    pd.testing.assert_frame_equal(result, expected)

    # Case 4: Empty addressed_to index
    affected = pd.Index(["A", "B"])
    addressed_to = pd.Index([])

    with pytest.raises(ValueError):
        result = _equal_distribution(affected, addressed_to)

    # Case 5: Both indices empty
    affected = pd.Index([])
    addressed_to = pd.Index([])

    with pytest.raises(ValueError):
        result = _equal_distribution(affected, addressed_to)

    # Case 6: Unequal distribution check (error expected)
    affected = pd.Index(["A", "B"])
    addressed_to = pd.Index(["X", "Y"])
    result = _equal_distribution(affected, addressed_to)
    for col in result.columns:
        assert result[col].sum() == 1  # Each column should sum to 1


def test_normalize_distribution():
    # Case 1: Normalizing a Series distribution
    dist = pd.Series([2, 3, 5], index=["X", "Y", "Z"])
    affected = pd.Index(["A"])
    addressed_to = pd.Index(["X", "Y", "Z"])
    result = _normalize_distribution(dist, affected, addressed_to)

    expected = pd.DataFrame({"A": [0.2, 0.3, 0.5]}, index=["X", "Y", "Z"])
    pd.testing.assert_frame_equal(result, expected)

    # Case 2: Normalizing a DataFrame distribution
    dist = pd.DataFrame({"A": [2, 3, 5], "B": [4, 6, 10]}, index=["X", "Y", "Z"])
    affected = pd.Index(["A", "B"])
    addressed_to = pd.Index(["X", "Y", "Z"])
    result = _normalize_distribution(dist, affected, addressed_to)

    expected = pd.DataFrame(
        {"A": [0.2, 0.3, 0.5], "B": [0.2, 0.3, 0.5]}, index=["X", "Y", "Z"]
    )
    pd.testing.assert_frame_equal(result, expected)

    # Case 6: Mismatched indices in Series
    dist = pd.Series([2, 3], index=["X", "Y"])
    affected = pd.Index(["A"])
    addressed_to = pd.Index(["X", "Y", "Z"])
    with pytest.raises(KeyError):
        _normalize_distribution(dist, affected, addressed_to)

    # Case 7: Invalid distribution type
    dist = [2, 3, 5]  # Not a Series or DataFrame
    affected = pd.Index(["A"])
    addressed_to = pd.Index(["X", "Y", "Z"])
    with pytest.raises(
        ValueError, match="given distribution should be a Series or a DataFrame"
    ):
        _normalize_distribution(dist, affected, addressed_to)
