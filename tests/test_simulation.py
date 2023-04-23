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
from boario.extended_models import ARIOPsiModel  # The core of the model
from boario.indicators import Indicators         # A class computing and storing indicators based on a simulation
from boario.event import Event, EventKapitalRebuild, EventArbitraryProd, EventKapitalRecover  # A class defining a shock on capital
from boario.utils.recovery_functions import *

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

# Tests that a Simulation object can be initialized with a valid model and default parameters. 
def test_initialize_simulation_with_valid_model_and_default_parameters(test_model):   
    # Act
    sim = Simulation(test_model)
    
    # Assert
    assert isinstance(sim, Simulation)
    assert sim.model == test_model
    assert sim._save_events == False
    assert sim._save_params == False
    assert sim._save_index == False
    assert sim.all_events == []
    assert sim._register_stocks == False
    assert sim.n_temporal_units_to_sim == 365
    assert sim._files_to_record == []
    assert isinstance(sim.boario_output_dir, str)
    assert sim.results_dir_name == None

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
    assert isinstance(sim._regional_sectoral_productive_capital_destroyed_evolution, np.ndarray)

    assert sim._production_evolution.shape == (sim.n_temporal_units_to_sim, test_model.model.n_sectors * test_model.n_regions)
    assert sim._production_cap_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._final_demand_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._io_demand_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._rebuild_demand_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._overproduction_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._final_demand_unmet_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._rebuild_production_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    assert sim._inputs_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors, test_model.n_sectors * test_model.n_regions)
    assert sim._limiting_inputs_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors, test_model.n_sectors * test_model.n_regions)
    assert sim._regional_sectoral_productive_capital_destroyed_evolution == (sim.n_temporal_units_to_sim, test_model.n_sectors * test_model.n_regions)
    
    assert sim.currently_happening_events == []
    assert Event.temporal_unit_range == sim.n_temporal_units_to_sim
    assert sim.n_temporal_units_simulated == 0
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
        sim.add_events(Event())
        
    with pytest.raises(TypeError):
        sim.add_event([])

# Tests that an error is raised when trying to run the simulation loop with a negative number of temporal units to simulate.  
def test_run_simulation_loop_with_negative_number_of_temporal_units(test_model):
    with pytest.raises(ValueError):
        sim = Simulation(test_model, n_temporal_units_to_sim=-1)

# Tests that the simulation results can be saved to memmaps and retrieved as pandas dataframes.  
def test_save_simulation_results_to_memmaps_and_retrieve_as_pandas_dataframes(test_model):
    sim = Simulation(test_model)
    sim.loop(progress=False)
    assert isinstance(sim.production_realised, pd.DataFrame)
    assert isinstance(sim.production_capacity, pd.DataFrame)
    assert isinstance(sim.final_demand, pd.DataFrame)
    assert isinstance(sim.intermediate_demand, pd.DataFrame)
    assert isinstance(sim.rebuild_demand, pd.DataFrame)
    assert isinstance(sim.overproduction, pd.DataFrame)
    assert isinstance(sim.final_demand_unmet, pd.DataFrame)
    assert isinstance(sim.rebuild_prod, pd.DataFrame)
    print(inputs_stocks)
    assert isinstance(sim.inputs_stocks, pd.DataFrame)
    assert isinstance(sim.limiting_inputs, pd.DataFrame)
    assert isinstance(sim.productive_capital_to_recover, pd.DataFrame)

# Tests the behavior of the simulation with different combinations of parameters.  
def test_of_simulation_with_different_combinations_of_parameters(test_model):
    sim = Simulation(test_model, register_stocks=True, n_temporal_units_to_sim=100,
                        save_events=True, save_params=True, save_index=True, save_records=['production_realised'])
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
