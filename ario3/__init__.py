r''' # What is Ario³ ? Ario³, is an object-oriented python implementation
project of the Adaptative Regional Input Output (ARIO) model.

Its objectives are to give an accessible and inter-operable implementation of
ARIO, as well as tools to visualize and analyze simulation outputs and to
evaluate the effects of many parameters of the model.

It is still an ongoing project.

# How does Ario³ work?

In a nutshell, Ario³ takes the following inputs : - an IO table (such as
EXIOBASE or EORA26) *imports it, creates a `mrio_module` object,* - simulation
and mrio parameters (as json files), which govern the simulation, - event(s)
description(s), which are used as the perturbation to analyse during the
simulation

And can produce the following outputs:

  - the step by step, sector by sector, region by region evolution of most
of the variables involved in the simulation (production, demand, stocks, ...)
  - aggregated indicators for the whole simulation (shortage durations,
aggregated impacts, ...)
  - more to come

# Example of use:

In this commented example, we run the model with one simple
event using 'usual' parameters, and compute the aggregated indicators of the
simulation. We suppose we run the following script from `~/ARIO3/`

```python

# We import the base of the model
import ario3.simulation.base as sim
# We also import the indicators module
from ario3.indicators.indicators import Indicators
import pathlib

# We instantiate a dictionary with the parameters
# (it is also possible to use a json file)

params = {
    # The name of the working directory to use (relative to current wd)
    "storage_dir": "storage",
    # The directory to use to store results (relative to storage_dir)
    # i.e. here, the model will look for files in ~/ARIO3/storage/ and
    # store results in ~/ARIO3/storage/results/
    "results_storage": "results",
    "bool_run_detailled": True,
    # This tells the model to register the evolution of the stocks
    # of every industry (the file can be quite large (2Gbytes+ for
    # a 365 days simulation with exiobase))
    "register_stocks": True,
    # Parameters of the model (we detail these in the documentation)
   "psi_param": 0.8,
   "model_time_step": 1,
   "timestep_dividing_factor": 365,
   "inventory_restoration_time": 40,
   "alpha_base": 1.0,
   "alpha_max": 1.25,
   "alpha_tau": 365,
   "rebuild_tau": 30,
   "n_timesteps": 365,
   "min_duration": (365 // 100) * 25,
   "impacted_region_base_production_toward_rebuilding": 0.001,
   "row_base_production_toward_rebuilding": 0.0,
   "mrio_params_file":"mrio_params.json"
}

# Here we define the event perturbing the simulation
event = {
    # The list of affected regions (their index in the mrio table)
    # Here we just have France
    "aff-regions": ["FR"],
    # The list of affected sectors
    # (here we specify that all sectors are impacted)
    "aff-sectors": "all",
    # The shares of the damages distributed between regions
    # (1 as we have only one region)
    "dmg-distrib-regions": [ 1 ],
    # The type of distribution of damages for the sectors
    # (more on that in the doc)
    "dmg-distrib-sectors-type": "gdp",
    # 'gdp' distribution doesn't require this parameter to be set
    "dmg-distrib-sectors": [],
    # The duration of the event (not implemented yet, so it has no effect)
    "duration": 1,
    # A name for the event (usefull when simulating multiple events)
    "name": "0",
    # The step at which the event shall occur during the simulation
    "occur": 5,
    # The quantity of damages caused by the event (in IO table monetary)
    "q_dmg":100000000,
    # The sectors mobilised to answer the rebuilding demand
    # and the share of the demand they answer
    "rebuilding-sectors": {
        "Construction (45)":0.15,
        "Manufacture of machinery and equipment n.e.c. (29)" : 0.20,
        "Manufacture of furniture; manufacturing n.e.c. (36)" : 0.20,
        "Manufacture of office machinery and computers (30)": 0.15,
    }
}

# We load the mrio table from a pickle file (created with the help of the
# pymrio module, more on that in the doc)
mrio_path = pathlib.Path(params['storage_dir'])/"mrio.pkl"

# We initiate a model instance ...
model = sim.Simulation(mrio_path, params)

# ... add the list of events (just one here) to the model ...
model.read_events_from_list([event])

# ... and launch the simulation with :
model.loop()

# Once the simulation is over we can compute some indicators :
indic = Indicators.from_storage_path(
                                     pathlib.Path(sim_params['storage_dir']),
                                     params=sim_params
)
indic.update_indicators()
indic.write_indicators()
```

This script will produce files in `~/ARIO3/storage/results/` :

 - `simulated_events.json` : A json record of the events that were simulated
   during the loop.

 - `indicators.json` : A json record (produced by `indic.write_indicators()`)
   of the computed indicators.

 - `record` files. These are [memmap nd.array](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html#numpy.memmap)
   of the different recorded variables.

## Record files

You may read these directly into a numpy array with :
```
np.memmap("results/+record_name+_record",
           mode='r+',
           dtype='float64',
           shape=(t,n_sectors*n_regions)
)
```
Where `shape` is the shape mentioned afterward.

   1. `classic_demand` : the sum of intermediate and final demand addressed to
   each industries. Its shape is `(n_timesteps, n_sectors*n_regions)`

   2. `iotable_XVA` : the realised production of each industry. Its shape is
   `(n_timesteps, n_sectors*n_regions)`

   3. `iotable_X_max` : the production capacity of each industry. Its shape is
   `(n_timesteps, n_sectors*n_regions)`

   4. `overprod_vector` : the overproduction scaling of each industry. Its
   shape is `(n_timesteps, n_sectors*n_regions)`

   5. `rebuild_demand` : the additional direct demand created by the event
   for rebuilding, for each industry.
   Its shape is `(n_timesteps, n_sectors*n_regions)`

   6. `rebuild_prod` : the part of production attributed to rebuilding, for each
   industry. Its shape is `(n_timesteps, n_sectors*n_regions)`

   7. `final_demand_unmet` : the final demand that was not met due to rationing,
   for each industry. Its shape is `(n_timesteps, n_sectors*n_regions)`

   8. `stocks` : the stocks of each input for each industry.
   Its shape is `(n_timesteps*n_sectors, n_sectors*n_regions)`.
   Note that this file is not created if `register_stocks` is set to `False`
   in the params.

   9. `limiting_stocks` : a boolean matrix, telling for each input and for each
   industry if the stock is limiting for production.
   Its shape is `(n_timesteps*n_sectors, n_sectors*n_regions)`.
   Reading this array directly require to change the dtype
   to 'bool' in the above command.

# More description to come

'''
