Simulation context and state variable saving
=========================================================


:py:class:`Simulation` objects act as wrappers for a model and one or multiple event, and thus sets a context for a simulation. It handles the following aspect:

    * The access to the different variables timeseries.
    * Which outputs to save.
    * The length in `unit step` of the simulation.
    * The list of events to simulate.
    * The looping process of the simulation.

Simulation length
___________________

The length of the simulation can be set by the ``n_temporal_units_to_sim`` argument at initialisation.

.. seealso::
   :ref:`temporal`

Running multiple simulations
_______________________________

At the moment we recommend redefining both a model and a simulation object when running multiple simulations.

Reading the outputs and saving files
__________________________________________

.. _variables_evolution:

Monitoring the model variables
---------------------------------

By default, a simulation records the evolution of variables in numpy arrays, which
are accessible directly as attributes as long as the ``Simulation`` object exists.
Optionally the records can also be set to memmaps instead, which are saved as files at the end of the simulation (note that these files are temporary by default).

These records contain the variables values for each regions for each sector for each temporal unit and can also be accessed as formatted DataFrames, where each row represents a temporal unit and the columns represent all the possible (region,sector) tuples, i.e., industries, ordered in lexicographic order.

Here is a commented list of the different variables accessible:

.. code:: python

        # The production actually realised at each step
        sim.production_realised

        # The production capacity
        sim.production_capacity

        # The share of realised production distributed to rebuilding
        sim.rebuild_prod

        # The overproduction factor
        sim.overproduction

        # The (total) intermediate demand (ie how much intermediate demand was addressed to sector i in region j)
        sim.intermediate_demand

        # The (total) final demand (note that the final demand is currently fix in the model)
        sim.final_demand

        # The (total) rebuild demand
        sim.rebuild_demand

        # The amount of final demand that couldn't be satisfied
        sim.final_demand_unmet

        # The remaining amount of destroyed (ie not recovered/rebuilt) capital
        sim.productive_capital_to_recover

        # Note that the following array have one more dimension,
        # its shape is (temporal units, sectors, regions*sectors)
        # This one states for each temporal unit, for each input, for each (region,sector)
        # if the input was limiting production. For efficiency, information is stored as a
        # byte, -1 for False, 1 for True
        sim.limiting_inputs

It is also possible to record the inputs stocks, but this is disabled by default as its shape is the same as
``limiting_inputs``, but its ``dtype`` is ``float64``, which can very rapidly lead to huge arrays difficult to have in memory.

.. code:: python

          # Setup the recording of stocks
          sim = simulation(model, register_stocks=True)

          # Access the array
          sim.inputs_stocks

.. hint::
   These DataFrames can easily be saved using any of pandas writers. BoARIO also makes it possible to save the raw arrays (see below).

.. _index_records:

Saving indexes, parameters and events simulated
-----------------------------------------------

In order to keep experiments organized and reproducible,
the following arguments can be used when instantiating a
``Simulation`` object:

* ``"save_index"`` : ``True|False``, if ``True``, saves a file :file:`boario_output_dir/results/jsons/indexes.json`, where the indexes (regions, sectors, final demand categories, etc.) are stored.

* ``"save_params"`` : ``True|False``, if ``True``, saves a file :file:`boario_output_dir/results/jsons/simulated_params.json`, where the simulation parameters are stored.

* ``"save_events"`` : ``True|False``, if ``True``, saves a file :file:`boario_output_dir/results/jsons/simulated_events.json`, where the indexes (regions, sectors, final demand categories, etc.) are stored.

.. _recording:

Record files
-------------

By defaults the arrays recording the evolution of variables are temporary files,
which are deleted when the ``Simulation`` object is destroyed.

It is however possible to ask the ``Simulation`` object to save any selection of these raw arrays,
by giving a list and an output directory when instantiating. Here is the complete list of variables than can be saved:

``['production_realised', 'production_capacity', 'final_demand', 'intermediate_demand', 'rebuild_demand',
'overproduction', 'final_demand_unmet', 'rebuild_prod', 'inputs_stocks', 'limiting_inputs', 'kapital_to_recover']``

.. attention::

   ``inputs_stocks`` still requires the argument ``register_stocks`` to be True in order for the file to be saved.


For example the following code will create the files ``"production_realised"`` and ``"final_demand_unmet"``
in the specified folder (or to a temporary directory prefixed by ``"boario"`` by default).

.. code:: python

          sim = Simulation(
              model,
              save_records=["production_realised", "final_demand_unmet"],
              boario_output_dir="folder of your choosing/",
          )


Files saved like this are raw numpy arrays and can then be read with:

.. code:: python

          import numpy as np

          # For all records except limiting_inputs and inputs_stocks
          np.memmap(
              "path/to/file",
              mode="r+",
              dtype="float64",
              shape=(n_temporal_units, n_sectors * n_regions),
          )

          # For limiting_inputs
          np.memmap(
              "path/to/file",
              mode="r+",
              dtype="byte",
              shape=(n_sectors * n_temporal_units, n_sectors * n_regions),
          )

          # For inputs_stocks
          np.memmap(
              "path/to/file",
              mode="r+",
              dtype="float64",
              shape=(n_sectors * n_temporal_units, n_sectors * n_regions),
          )
