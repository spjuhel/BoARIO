Detailled examples
=======================

For the following examples, we assume you have a working python environment,
with all required packages [#requirements]_. These examples were tested using ``Ubuntu 20.04``,
``conda 4.12.0`` and ``python 3.8.8``.

We also assume BoARIO source files are located in ``~/BoARIO/``.

If you encounter any problems with one of the examples bellow, please `raise an issue`_
on the repository, or `contact the developer`_.

.. [#requirements] This list will come soon !

.. _raise an issue: https://github.com/spjuhel/BoARIO/issues/new

.. _contact the developer: pro@sjuhel.org


Minimal working example
____________________________

On the `github repository`_ you should find all available resources for generating
a minimal working example of a simulation with the model, with the exception of an
Exiobase3 input-output tables source file, which can be found `here`_.

For the sake of simplicity, the script generates a 7-sectors MRIO from the source file.
To do so, place yourself in the BoARIO source directory and use the following commands:

.. code:: console
          cd other/ && wget https://zenodo.org/record/5589597/files/IOT_2011_ixi.zip
          cd ../
          python scripts/generate-example-files.py -o ~/BoARIO/testing ./other/


.. note::
   The parsing of the zip file can take some time, be patient.

.. note::
   You may replace year ``2011`` when downloading the MRIO by any year you want.
   If you have already downloaded the zip file, you can simply copy it into ``./other``

.. note::
   You can choose to create the testing directory anywhere, but note you will have to change
   the paths given bellow.

The script should tell you if any problem arise. If all goes smoothly, it will generate
the specified directory, with a pickle file of the aggregated MRIO (named ``exiobase3_minimal``),
as well as ``params.json``, ``mrio_params.json`` and ``event.json`` file, which you may modify should
you want to try out different parameters (See the relevant sections in the Contents menu).

Once this is done, we suggest you use a Jupyter Notebook, but you can easily adapt the following
using a python or IPython console, or a script directly.

First, load BoARIO with the following:

.. code:: python
          import sys
          # We need the following because BoARIO is not yet installable
          sys.path.insert(1, '~/BoARIO/')
          # We import the base for simulation
          import boario.simulation as sim
          # pathlib is a very useful library for handling paths, although you can use strings directly
          import pathlib

Then you can create the simulation environment with (we recommend using the "ARIOPsi" version at the moment, which is more stable than the base version):

.. code:: python
          mrio_path = pathlib.Path("~/BoARIO/testing/exiobase3_minimal.pkl")
          params_path = pathlib.Path("~/BoARIO/testing/params.json")
          mrio_params_path = pathlib.Path("~/BoARIO/testing/mrio_params.json")
          simulation_test = sim.Simulation(params_path, mrio_path, mrio_params=mrio_params_path, modeltype="ARIOPsi")


Initialisation shows a lot of logs which should be self-explanatory.
Once you see:

.. code:: console
          [INFO] - [simulation.py > __init__() > 239] - Initialized !

The simulation context is ready.

You can then load the event(s) and launch the simulation with:

.. code:: python
          simulation_test.read_events(event_path)
          simulation_test.loop()

Once again, a lot of logs should pop-up and the run should execute. An ETA shows how long the run should take.

A ``result`` directory is created inside ``~/BoARIO/testing``, containing ``record`` files. These are :py:class:`numpy.memmap` of the different recorded variables.

Record files
-------------

You may read these directly into a numpy array with :

.. code:: python

    np.memmap("results/+record_name+_record",
               mode='r+',
               dtype='float64',
               shape=(t,n_sectors*n_regions)
    )

Where ``shape`` is the shape mentioned afterward.

1. ``classic_demand`` : the sum of intermediate and final demand addressed to each industries. Its shape is ``(n_timesteps, n_sectors*n_regions)``

2. ``iotable_XVA`` : the realised production of each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

3. ``iotable_X_max`` : the production capacity of each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

4. ``overprod_vector`` : the overproduction scaling of each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

5. ``rebuild_demand`` : the additional direct demand created by the event for rebuilding, for each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

6. ``rebuild_prod`` : the part of production attributed to rebuilding, for each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

7. ``final_demand_unmet`` : the final demand that was not met due to rationing, for each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

8. ``stocks`` : the stocks of each input for each industry. Its shape is ``(n_timesteps*n_sectors, n_sectors*n_regions)``. Note that this file is not created if ``register_stocks`` is set to ``False`` in the simulation parameters.

9. ``limiting_stocks`` : a boolean matrix, telling for each input and for each industry if the stock is limiting for production. Its shape is ``(n_timesteps*n_sectors, n_sectors*n_regions)``. Reading this array directly require to change the dtype to 'bool' in the above command.

Indicators and parquet files
-------------------------------

You may also run:

.. code::
   from boario.indicators import Indicators
   indic = Indicators.from_folder(
                               pathlib.Path("~/BoARIO/testing/results"),
                               indexes_file=pathlib.Path("~/BoARIO/testing/results/indexes.json"))
   indic.update_indicators()
   indic.write_indicators()


Which generate easier to read parquet files (using :py:function:`read_parquet`) as well as ``indicators.json``, ``fd_loss.json`` and ``prod_chg.json`` which show various indicators.

.. note::
   The script also generate:
   - ``simulated_events.json`` : A json record of the events that were simulated during the loop.
   - ``simulated_params.json`` : A json record of parameters that were used during the loop.

An Indicators object contains mostly all results from the simulation in dataframe. For instance :pythoncode:`indic.prod_df` is a dataframe of the production of each sector of each region for every step. Note that some dataframes are in wide format while other are in long format, for treatment purpose. Also note that some of these dataframes are saved in the result folder as `parquet`_ files. They are simply the memmaps ``records`` with the indexes.

Calling :pythoncode:`indic.update_indicators()` fills the :pythoncode:`indic.indicators` dictionary with the following indicators:

- The total (whole world, all sectors) final consumption not met during the simulation :pythoncode:`indicator['tot_fd_unmet']`.

- The final consumption not met in the region(s) affected by the shock :pythoncode:`indicator['aff_fd_unmet']`.

- The rebuild duration (ie the number of step during which rebuild demand is not zero) :pythoncode:`indicator['rebuild_durations']`.

- If there was a shortage (:pythoncode:`indicator['shortage_b']`), its start and end dates :pythoncode:`indicator['shortage_date_start']` and :pythoncode:`indicator['shortage_date_end']`.

- The top five `(region,sectors)` tuples where there was the biggest absolute change of production compared to a no shock scenario.

It also produce dataframes indicators :

- The production change by region over the simulation (:pythoncode:`indic.prod_chg_region`) giving for each region the total gain or loss in production during the simulation.

- The final consumption not met by region over the simulation (:pythoncode:`indic.df_loss_region`) giving for each region the total loss in consumption during the simulation.

.. _`parquet`: https://parquet.apache.org/
.. _github repository: https://github.com/spjuhel/BoARIO
.. _here: https://zenodo.org/record/5589597

Simple event
______________

In the following example we show how to set up a simple mono-event simulation, using parameters defined directly as dictionaries:

.. include:: ../../../../api-examples/simulation/example-read_events_from_list.rstinc
