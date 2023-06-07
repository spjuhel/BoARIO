.. role:: pythoncode(code)
   :language: python

.. _boario-quickstart:

Quickstart and examples
==============================

For the following examples, we assume you have followed installation instructions.

If you encounter any problems with one of the examples bellow, please `raise an issue`_
on the repository, or `contact the developer`_.

.. _raise an issue: https://github.com/spjuhel/BoARIO/issues/new

.. _contact the developer: pro@sjuhel.org


Quickstart example
___________________

See here `link <notebooks/boario-quickstart.ipynb>`_

ARIO vs :class:`~boario.model_base.ARIOBaseModel` vs :class:`~boario.extended_models.ARIOPsiModel`
_____________________________________________________________________________________________________________

Currently, two model classes are implemented, :class:`~boario.model_base.ARIOBaseModel` and :class:`~boario.extended_models.ARIOPsiModel`.
:class:`~boario.model_base.ARIOBaseModel` is essentially a theoretical implementation used to test a "simplistic and essential" version of the model, and should not
be used directly other than for developing new variants.

:class:`~boario.extended_models.ARIOPsiModel` mostly implements the version presented in :cite:`2013:hallegatte` for the multi-regional case,
as well as (optionally) the intermediate order mechanism presented in :cite:`2020:guan`.

One mechanism currently not implemented is the `macro effect` on final demand described in :cite:`2008:hallegatte`. This mechanism should be implemented in a future update.

:class:`~pymrio.core.mriosystem.IOSystem` input
________________________________________________________

The :class:`~pymrio.core.mriosystem.IOSystem` given to instantiate a :class:`~boario.extended_models.ARIOPsiModel` has to
have intermediate demand matrix ``Z``, final demand matrix ``Y`` and gross production vector ``x`` as attributes, and be balanced.

This should be the case for all MRIO parsed with the ``pymrio`` package.
Refer to its `documentation <https://pymrio.readthedocs.io/en/latest/>`_ for more details.

.. attention::

   Note that the (region,sector) Multiindexes for the matrices and vector are reordered by BoARIO to be in lexicographic order.

.. _temporal:

Temporal dimension
______________________


The temporal dimension is an important aspect of dynamically modeling indirect economic impacts.
Historically, ARIO has been used both using weekly and daily steps, but mostly the latter case.
BoARIO's implementation of ARIO aims at being independent of the ``temporal unit``
considered, notably to study how this aspect affect results.

This means it is virtually possible to run ARIO on any temporal granularity of your choosing.

For efficiency purpose, this implementation allows to simulate only some ``temporal unit`` and interpolate in between.
Hence a ``step`` can represent multiple ``temporal units``. Although by default, a ``step`` equals a ``temporal unit`` equals a `day` and
defaults values of characteristic times and other time related variable are accordingly expressed in number of days, these three terms are conceptually
different. For this reason we will favor the term ``temporal unit`` to designate the atomic period in the model throughout this documentation.

The number of ``temporal units`` to simulate can be set when instantiating the ``Simulation`` object like so:

.. code:: python

          sim = Simulation(model, n_temporal_units_to_sim=730)

By default, simulation run for 365 `temporal units` which are days by default.

.. _model_parameters:

Changing the model parameters
__________________________________


There are multiple theoretical and implemented versions of the ARIO model, each
with various parameters. One objective of BoARIO is to offer an extensive,
modular and adaptable implementation of these versions, in order to allow easy
access to version comparison, parameters values exploration and modeling
improvement.

If you are not familiar with the model, it is strongly advised to read the :ref:`boario-math` page of this documentation,
as well as :cite:`2013:hallegatte`.

Parameters are set when instantiating the model. The following block shows all currently available parameters as well as their default value.

.. code:: python

          model = ARIOPsiModel(
              pym_mrio=mrio,
              order_type="alt",
              alpha_base=1.0,
              alpha_max=1.25,
              alpha_tau=365,
              rebuild_tau=60,
              main_inv_dur=90,
              monetary_factor=10**6,
              temporal_units_by_step=1,
              iotable_year_to_temporal_unit_factor=365,
              infinite_inventories_sect=None,
              inventory_dict=None,
              productive_capital_vector=None,
              productive_capital_to_VA_dict=None,
              psi_param = 0.80,
              inventory_restoration_tau = 60,
          )


Here a quick description of each parameters. Please refer to both :ref:`the mathematical description<boario-math>` and the :ref:`api-ref` for further details.

* ``order_type`` : Setting it to ``"alt"`` makes the model use the intermediate order mechanism described in :cite:`2020:guan`. Any other value makes the model use the `classic` order mechanism used in :cite:`2013:hallegatte` (see :ref:`alt_orders`)

* ``alpha_base``, ``alpha_max``, ``alpha_tau`` respectively set the base overproduction, the maximum overproduction, and its characteristic time (in `temporal unit`).

* ``rebuild_tau`` sets the default rebuilding or recovering characteristic time for events (this value is overridden if specified directly in the Event object)

* ``inventory_dict`` should be a dictionary of ``sector:duration`` format, where all sector are present and ``duration`` is both the initial and goal duration for this input stock.

* ``main_inv_dur`` sets the default initial/goal inventory duration in `temporal unit` for all sectors if inventory_dict is not given.

* ``infinite_inventories_sect`` should be a list of inputs never constraining production (the stocks for these input will be virtually infinite when considering stock constraints) (overridden by ``inventory_dict``)

* ``monetary_factor`` should be equal to the monetary factor of the MRIO used (most of the time MRIO are in millions €/$, hence the default :math:`10^6`)

* ``temporal_units_by_step`` the number of `temporal units` to simulate every step. Setting it to 5 will divide the computation time by 5, but only one every 5 `temporal units` will actually be simulated. See :ref:`temporal`.

.. _year_to_temporal_unit_factor:

* ``iotable_year_to_temporal_unit_factor`` defines the `temporal unit` assuming the MRIO contains yearly values. Note that this has not been extensively tested and should be used with care.

* ``productive_capital_to_VA_dict`` should be a dictionary of ``sector:ratio`` format, where ratio is an estimate of Capital Stock over Value Added ratio. This is used to estimate the capital stock of each sector. By default the ratio is 4/1 for all sectors.

* ``productive_capital_vector`` can directly set the capital stock for all industries (ie regions*sectors sized). This overrides ``kapital_to_VA_dict``.

* ``psi_param`` and ``inventory_restoration_tau`` : see :ref:`boario-math-dyn`

.. note::

   All arguments except the mrio are keyword arguments (`ie` not positional), meaning you always need to specify <parameter = value>.
   (This also means you can put them in any order). The reason for this is to make parameter setting entirely explicit.


Reading the outputs and saving files
__________________________________________

.. _variables_evolution:

Monitoring the model variables
------------------------------

By default, simulations record the evolution of variables in temporary files, and the arrays
are accessible directly as attributes as long as the ``Simulation`` object exists.

Their available as DataFrame and contain the variables values for each regions for each sector for each temporal unit.
Each row represent a temporal unit. The columns are all the possible (region,sector) tuples, ie industries,
ordered in lexicographic order.

Here is a commented list of these attributes:

.. code:: python

        # The realised production
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
        # their shape is (temporal units, sectors, regions*sectors)
        # This one states for each temporal unit, for each input, for each (region,sector)
        # if the input was limiting production. For efficiency, information is stored as a
        # byte, -1 for False, 1 for True
        sim.limiting_inputs

It is also possible to record the inputs stocks, but this is disabled by defaults as its shape is the same as
``limiting_inputs``, but its ``dtype`` is ``float64``, which can very rapidly lead to huge files.

.. code:: python

          # Setup the recording of stocks
          sim = simulation(model, register_stocks=True)

          # Access the array
          sim.inputs_stocks

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

.. _github repository: https://github.com/spjuhel/BoARIO
.. _here: https://zenodo.org/record/5589597
