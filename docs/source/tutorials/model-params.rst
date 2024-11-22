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
              pym_mriot=mriot,
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


Here is a quick description of each parameters. Please refer to both :ref:`the mathematical description<boario-math>` and the :ref:`api-ref` for further details.

* ``order_type`` : Setting it to ``"alt"`` makes the model use the intermediate order mechanism described in :cite:`2020:guan`. Any other value makes the model use the `classic` order mechanism used in :cite:`2013:hallegatte` (see :ref:`alt_orders`)

* ``alpha_base``, ``alpha_max``, ``alpha_tau`` respectively set the base overproduction, the maximum overproduction, and its characteristic time (in `temporal units`).

* ``rebuild_tau`` sets the default rebuilding or recovering characteristic time for events (this value is overridden if specified directly in the Event object)

* ``inventory_dict`` should be a dictionary of ``sector:duration`` format, where all sector are present and ``duration`` is both the initial and goal duration for this input stock.

* ``main_inv_dur`` sets the default initial/goal inventory duration in `temporal units` for all sectors if inventory_dict is not given.

* ``infinite_inventories_sect`` should be a list of inputs never constraining production (the stocks for these input will be virtually infinite when considering stock constraints) (overridden by ``inventory_dict``)

* ``monetary_factor`` should be equal to the monetary factor of the MRIOT used (most of the time MRIOT are in millions €/$, hence the default :math:`10^6`)

* ``temporal_units_by_step`` the number of `temporal units` to simulate every step. Setting it to 5 will divide the computation time by 5, but only one every 5 `temporal units` will actually be simulated. See :ref:`temporal`.

.. _year_to_temporal_unit_factor:

* ``iotable_year_to_temporal_unit_factor`` defines the `temporal unit` assuming the MRIOT contains yearly values. Note that this has not been extensively tested and should be used with care.

* ``productive_capital_to_VA_dict`` should be a dictionary of ``sector:ratio`` format, where ratio is an estimate of Capital Stock over Value Added ratio. This is used to estimate the capital stock of each sector. By default the ratio is 4/1 for all sectors.

* ``productive_capital_vector`` can directly set the capital stock for all industries (ie regions*sectors sized). This overrides ``productive_capital_to_VA_dict``.

* ``psi_param`` and ``inventory_restoration_tau`` : see :ref:`boario-math-dyn`

.. note::

   All arguments except the mriot are keyword arguments (`i.e.`, not positional), meaning you always need to specify <parameter = value>.
   (This also means you can put them in any order). The reason for this is to make parameter setting entirely explicit.


.. _temporal:

Focus on the temporal dimension
----------------------------------

.. warning::
   These features have not been extensively tested and should be handled with great care.
   Feedback is welcome!


The temporal dimension is an important aspect of dynamically modeling indirect economic impacts.
Historically, ARIO has been used both using weekly and daily steps, but mostly the latter case.
BoARIO's implementation of ARIO aims at being independent of the ``temporal unit``
considered, notably to study how this aspect influences the results.

This means it is possible to run ARIO on virtually any temporal granularity of your choosing.

Furthermore, for efficiency purposes, this implementation allows you to simulate only some ``temporal units``
and interpolate state variables between them.

Hence a ``step`` can represent multiple ``temporal units``. By default, a ``step`` equals a ``temporal unit`` equals a `day` and
defaults values of characteristic times and other time related variable are accordingly expressed in number of days.
However, because it is customisable, we favor the term ``temporal unit`` to designate the atomic period in the model throughout this documentation.

The number of ``temporal units`` to simulate can be set when instantiating the ``Simulation`` object like so:

.. code:: python

          sim = Simulation(model, n_temporal_units_to_sim=730)

By default, simulation run for 365 `temporal units`, representing days.
