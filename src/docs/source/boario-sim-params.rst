.. _boario-sim-params:

##########################
Simulation parameters
##########################

There are multiple theoretical and implemented versions of the ARIO model, each
with various parameters. One objective of BoARIO is to offer an extensive,
modular and adaptable implementation of these versions, in order to allow easy
access to version comparison, parameters values exploration and modeling
improvement.

Simulation parameters cover all parameters related to the simulation context
(directories, numbers of days/steps) and the ARIO model (which version,
parameters values, ...), they are stored in a python dictionary or JSON file,
loaded when initializing a :class:`~boario.simulation.Simulation` instance. They
are generally referred to as ``params`` (where MRIO table parameters and events
parameters are generally referred to with a prefix, i.e. ``mrio_params`` and
``event_params``). This page details each parameters and their use.

Files, directories, and results parameters
===================================================

BoARIO uses and generates quite a number of files. In order to keep experiments
organized, the following parameters are defined:

* ``"output_dir"`` : The path [#path]_ where output files should be stored.
* ``"result_storage"`` : The name of the directory inside ``"output_dir"`` where
  to store the results.
* ``"register_stocks"`` : ``True|False``, if ``True``, register the stocks'
  matrix for the whole simulation
  (A ``(n_steps, n_regions*n_sectors, n_sectors)``
  sized matrix of 32 bit floats which can quickly become a huge file). Writing
  the file may also hamper execution speed.
* ``"mrio_params_file"`` : The path where MRIO table parameters JSON file should
  be.

.. _boario-sim-params-time:

Steps/Time parameters
==========================

The ARIO model has been run using different time unit (weeks, day(s)), these
parameters should allow to choose the time unit, the numbers of time unit
per simulated step and the number of step to simulate. [#name]_

* ``"timestep_dividing_factor"`` : The number of time unit in a year (i.e. 365 for daily time unit, 52 weekly, ...)
* ``"model_time_step"`` : The number of time unit in a simulated step
* ``"n_timesteps"`` : The number of steps to simulate
* ``"min_duration"`` : If the simulation is set to stop on a certain condition [#condition]_, wait at least this number of time unit before checking for this condition

ARIO parameters
===================

* ``"model_type"`` : Either ``"ARIOBase"`` or ``"ARIOPsi"``.
  The Base model correspond to a simplified hybrid version of Hallegatte2013_ and Guan2020_ models.
  Inventory constraints are direct (no `Psi` parameter) and there is no characteristic time for inventories resupplying.
  The Psi model correspond mostly to Hallegatte2013_ version. Parameter `Psi` is used when computing inventories constraints and there is a characteristic time for inventories resupplying. See :ref:`boario-math` or the model classes in the :ref:`api-ref` for more details.
* ``"psi`` : If model type is ``ARIOPsi``, set the value for this parameter. See :ref:`boario-math`.
* ``"order_type"`` : If ``"alt"``, sets the _`order module` to the one introduced by Guan2020_ version, where industries adapt there orders based on suppliers production. Else, orders stays proportional to the initial transaction matrix. See :ref:`order module <boario-math-orders>`.
* ``"alpha_base"`` : Initial overproduction factor. Should be ``1.0`` most of the time. See :ref:`boario-math-overprod`.
* ``"alpha_max"`` : Maximum overproduction factor. Usually set to ``1.25`` in the literature.
* ``"alpha_tau"`` : Overproduction characteristic time, in days. Usually set to ``365`` in the literature.
* ``"rebuild_tau"`` : Rebuilding characteristic time, in days.
* ``"impacted_region_base_production_toward_rebuilding"`` : [WIP, not active] Maximum initial fraction of production which can be allocated towards rebuilding, in the impacted region(s).
* ``"row_base_production_toward_rebuilding"`` : [WIP, not active] Maximum initial fraction of production which can be allocated towards rebuilding, in the non-impacted regions (Rest of the World).

.. [#path] Path can be given as absolute or as relative to the working directory
           (i.e. the directory from which the program is executed, in doubt use
           absolute)

.. [#name] These parameters names will probably change at some point as they are quite unclear at the moment

.. [#condition] Not working yet

.. _Hallegatte2013: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1539-6924.2008.01046.x

.. _Guan2020: https://www.nature.com/articles/s41562-020-0896-8

.. _contact the developer: pro@sjuhel.org

.. _github repository: https://github.com/spjuhel/BoARIO
