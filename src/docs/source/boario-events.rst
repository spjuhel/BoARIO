.. _boario-events:

########################
How to define Events
########################

===============
Introduction
===============

The BoARIO model is used to study the indirect economic impacts consequent to `local` event [#local]_.
At the moment, a shock is represented by a quantity of damages expressed in monetary value and leading to:

1. A destruction of capital leading to a reduction in production capacity production (See :ref:`boario-math-prod`).
2. A rebuilding demand, corresponding to destroyed capital and addressed to a set of rebuilding sectors (See :ref:`boario-math-rebuilding-demand`).

To define such an event, we hence need the following information:

1. A quantity of damages
2. The region(s) affected
3. The sector(s) affected
4. The sector(s) considered to be rebuilding sectors

In addition, it is also possible to define a duration (corresponding to the time before rebuilding can start)
as well as time of occurrence (when studying interaction between multiple events).

During a simulation BoARIO uses :class:`~boario.event.Event` objects, which are created via either python ``dict`` or JSON files.
This page details the different keys and possible associated values.

.. [#local] At the moment events are local to the regional unit of the MRIO used (e.g. a country/world region in the case of Exiobase3).


===============================
Define affected region(s)
===============================

There are two ``key:value`` pairs governing the affected region(s):

1. ``"aff_regions" : [list of affected regions]``
2. ``"dmg_distrib_regions" : [list of floats]``

The first list has to contain the `region id(s)` (at least one) of the affected regions. The id(s) have to correspond to valid indexes in the MRIO used.
For example, Exiobase3 uses ISO-ALPHA2 codes (``FR`` for France, ``DE`` for Germany, ``CN`` for China, etc.) and custom 2-letters code for world regions.

The second list has to contain a same-length list of decimal values indicating how damages are distributed between multiple regions if there are.
Note that the sum of the list has to equal 1.
For convenience, in the case all affected regions share the damages equally, it is possible to replace the list by the ``"shared"`` keyword.

The following would define an event affecting both France and Germany, and where France would receive 30% of the damages and Germany 70%:

.. code-block:: json

   {
   "aff_regions": [
                "FR", "DE"
                ],
   "dmg_distrib_regions": [
                0.3, 0.7
                ],
   }

Depending on the modeling goal, it is also possible to define two events, one for France, and one for Germany.

================================================
Define damages, duration and occurrence
================================================

The key for damages is ``"q_dmg"``. Damages should be given in the same currency as the one used
in the MRIO, without a decimal prefix, i.e. if the MRIO uses M€, damage have to be given in € [#mrio-params]_.

The key for duration is ``"duration"``. Its value should be a number of days, and represent the period
before rebuilding can start.

The key for occurrence is ``"occur"``. Its value should be an integer comprised between 1 and the number of days simulated.
When simulating only one event, this value is of limited use, although you may want to set it some days after the start
for clearer visualisation (or in order to verify that the model remains stable before the shock). Note that when simulating more than
one day per step [#daystep]_, if the given value is not a multiple of the number of days simulated by step, the event will actually
occur during the closest upper multiple.

.. [#mrio-params] This is inline with the ``"monetary_factor"`` parameter specified in the MRIO parameter file (see :ref:`boario-mrio-params`)

.. [#daystep] See the ``"model_time_step"`` parameter in :ref:`boario-sim-params`.


.. _aff-sectors-params:

============================================================
Affected sectors and sector damage distribution
============================================================

Affected sectors are defined by the ``"aff_sectors" : [list of affected sectors]`` key:value pair.
The list has to contain valid sector names in the MRIO used. It is also possible to replace the list
by the ``"all"`` value, if all sectors are considered impacted.

As different events (as well as contexts) may affect different sectors differently, it is possible to define how
damages are distributed along the impacted sectors by either of the following:

1. Setting ``"dmg_distrib_sectors_type"`` to ``"gdp"``, damages are distributed along the impacted sectors proportionally to their GDP contribution.
2. Otherwise, setting the ``"dmg_distrib_sectors"`` to a list of decimal values comprised between 0 and 1 and summing to 1, where each value defines the share of the damages distributed to the corresponding sector in the ``"aff_sectors"`` list.

.. _reb-sectors-params:

======================
Rebuilding sectors
======================

Rebuilding sectors are defined by the ``"rebuilding-sectors"`` parameter using the following format:

.. code-block:: json

   {
    "rebuilding-sectors": {
        "sector1": 0.30,
        "sector3": 0.70
    }
   }

Where ``"sector1"`` and ``"sector3"`` are valid sector names in the MRIO used.
In this case, the ``sector1`` will answer 30% of the rebuilding demand, and ``"sector3"`` the remaining 70%.
Note that the demand is addressed to all industries corresponding to these sectors. How demand is distributed
along the different regions of the MRIO is governed by the transaction matrix of the MRIO.

==============
Convenience
==============

For convenience of use :

1. When dealing with multiple events, it is possible to set the ``"name"`` key to any value,
which only purpose is to give an id to the event.
2. When invoking the :meth:`~boario.simulation.Simulation.read_events_from_list`, a ``simulated_events.json`` file
is created in the results directory with the list of events dictionaries given.
3. Multiple checks are done when initializing an :class:`~boario.event.Event` object and during simulation, raising errors if values are incorrect.
However, it is highly possible that some cases are not covered. Don't hesitate to `contact the developer`_ or better create an issue on the `github repository`_


.. _contact the developer: pro@sjuhel.org

.. _github repository: https://github.com/spjuhel/BoARIO
