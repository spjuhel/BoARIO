.. _boario-events:

########################
How to define Events
########################

===============
Introduction
===============

The BoARIO model is used to study the indirect economic impacts consequent to a `local` event [#local]_.
At the moment, a shock is represented either by:

* A quantity of damages expressed in monetary value and leading to:

  1. A destruction of capital leading to a reduction in production capacity production (See :ref:`boario-math-prod`).
  2. A possible rebuilding demand, corresponding to destroyed capital or a part of it and addressed to a set of rebuilding sectors.

* Directly by a fraction of production capacity lost for a certain period, which is then recovered exogenously.

To define such an event, we hence need the following information:

1. A quantity of damages (or fraction of production capacity)
2. The region(s) affected
3. The sector(s) affected
4. In the case a rebuilding demand is considered: the sector(s) considered to be rebuilding sectors

In addition, it is also possible to define a duration (corresponding to the time before rebuilding can start)
as well as time of occurrence (when studying interaction between multiple events), and a recovery duration (:ref:`boario-math-recovery`).

During a simulation BoARIO uses :class:`~boario.event.Event` objects.
This page details the different aspect of instantiating such objects.

.. [#local] At the moment events are local to the regional unit of the MRIOT used (e.g. a country/world region in the case of EXIOBASE3 MRIO).

Simulation context
------------------

Creating events requires a simulation context to exist,
notably to assess the validity of the event (the affected regions, sectors, the occurrence, ...). Hence,
trying to instantiate an ``Event`` object without already having a ``Simulation`` (and therefore an ``ARIOPsiModel``)
will raise an error.

In the following, we will assume the following code was run initially:

.. code:: python

        import pymrio
        import pandas as pd

        from boario.simulation import Simulation
        from boario.extended_models import ARIOPsiModel

        mriot = pymrio.load_test().calc_all()
        model = ARIOPsiModel(mriot)
        sim = Simulation(model, n_temporal_units_to_sim=365)

Here are the regions and sectors defined in this context:

.. code:: pycon

          >>> model.regions

          array(['reg1', 'reg2', 'reg3', 'reg4', 'reg5', 'reg6'], dtype='<U4')

          >>> model.sectors

          array(['construction', 'electricity', 'food', 'manufactoring', 'mining',
                 'other', 'trade', 'transport'], dtype='<U13')


=========================================
Different types of Event objects
=========================================

Currently three types of events are implemented and both consider
a destruction of capital as the impact:

* :class:`~boario.event.EventKapitalRecover` defines events for which the destroyed
  capital is recovered (i.e. regained along a specified ``recovery curve`` without an
  associated rebuilding demand and rebuilding sectors)

* :class:`~boario.event.EventKapitalRebuild` define events for which the destroyed
  capital is rebuild (i.e. creates a corresponding rebuilding demand addressed toward
  a set of rebuilding sectors)

* :class:`~boario.event.EventArbitraryProd` defines events for which production capacity
  is arbitrarily decrease for a set of industries.

========================================
Defining Event from an impact vector
========================================

The more direct way to instantiate an event is to use a :class:`pandas.Series` object
where the index is the set of affected industries and the values are the impact for each.

Suppose you want to represent an event impacting the "manufactoring" an "mining" sectors of region "reg2" for respectively
5'000'000€ and 3'000'000€ (assuming the MRIO is in €). You can define the following ``Series``:

.. code:: pycon

   >>> import pandas as pd
   >>> impact = pd.Series(
   ...     data=[5000000.0, 3000000.0],
   ...     index=pd.MultiIndex.from_product(
   ...         [["reg2"], ["manufactoring", "mining"]], names=["region", "sector"]
   ...     ),
   ... )
   >>> impact

   region  sector
   reg2    manufactoring    5000000.0
           mining           3000000.0
   dtype: float64


Create a :class:`~boario.event.EventKapitalRecover`
---------------------------------------------------

For this type of event you need to specify the characteristic time for recovery ``"recovery_time"``:
let us use 30 days here.

You can also choose a recovery function/curve between ``"linear"`` (by default), ``"concave"``
and ``"convexe"`` following what is done in :cite:`2019:koks,2016:koks`.

You may also choose a specific ``occurrence`` (default is 1) which is especially useful if you
simulate multiple events.

.. warning::

   Note that it is not advised to set the occurrence at 0 as some indicators require the first
   step to be at equilibrium.

You may as well choose a ``duration`` for the event. The duration is the amount of `temporal units`
before which recovery starts. It allows the possibility to represent delayed recovery due to the event
(e.g. for a flood the region inaccessible because of the water)

Finally for convenience you can give a name for the event.

.. code:: python

          ev = EventKapitalRecover.from_series(
              impact=impact,
              recovery_time=30,
              recovery_function="concave",
              occurrence=5,
              duration=7,
              name="Flood in reg2",
          )


Create a :class:`~boario.event.EventKapitalRebuild`
------------------------------------------------------

When creating this type of event, you need to specify the rebuilding characteristic time ``"rebuild_tau"``
as well which are the rebuilding sectors and how the rebuilding demand is distributed among them.

The rebuilding sectors can be given as a ``Series`` where the index are the sectors, and the values
are the share of the rebuilding demand they will answer (hence, it should sum to 1).

.. hint::

   The rebuilding sectors can also be given as a dict, where keys are sectors and values are shares.

By default, the rebuilding demand is equal to the totality of the impact (assuming all the value that
was destroyed is rebuilt), but you may set a ``"rebuilding_factor"`` (default 1) to define a lower
(or greater) rebuilding demand.

Otherwise, you can also set ``occurrence``, ``duration`` and ``name``  similarly to
:class:`~boario.event.EventKapitalRecover`.

The following code defines an event for which the rebuilding demand is 90% of the capital destroyed,
and where 80% of the demand is answered by the construction sector, and 20% by the manufactoring sector.

.. code:: python

          ev = EventKapitalRebuild.from_series(
              impact=impact,
              rebuild_tau=60,
              rebuilding_sectors={"construction": 0.8, "manufactoring": 0.2},
              rebuilding_factor=0.9,
              occurrence=5,
              duration=7,
              name="Flood in reg2",
          )

Create a :class:`~boario.event.EventArbitraryProd`
------------------------------------------------------

.. warning::
   A critical bug was found for this class and this type of Event has been made unavailable
   until fixed.

When creating this type of event, the impact values should be value between 0 and 1 stating
the fraction of production capacity unavailable due to the event.

As for :class:`~boario.event.EventKapitalRecover`, a recovery function and a recovery time may be given.
Otherwise, production capacity is restored instantaneously after the duration of the event has elapsed.

.. code:: pycon

   >>> import pandas as pd
   >>> impact = pd.Series(
   ...     data=[0.3, 0.1],
   ...     index=pd.MultiIndex.from_product(
   ...         [["reg2"], ["manufactoring", "mining"]], names=["region", "sector"]
   ...     ),
   ... )
   >>> impact

   region  sector
   reg2    manufactoring    0.3
           mining           0.1
   dtype: float64

.. code:: python

          ev = EventArbitraryProd.from_series(
              impact=impact,
              occurrence=5,
              duration=7,
              recovery_function="linear",
              recovery_time=5,
          )

================================
Defining events from a scalar
================================

You can also define an event from a scalar impact
(except for :class:`~boario.event.EventArbitraryProd` at the moment).
This requires to define which industries are affected and
how the impact is distributed among the industries.

You can take a look at the quickstart example
`here <notebooks/boario-quickstart.ipynb>`_.

In this section, we go over the different cases first, and then show
code examples for each case.

In order to define which industries are affected you can:

1. Directly give them as a pandas MultiIndex with affected regions as the first level
   and affected sectors at the second.

2. Give them as a list of regions affected, as well as a list of sectors affected.
   The resulting affected industries being the cartesian product of those two lists.

.. warning::

  Note that the second option does not allow to have different sectors affected in each region.

By default, the impact will be uniformally distributed among the affected regions and
the impact per region is then also uniformally distributed among the affected sector in the region.

Otherwise, there are multiple ways to setup a custom distribution:

1. Directly give a vector (any variable that can be transformed as a numpy array)
   of per affected industry share of the impact (although in this case you should
   probably directly give the impact vector).
2. Give a vector of the share per region, and the share per sector.
3. Give a vector of the share per region, and specify ``"gdp"`` for the per sector
   distribution. This will distribute each regional impact toward each affected sector
   proportionally to their share of the regional GDP among the affected sectors.
   For example: Suppose we look at the impact in region ``"reg2"``, and suppose
   ``"manufactoring"`` and ``"mining"`` are both affected. Now suppose
   ``"manufactoring"`` account for 40% of ``"reg2"``'s GDP and ``"mining"``
   for 10%. The ``"manufactoring"`` sector will receive :math:`(40 * 100) / (40 + 10) = 80\%`
   of the impact and ``"mining"`` the remaining :math:`(10 * 100) / (40 + 10) = 20\%`.

.. note::

  The GDP shares are estimated from the MRIO table used, based on the Value Added,
  which itself is computed as the gross output minus the intermediate demand:

  :math:`\textrm{GVA} = \iox - \ioz`

.. warning::

  You should not assume the default impact distribution setting is a good representation
  of the general case, as different regions and sectors are most probably differently
  impacted by an event. It is strongly advised to setup your own distribution in accordance
  with your study.

.. code:: pycon

  >>> impact = 1000000
  >>> aff_industries = pd.MultiIndex(
  ...     [("reg1", "manufactoring"), ("reg3", "mining")], names=["region", "sector"]
  ... )


.. _contact the developer: pro@sjuhel.org

.. _github repository: https://github.com/spjuhel/BoARIO
