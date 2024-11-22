.. _mriot-input:

##############
MRIOT Input
##############

Using a :class:`~pymrio.core.mriosystem.IOSystem` as input
================================================================

The :class:`~pymrio.core.mriosystem.IOSystem` given to instantiate a model requires to
contain intermediate demand matrix ``Z``, final demand matrix ``Y`` and gross production
vector ``x`` as attributes, and to be balanced.

This should be the case for all MRIOT parsed with the ``pymrio`` package.
Refer to its `documentation <https://pymrio.readthedocs.io/en/latest/>`_ for more details.

In particular, if you want to build your own MRIOT based on your own data source, you should read
`this part <https://pymrio.readthedocs.io/en/latest/notebooks/pymrio_directly_assign_attributes.html>`_
of pymrio's documentation.

.. attention::

   Note that the (region,sector) Multi-indexes for the matrices and vector are reordered by BoARIO to be in lexicographic order.


Productive capital
====================

The model requires an estimation of the productive capital of each industries in
order to model the effect of productive capital destruction. In `BoARIO` this
stock can either be directly given by the user as a parameter to the model, or
estimated from the value added computed from the MRIOT using per sectors ratios.

.. attention::

   If no ratios are given, the models defaults to `productive
   capital = 4 * VA` :cite:`2008:hallegatte`. This is a very coarse assumption
   that you should not rely on.

.. attention::

   The model checks whether the value added computed as ``source_mriot.x.T -
   source_mriot.Z.sum(axis=0)`` (`i.e.`, the gross output minus the intermediate
   requirements) contains any negative values, and set it to 0. for industries
   (and raises a warning)

   This in turn sets the productive capital for these industries to be null too
   when it is estimated from VA to capital ratios.

Using a Local Input-Output Table (LIOT)
===========================================

So far the model has not been tested with LIOT. While it doesn't pose any problem conceptually, the shape of the data might not fit the model.

We suggest the user defines a two-regions MRIOT with one region being the local one, and the other being the `Rest of the World` made of the exports and imports flows.

Feedback on successively running the model in such setup would be greatly appreciated!
