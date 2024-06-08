.. _mriot-input:

MRIOT Input
============


Using a :class:`~pymrio.core.mriosystem.IOSystem` as input
________________________________________________________________

The :class:`~pymrio.core.mriosystem.IOSystem` given to instantiate a model requires to
contain intermediate demand matrix ``Z``, final demand matrix ``Y`` and gross production
vector ``x`` as attributes, and to be balanced.

This should be the case for all MRIOT parsed with the ``pymrio`` package.
Refer to its `documentation <https://pymrio.readthedocs.io/en/latest/>`_ for more details.

In particular, if you want to build your own MRIOT based on your own data source, you should read
`this part <https://pymrio.readthedocs.io/en/latest/notebooks/pymrio_directly_assign_attributes.html>`_
of pymrio's documentation.

.. attention::

   Note that the (region,sector) Multiindexes for the matrices and vector are reordered by BoARIO to be in lexicographic order.
