.. _mriot-input:

MRIOT Input
============


Using a :class:`~pymrio.core.mriosystem.IOSystem` as input
________________________________________________________________

The :class:`~pymrio.core.mriosystem.IOSystem` given to instantiate a :class:`~boario.extended_models.ARIOPsiModel` has to
have intermediate demand matrix ``Z``, final demand matrix ``Y`` and gross production vector ``x`` as attributes, and be balanced.

This should be the case for all MRIOT parsed with the ``pymrio`` package.
Refer to its `documentation <https://pymrio.readthedocs.io/en/latest/>`_ for more details.

.. attention::

   Note that the (region,sector) Multiindexes for the matrices and vector are reordered by BoARIO to be in lexicographic order.
