================
``boario.event``
================

.. automodule:: boario.event

   .. contents::
      :local:

.. currentmodule:: boario.event


Classes
=======

- :py:class:`Event`:
  Base class for events (Abstract)

- :py:class:`EventKapitalDestroyed`:
  Base subclass for events with productive capital destruction

- :py:class:`EventArbitraryProd`:
  Subclass for events with arbitrary impact on production capacity

- :py:class:`EventKapitalRecover`:
  Subclass for events where destroyed capital is recovered over time (without reconstruction demand)

- :py:class:`EventKapitalRebuild`:
  Subclass for events where destroyed capital requires rebuilding (Abstract)


.. autoclass:: Event
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: Event
      :parts: 1
      
   .. rubric:: Attributes and Methods

.. autoclass:: EventKapitalDestroyed
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: EventKapitalDestroyed
      :parts: 1
      
   .. rubric:: Attributes and Methods

.. autoclass:: EventArbitraryProd
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: EventArbitraryProd
      :parts: 1
      
   .. rubric:: Attributes and Methods

.. autoclass:: EventKapitalRecover
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: EventKapitalRecover
      :parts: 1
      
   .. rubric:: Attributes and Methods

.. autoclass:: EventKapitalRebuild
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: EventKapitalRebuild
      :parts: 1
      
   .. rubric:: Attributes and Methods


Variables
=========

- :py:data:`Impact`
- :py:data:`IndustriesList`
- :py:data:`SectorsList`
- :py:data:`RegionsList`

.. autodata:: Impact
   :annotation:

   .. code-block:: text

      typing.Union[list, dict, numpy.ndarray, pandas.core.frame.DataFrame, pandas.core.series.Series, int, float, numpy.integer]

.. autodata:: IndustriesList
   :annotation:

   .. code-block:: text

      typing.Union[typing.List[typing.Tuple[str, str]], pandas.core.indexes.multi.MultiIndex]

.. autodata:: SectorsList
   :annotation:

   .. code-block:: text

      typing.Union[typing.List[str], pandas.core.indexes.base.Index, str]

.. autodata:: RegionsList
   :annotation:

   .. code-block:: text

      typing.Union[typing.List[str], pandas.core.indexes.base.Index, str]
