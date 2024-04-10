=====================
``boario.model_base``
=====================

.. automodule:: boario.model_base

   .. contents::
      :local:

.. currentmodule:: boario.model_base


Functions
=========

- :py:func:`lexico_reindex`:
  Reindex IOSystem lexicographicaly


.. autofunction:: lexico_reindex


Classes
=======

- :py:class:`ARIOBaseModel`:
  Undocumented.


.. autoclass:: ARIOBaseModel
   :members:

   .. rubric:: Inheritance
   .. inheritance-diagram:: ARIOBaseModel
      :parts: 1
      
   .. rubric:: Attributes and Methods


Variables
=========

- :py:data:`INV_THRESHOLD`
- :py:data:`VALUE_ADDED_NAMES`
- :py:data:`VA_idx`

.. autodata:: INV_THRESHOLD
   :annotation:

   .. code-block:: text

      0

.. autodata:: VALUE_ADDED_NAMES
   :annotation:

   .. code-block:: text

      ['VA',
       'Value Added',
       'value added',
       'factor inputs',
       'factor_inputs',
       'Factors Inputs',
       'Satellite Accounts',
       'satellite accounts',
       'satellite_accounts',
       'satellite']

.. autodata:: VA_idx
   :annotation:

   .. code-block:: text

      array(['Taxes less subsidies on products purchased: Total',
             'Other net taxes on production',
             "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
             "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
             "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
             'Operating surplus: Consumption of fixed capital',
             'Operating surplus: Rents on land',
             'Operating surplus: Royalties on resources',
             'Operating surplus: Remaining net operating surplus'], dtype=object)
