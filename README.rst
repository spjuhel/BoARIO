.. role:: pythoncode(code)
   :language: python

#######
BoARIO
#######

BoARIO : The Adaptative Regional Input Output model in python.

What is BoARIO ?
=================

BoARIO, is a python implementation project of the Adaptative Regional Input Output (ARIO) model [`Hallegatte 2013`_].

Its objectives are to give an accessible and inter-operable implementation of ARIO, as well as tools to visualize and analyze simulation outputs and to
evaluate the effects of many parameters of the model.

This implementation would not have been possible without the `Pymrio`_ module and amazing work of [`Stadler 2021`_] !

It is still an ongoing project (in parallel of a PhD project).

.. _`Stadler 2021`: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.251/
.. _`Hallegatte 2013`: https://doi.org/10.1111/j.1539-6924.2008.01046.x
.. _`Pymrio`: https://pymrio.readthedocs.io/en/latest/intro.html

Here is a non-exhaustive chronology of academic works with or about the ARIO model :

.. image:: https://raw.githubusercontent.com/spjuhel/BoARIO/master/imgs/chronology.svg?sanitize=true
           :width: 900
           :alt: ARIO academic work chronology

What is ARIO ?
===============

ARIO stands for Adaptive Regional Input-Output. It is an hybrid input-output / agent-based economic model, designed to compute indirect costs from economic shocks. Its first version dates back to 2008 and has originally been developed to assess the indirect costs of natural disasters (Hallegatte 2008).

In ARIO, the economy is modelled as a set of economic sectors and a set of regions. Each economic sector produces its generic product and draws inputs from an inventory. Each sector answers to a total demand consisting of a final demand (household consumption, public spending and private investments) of all regions (local demand and exports) and intermediate demand (through inputs inventory resupply). An initial equilibrium state of the economy is built based on multi-regional input-output tables (MRIO tables).


Where to get BoARIO ?
==========================

You can install BoARIO from ``pip`` with:

.. code:: console

   pip install boario


The full source code is also available on Github at: https://github.com/spjuhel/BoARIO

More info in the :ref:`installation page<installation>` of the documentation.

How does BoARIO work?
=========================

In a nutshell, BoARIO takes the following inputs :

- an IO table (such as EXIOBASE3 or EORA26) in the form of an :class:`pymrio.IOSystem` object, using the `Pymrio`_ python package.

- multiple parameters which govern the simulation,

- event(s) description(s), which are used as the perturbation to analyse during the simulation

And produce the following outputs:

- the step by step, sector by sector, region by region evolution of most of the variables involved in the simulation (`production`, `demand`, `stocks`, ...)

- aggregated indicators for the whole simulation (`shortages duration`, `aggregated impacts`, ...)

Example of use
=================

See :ref:`Boario quickstart<boario-quickstart>`.

Credits
========

Associated PhD project
------------------------

This model is part of my PhD on the indirect impact of extreme events.
This work was supported by the French Environment and Energy Management Agency
(`ADEME`_).

.. image:: https://raw.githubusercontent.com/spjuhel/BoARIO/master/imgs/Logo_ADEME.svg?sanitize=true
           :width: 400
           :alt: ADEME Logo

.. _`ADEME`: https://www.ademe.fr/

Development
------------

* Samuel Juhel

Contributions
---------------

All :ref:`contributions<contributions>` to the project are welcome !

Acknowledgements
------------------

I would like to thank Vincent Viguie, Fabio D'Andrea my PhD supervisors as well as Célian Colon and Alessio Ciulo for their inputs during the model implementation.
