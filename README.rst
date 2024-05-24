.. role:: pythoncode(code)
   :language: python

#######
BoARIO
#######
|build-status| |black| |contribute| |licence| |pypi| |pythonv| |joss|

.. |build-status| image:: https://img.shields.io/github/actions/workflow/status/spjuhel/boario/CI.yml
   :target: https://github.com/spjuhel/BoARIO/actions/workflows/CI.yml
   :alt: GitHub Actions Workflow Status
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000
   :target: https://github.com/psf/black
   :alt: Code Style - Black
.. |contribute| image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
   :target: https://github.com/spjuhel/BoARIO/issues
   :alt: Contribution - Welcome
.. |licence| image:: https://img.shields.io/badge/License-GPLv3-blue
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: Licence - GPLv3
.. |pypi| image:: https://img.shields.io/pypi/v/boario
   :target: https://pypi.org/project/boario/
   :alt: PyPI - Version
.. |pythonv| image:: https://img.shields.io/pypi/pyversions/boario
   :target: https://pypi.org/project/boario/
   :alt: PyPI - Python Version
.. |joss| image:: https://joss.theoj.org/papers/71386aa01a292ecff8bafe273b077701/status.svg
   :target: https://joss.theoj.org/papers/71386aa01a292ecff8bafe273b077701
   :alt: Joss Status

`BoARIO` : The Adaptative Regional Input Output model in python.

.. _`Documentation Website`: https://spjuhel.github.io/BoARIO/boario-what-is.html

Disclaimer
===========

Indirect impact modeling is tied to a lot of uncertainties and complex dynamics.
Any results produced with `BoARIO` should be interpreted with great care. Do not
hesitate to contact the author when using the model !

What is BoARIO ?
=================

BoARIO, is a python implementation project of the Adaptative Regional Input Output (ARIO) model [`Hal13`_].

Its objectives are to give an accessible and inter-operable implementation of ARIO, as well as tools to visualize and analyze simulation outputs and to
evaluate the effects of many parameters of the model.

This implementation would not have been possible without the `Pymrio`_ module and amazing work of [`Sta21`_].

It is still an ongoing project (in parallel with a PhD project).

.. _`Sta21`: https://openresearchsoftware.metajnl.com/articles/10.5334/jors.251/
.. _`Hal13`: https://doi.org/10.1111/j.1539-6924.2008.01046.x
.. _`Pymrio`: https://pymrio.readthedocs.io/en/latest/intro.html

You can find most academic literature using ARIO or related models `here <https://spjuhel.github.io/BoARIO/boario-references.html>`_


What is ARIO ?
===============

ARIO stands for Adaptive Regional Input-Output. It is an hybrid input-output / agent-based economic model,
designed to compute indirect costs from economic shocks. Its first version dates back to 2008 and has originally
been developed to assess the indirect costs of natural disasters [`Hal08`_].

In ARIO, the economy is modelled as a set of economic sectors and a set of regions.
Each economic sector produces its generic product and draws inputs from an inventory.
Each sector answers to a total demand consisting of a final demand (household consumption,
public spending and private investments) of all regions (local demand and exports) and
intermediate demand (through inputs inventory resupply). An initial equilibrium state of
the economy is built based on multi-regional input-output tables (MRIOTs).

For a more detailed description, please refer to the `Mathematical documentation`_ of the model.

Multi-Regional Input-Output tables
-------------------------------------

Multi-Regional Input-Output tables (MRIOTs) are comprehensive economic data sets
that capture inter-regional trade flows, production activities, and consumption
patterns across different regions or countries. These tables provide a detailed
breakdown of the flows of goods and services between industries within each
region and between regions themselves. MRIOTs are constructed through a
combination of national or regional input-output tables, international trade
data, and other relevant economic statistics. By integrating data from multiple
regions, MRIOTs enable the analysis of global supply chains, international trade
dependencies, and the estimation of economic impacts across regions. However,
they also come with limitations, such as data inconsistencies across regions,
assumptions about trade patterns and production technologies, and the challenge
of ensuring coherence and accuracy in the aggregation of data from various
sources.

.. _`Mathematical documentation`: https://spjuhel.github.io/BoARIO/boario-math.html

.. _`Hal08`: https://doi.org/10.1111/risa.12090

Where to get BoARIO ?
==========================

You can install BoARIO from ``pip`` with:

.. code:: console

   pip install boario

Or from ``conda-forge`` using conda (or mamba):

.. code:: console

   conda install -c conda-forge boario


The full source code is also available on Github at: https://github.com/spjuhel/BoARIO

More info in the `installation <https://spjuhel.github.io/BoARIO/boario-installation.html>`_ page of the documentation.

How does BoARIO work?
=========================

In a nutshell, BoARIO takes the following inputs :

- a (possibly Environmentally Extended) Multi-Regional IO table (such as `EXIOBASE 3`_ or `EORA26`_) in the form of an ``pymrio.IOSystem`` object, using the `Pymrio`_ python package. Please reference the `Pymrio documentation <https://github.com/IndEcol/pymrio>`_ for details on methods available to pymrio objects.

- multiple parameters which govern the simulation,

- event(s) description(s), which are used as the perturbation to analyse during the simulation

And produces the following outputs:

- the step by step, sector by sector, region by region evolution of most of the variables involved in the simulation (`production`, `demand`, `stocks`, ...)

- aggregated indicators for the whole simulation (`shortages duration`, `aggregated impacts`, ...)

.. _`EXIOBASE 3`: https://www.exiobase.eu/
.. _`EORA26`: https://worldmrio.com/eora26/

Example of use
=================

See `Boario quickstart <https://spjuhel.github.io/BoARIO/boario-tutorials.html>`_.

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

** Samuel Juhel (pro@sjuhel.org)

Contributions
---------------

All `contributions <https://spjuhel.github.io/BoARIO/development.html>`_ to the project are welcome !

Acknowledgements
------------------

I would like to thank Vincent Viguie, Fabio D'Andrea my PhD supervisors as well as Célian Colon, Alessio Ciulo and Adrien Delahais
for their inputs during the model implementation.
