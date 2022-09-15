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
           :alt: ADEME Logo

Where to get it ?
===================

The full source code is available on Github at: https://github.com/spjuhel/BoARIO

How does BoARIO work?
======================

In a nutshell, BoARIO takes the following inputs :

- an IO table (such as EXIOBASE3 or EORA26) in the form of an `IOSystem` object (define by the `pymrio` package)

- simulation and mrio parameters (as json files or dictionaries), which govern the simulation,

- event(s) description(s) (as json files or dictionaries), which are used as the perturbation to analyse during the simulation

in order to produce the following outputs:

- the step by step, sector by sector, region by region evolution of most of the variables involved in the simulation (production, demand, stocks, ...)

- aggregated indicators for the whole simulation (shortages duration, aggregated impacts, ...)

- more to come

Useful scripts
=================

.. warning::
   Probably deprecated !

The github repository contains a variety of `useful scripts`_ to ease the generation of input data, run simulations, produce indicators, etc. such as :

1. ``python aggreg_exio3.py exio_path.zip aggreg_path.ods sector_names.json [region_aggreg.json] -o output`` :

This script takes an EXIOBASE3 MRIO zip file, the ``input`` sheet of a libreoffice spreadsheet, a json file for sector renaming and optionally a json file for region aggregation (see examples of such files in the git repo, `here`_) and produce a pickle file (python format) of the corresponding MRIO tables directly usable by the model. ``region_aggreg.json`` must have the following form :

.. code:: json

   {
   "aggregates": {
          "original region": "new region",
          "original region": "new region",
          },
   "missing": "name for all region not named before"
   }

These scripts are mainly thought to be used for my PhD project and with the `Snakemake workflow`_ also available on the repository. (Description of this process soon !)

.. _`useful scripts`: https://github.com/spjuhel/BoARIO/tree/master/scripts
.. _`here`: https://github.com/spjuhel/BoARIO/tree/master/other
.. _`Snakemake workflow`: https://github.com/spjuhel/BoARIO/tree/master/workflow

More description to come
=============================

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

* Be the first `contributor`_ !

.. _`contributor`: https://spjuhel.github.io/BoARIO/development.html

Acknowledgements
------------------

I would like to thank Vincent Viguie, Fabio D'Andrea my PhD supervisors as well as Célian Colon for their inputs during the model implementaiton.
