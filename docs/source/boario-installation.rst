###############
Installation
###############

.. _installation:

Install via pip
==================

There is a pip package available, which you can install with:

.. code-block:: console

   pip install boario

There is also a conda recipe on conda-forge, which you can install with:

.. code-block:: console

   conda install -c conda-forge boario


Otherwise the source-code is available at the GitHub repo: https://github.com/spjuhel/BoARIO

If you encounter any problems while installing the package, please `raise an issue`_
on the repository, or `contact the developer`_.

.. _raise an issue: https://github.com/spjuhel/BoARIO/issues/new

.. _contact the developer: pro@sjuhel.org

Install from a specific branch
=================================

Note that you can install BoARIO directly from a branch of its GitHub repository using:

.. code-block:: console

   pip install git+https://github.com/spjuhel/boario.git@branch-name

Requirements
===============

BoARIO is currently tested against python ``3.9, 3.10, 3.11``, on Linux, Windows and MacOs.
However be aware the testing suite is still imperfect at this point.

Installing via pip or conda should install all the required dependencies.

You can also check the ``pyproject.toml`` file in the source code to see all current dependencies.

If you are experiencing difficulties in setting up a working environment feel
free to open an issue on the Github repository or to contact the developer.
