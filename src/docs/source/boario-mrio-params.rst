.. _boario-mrio-params:

########################################
Set the MRIO table parameters
########################################

MRIO parameters cover all parameters related to the MRIO table, just as simulation parameters
they are stored in a python dictionary or JSON file,
loaded when initializing a :class:`~boario.model_base.ARIOBaseModel` or :class:`~boario.extended_models.ARIOModelPsi` instance. They are generally referred to as ``sector_params``, ``mrio_params`` or even ``mrio_name_sector_params`` (whereas simulation parameters and events
parameters are generally referred to as ``params`` and
``event_params``). This page details each parameters and their use.

* ``"monetary_unit"`` : Should be 1000000 if the table uses millions of €/$/£/...
* ``"main_inv_dur"`` : Used to make easier the change of inventory duration of most sectors when experimenting.
* ``"capital_ratio_dict"`` : Dictionary of the Value Added over Capital ratio for all sectors. Format is ``"sector_id":float``. Used to estimate the capital stock of each sector based on their VA in the MRIO.
* ``"inventories_dict"`` : Dictionary of the initial size of inventories of each sector, see :ref:`the math behind <boario-math-initial-inv>`. Format is ``"sector_id":float|Infinity``. As an example : ``"Aluminium production": 90.0`` means that **all sectors** have initially in stock an amount of products from the ``"Aluminium production"`` sector allowing them to produce during 90.0 days at initial production level. ``Infinity`` or ``"inf"`` is used for products which should not hamper production when `supply` is perturbed.

.. _contact the developer: pro@sjuhel.org

.. _github repository: https://github.com/spjuhel/BoARIO
