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
           :width: 800
           :alt: ADEME Logo

References
-----------

[AH19] Andre F. T. Avelino and Geoffray J. D. Hewings. “The Challenge of Estimating the Impact of Disasters: Many Approaches, Many Limitations and a Compromise”. In: Advances in Spatial and Economic Modeling of Disaster Impacts. Ed. by Yasuhide Okuyama and Adam Rose. Advances in Spatial Science. Springer International Publishing, 2019. Chap. 7, pp. 163–189. doi: 10.1007/978-3-030-16237-5.

[CHR19] Célian Colon, S. Hallegatte, and Julie Rozenberg. Transportation and Supply Chain Resilience in the United Republic of Tanzania: Assessing the Supply-Chain Impacts of Disaster-Induced Transportation Disruptions. 00001. World Bank, June 2019. doi: 10.1596/31909.

[Gua+20] Dabo Guan et al. “Global Supply-Chain Effects of Covid-19 Control Measures”. In: Nature Human Behaviour 4.6 (2020), pp. 577–587. doi: 10.1038/s41562-020-0896-8.

[Hal+11] Stéphane Hallegatte et al. “Assessing Climate Change Impacts, Sea Level Rise and Storm Surge Risk in Port Cities: a Case Study on Copenhagen”. In: Climatic Change 104.1 (2011), pp. 113–137. doi: 10.1007/s10584-010-9978-3.

[Hal08] S. Hallegatte. “An Adaptive Regional Input-Output Model and Its Application to the Assessment of the Economic Cost of Katrina”. In: Risk Analysis 28.3 (June 2008). 00483, pp. 779–799. doi: 10.1111/j.1539-6924.2008.01046.x.

[Hal13] Stéphane Hallegatte. “Modeling the Role of Inventories and Heterogeneity in the Assessment of the Economic Costs of Natural Disasters”. In: Risk Analysis 34.1 (2013), pp. 152–167. doi: 10.1111/risa.12090.

[HHT12] Fanny Henriet, Stéphane Hallegatte, and Lionel Tabourier. “Firm-Network Characteristics and Economic Robustness To Natural Disasters”. In: Journal of Economic Dynamics and Control 36.1 (2012), pp. 150–167. doi: 10.1016/j.jedc.2011.10.001.

[Kok+15] E. E. Koks et al. “Integrated Direct and Indirect Flood Risk Modeling: Development and Sensitivity Analysis”. In: Risk Analysis 35.5 (2015), pp. 882–900. doi: 10.1111/risa.12300.

[Ran+10] Nicola Ranger et al. “An Assessment of the Potential Impact of Climate Change on Flood Risk in Mumbai”. In: Climatic Change 104.1 (2010), pp. 139–167. doi: 10.1007/s10584-010-9979-2.

[Wan+18] Cailin Wang et al. “Quantifying the Spatial Ripple Effect of the Bohai Sea Ice Disaster in the Winter of 2009/2010 in 31 Provinces of China”. In: Geomatics, Natural Hazards and Risk 9.1 (2018), pp. 986–1005. doi: 10.1080/19475705.2018.1489312.

[Wan+20] Daoping Wang et al. “Economic Footprint of California Wildfires in 2018”. In: Nature Sustainability 4.3 (2020), pp. 252–260. doi: 10.1038/s41893-020-00646-7.

[Zha+18] Zhengtao Zhang et al. “Analysis of the Economic Ripple Effect of the United States on the World Due To Future Climate Change”. In: Earth’s Future 6.6 (2018), pp. 828–840. doi: 10.1029/2018ef000839.

[Zha+21] Zhengtao Zhang et al. “Analysis of the Impact of Dynamic Economic Resilience on Post-Disaster Recovery ”Secondary Shock” and Sustainable Improvement of System Performance”. In: Safety Science 144.nil (2021), p. 105443. doi: 10.1016/j.ssci.2021.105443.


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

Example of use
================

In this commented example, we run the model with one simple
event using 'usual' parameters, and compute the aggregated indicators of the
simulation. We suppose we run the following script from ``~/boario/``

.. code:: python

    import sys
    # insert boario into python path (There is no installer yet)
    sys.path.insert(1, '../')

    # We import the base of the model
    import boario.simulation as sim
    # We also import the indicators module
    from boario.indicators import Indicators
    import pathlib

    # We instantiate a dictionary with the parameters
    # (it is also possible to use a json file)

    params = {
        # The name of the working directory to use (relative to current wd)
        "storage_dir": "storage",
        # The directory to use to store results (relative to storage_dir)
        # i.e. here, the model will look for files in ~/boario/storage/ and
        # store results in ~/boario/storage/results/
        "results_storage": "results",
        "bool_run_detailled": True,
        # This tells the model to register the evolution of the stocks
        # of every industry (the file can be quite large (2Gbytes+ for
        # a 365 days simulation with exiobase))
        "register_stocks": True,
        # Parameters of the model (we detail these in the documentation)
       "psi_param": 0.9,
       "model_time_step": 1,
       "timestep_dividing_factor": 365,
       "inventory_restoration_time": 90,
       "alpha_base": 1.0,
       "alpha_max": 1.25,
       "alpha_tau": 365,
       "rebuild_tau": 60,
       "n_timesteps": 365,
       "min_duration": (365 // 100) * 25,
       "impacted_region_base_production_toward_rebuilding": 0.001,
       "row_base_production_toward_rebuilding": 0.0,
       "mrio_params_file":"mrio_params.json"
    }

    # Here we define the event perturbing the simulation
    event = {
        # The list of affected regions (their index in the mrio table)
        # Here we just have France
        "aff-regions": ["FR"],
        # The list of affected sectors
        # (here we specify that all sectors are impacted)
        "aff-sectors": "all",
        # The shares of the damages distributed between regions
        # (1 as we have only one region)
        "dmg-distrib-regions": [ 1 ],
        # The type of distribution of damages for the sectors
        # (more on that in the doc)
        "dmg-distrib-sectors-type": "gdp",
        # 'gdp' distribution doesn't require this parameter to be set
        "dmg-distrib-sectors": [],
        # The duration of the event.
        "duration": 1,
        # A name for the event (usefull when simulating multiple events)
        "name": "0",
        # The step at which the event shall occur during the simulation
        "occur": 5,
        # The quantity of damages caused by the event (in IO table monetary)
        "q_dmg":100000000,
        # The sectors mobilised to answer the rebuilding demand
        # and the share of the demand they answer
        "rebuilding-sectors": {
            "Construction (45)":0.15,
            "Manufacture of machinery and equipment n.e.c. (29)" : 0.20,
            "Manufacture of furniture; manufacturing n.e.c. (36)" : 0.20,
            "Manufacture of office machinery and computers (30)": 0.15,
        }
    }

    # We load the mrio table from a pickle file (created with the help of the
    # pymrio module, more on that in the doc)
    mrio_path = pathlib.Path(params['storage_dir'])/"mrio.pkl"

    # We initiate a model instance ...
    model = sim.Simulation(mrio_path, params)

    # ... add the list of events (just one here) to the model ...
    model.read_events_from_list([event])

    # ... and launch the simulation with :
    model.loop()

    # Once the simulation is over we can compute some indicators :
    indic = Indicators.from_storage_path(
                                         pathlib.Path(sim_params['storage_dir']),
                                         params=sim_params
    )
    indic.update_indicators()
    indic.write_indicators()

This script will produce files in ``~/boario/storage/results/`` :

 - ``simulated_events.json`` : A json record of the events that were simulated
   during the loop.

 - ``indicators.json`` : A json record (produced by :func:`~boario.indicators.Indicators.write_indicators`)
   of the computed indicators.

 - ``record`` files. These are :py:class:`numpy.memmap`:
   of the different recorded variables.

Record files
------------

You may read these directly into a numpy array with :

.. code:: python

    np.memmap("results/+record_name+_record",
               mode='r+',
               dtype='float64',
               shape=(t,n_sectors*n_regions)
    )

Where ``shape`` is the shape mentioned afterward.

   1. ``classic_demand`` : the sum of intermediate and final demand addressed to
   each industries. Its shape is ``(n_timesteps, n_sectors*n_regions)``

   2. ``iotable_XVA`` : the realised production of each industry. Its shape is
   ``(n_timesteps, n_sectors*n_regions)``

   3. ``iotable_X_max`` : the production capacity of each industry. Its shape is
   ``(n_timesteps, n_sectors*n_regions)``

   4. ``overprod_vector`` : the overproduction scaling of each industry. Its
   shape is ``(n_timesteps, n_sectors*n_regions)``

   5. ``rebuild_demand`` : the additional direct demand created by the event
   for rebuilding, for each industry.
   Its shape is ``(n_timesteps, n_sectors*n_regions)``

   6. ``rebuild_prod`` : the part of production attributed to rebuilding, for each
   industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

   7. ``final_demand_unmet`` : the final demand that was not met due to rationing,
   for each industry. Its shape is ``(n_timesteps, n_sectors*n_regions)``

   8. ``stocks`` : the stocks of each input for each industry.
   Its shape is ``(n_timesteps*n_sectors, n_sectors*n_regions)``.
   Note that this file is not created if ``register_stocks`` is set to ``False``
   in the simulation parameters.

   9. ``limiting_stocks`` : a boolean matrix, telling for each input and for each
   industry if the stock is limiting for production.
   Its shape is ``(n_timesteps*n_sectors, n_sectors*n_regions)``.
   Reading this array directly require to change the dtype
   to 'bool' in the above command.

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
