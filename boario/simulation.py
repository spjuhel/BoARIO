# BoARIO : The Adaptative Regional Input Output model in python.
# Copyright (C) 2022  Samuel Juhel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
Simulation module

This module defines the Simulation object, which represent a BoARIO simulation environment.

'''

from __future__ import annotations
from contextlib import redirect_stdout
import json
import pickle
import pathlib
from typing import Union
import logging
import math
import numpy as np
import progressbar
import pymrio as pym
from pymrio.core.mriosystem import IOSystem

from boario.event import Event
from boario.model_base import ARIOBaseModel
from boario.extended_models import ARIOModelPsi
from boario import logger
from boario.logging_conf import DEBUGFORMATTER

__all__=['Simulation']

class Simulation(object):
    """Defines a simulation object with a set of parameters and an IOSystem.

    This class wraps an :class:`~boario.model_base.ARIOBaseModel` or :class:`~boario.extended_models.ARIOModelPsi`, and create the context for
    simulations using this model. It stores execution parameters as well as events perturbing
    the model.

    Attributes
    ----------
    params : dict
        Parameters to run the simulation with. If str or Path, it must lead
        to a json file containing a dictionary of the parameters.

    results_storage : pathlib.Path
        Path to store the results to.

    model : Union[ARIOBaseModel, ARIOModelPsi]
        The model to run the simulation with.

    current_day : int
        Tracks the number of `days` elapsed since simulation start.
        This may differs from the number of `steps` if the parameter `model_time_step` differs from 1 day as `current_day` is actually `step` * `model_time_step`.

    n_days_to_sim : int
        The total number of `days` to simulate.

    events : list[Event]
        The list of events to shock the model with during the simulation.

    current_events : list[Event]
        The list of events that

    Raises
    ------
    TypeError
        This error is raised when parameters files (for either the simulation or the mrio table) is not of a correct type.

    FileNotFoundError
        This error is raised when one of the required file to initialize
        the simulation was not found and should print out which one.
    """
    def __init__(self, params: Union[dict, str, pathlib.Path], mrio_system: Union[IOSystem, str, pathlib.Path, None] = None, modeltype:str = "ARIOBase") -> None:
        """Initialisation of a Simulation object uses these parameters

        Parameters
        ----------
        params : Union[dict, str, pathlib.Path]
            Parameters to run the simulation with. If str or Path, it must lead
            to a json file containing a dictionary of the parameters.
        mrio_system : Union[IOSystem, str, pathlib.Path, None], default: None
            ``pymrio.IOSystem`` to run the simulation with. If str or Path, it must
            lead to either :

            1) a pickle file of an IOSystem that was previously
            generated with the pymrio package (be careful that such files are
            not cross-system compatible) or,
            2) a directory loadable with ``pymrio.load_all`` (see `Loading, saving and exporting pymrio data <https://pymrio.readthedocs.io/en/latest/notebooks/load_save_export.html>`_).
        """
        logger.info("Initializing new simulation instance")
        super().__init__()
        if isinstance(params, str):
            params_path=pathlib.Path(params)
        if isinstance(params, pathlib.Path):
            params_path=params
            if not params_path.is_dir():
                raise FileNotFoundError("This path should be a directory containing the different simulation parameters files:", params)
            simulation_params_path = params_path / 'params.json'
            if not simulation_params_path.exists():
                raise FileNotFoundError("Simulation parameters file not found, it should be here: ",simulation_params_path.absolute())
            else:
                with simulation_params_path.open() as f:
                    logger.info("Loading simulation parameters from {}".format(simulation_params_path))
                    simulation_params = json.load(f)
        if isinstance(params, dict):
            logger.info("Loading simulation parameters from dict")
            simulation_params = params
            if simulation_params['mrio_params_file'] is None:
                logger.warn("Params given as a dict but 'mrio_params_file' does not exist. Will try with default one.")
        else:
            raise TypeError("params must be either a dict, a str or a pathlib.Path, not a %s", str(type(params)))

        mrio_params = None
        if simulation_params['mrio_params_file'] is not None:
            mrio_params_path = pathlib.Path(simulation_params['mrio_params_file'])
            if not mrio_params_path.exists():
                logger.warning('MRIO parameters file specified in simulation params was not found, trying the default one.')
            else:
                with mrio_params_path.open() as f:
                    mrio_params = json.load(f)
            if pathlib.Path('mrio_sectors_params.json').exists():
                # Default values seem to require a file from the user, shouldn't they instead be hardcoded (otherwise they are not default)?
                if simulation_params['mrio_params_file'] is not None and simulation_params['mrio_params_file'] != ('mrio_sectors_params.json'):
                    logger.warning("A json file (%s) for MRIO parameters has been found but differs from the file specified in the simulation parameters (%s)", 'mrio_sectors_params.json', simulation_params['mrio_params_file'])
                    logger.warning("Loading the json file")
                    mrio_params_path = pathlib.Path('mrio_sectors_params.json')
                    with mrio_params_path.open() as f:
                        mrio_params = json.load(f)
        if mrio_params is None:
            raise FileNotFoundError("Couldn't load the MRIO params (tried with simulation params and the default file)")

        if isinstance(mrio_system, str):
            mrio_system = pathlib.Path(mrio_system)
        if isinstance(mrio_system, pathlib.Path):
            if not mrio_system.exists():
                raise FileNotFoundError("This file does not exist: ",mrio_system)
            if mrio_system.suffix == '.pkl':
                with mrio_system.open(mode='rb') as fp:
                    mrio = pickle.load(fp)
            else:
                if not mrio_system.is_dir():
                    raise FileNotFoundError("This file should be a pickle file or a directory loadable by pymrio: ", mrio_system.absolute())
                mrio = pym.load_all(mrio_system.absolute())
        elif isinstance(mrio_system, IOSystem):
            mrio = mrio_system
        else:
            raise TypeError("mrio_system must be either a str, a pathlib.Path or an pymrio.IOSystem, not a %s", str(type(mrio_system)))

        if not isinstance(mrio, IOSystem):
            raise TypeError("At this point, mrio should be an IOSystem, not a %s", str(type(mrio)))

        self.params = params
        self.results_storage = results_storage = pathlib.Path(self.params['output_dir']+"/"+self.params['results_storage'])
        if not results_storage.exists():
            results_storage.mkdir(parents=True)
        if modeltype == "ARIOBase":
            self.model = ARIOBaseModel(mrio, mrio_params, simulation_params, results_storage)
        elif modeltype == "ARIOPsi":
            self.model = ARIOModelPsi(mrio, mrio_params, simulation_params, results_storage)
        else:
            raise NotImplementedError("""This model type is not implemented : {}
Available types are {}
            """.format(modeltype,str(["ARIOBase","ARIOPsi"])))
        self.events = []
        self.current_events = []
        self.events_timings = set()
        self.n_days_to_sim = simulation_params['n_timesteps']
        self.n_timesteps_to_sim = simulation_params['n_timesteps_to_sim']

        self.current_day = 0
        self.n_steps_simulated = 0
        self._monotony_checker = 0
        self.scheme = 'proportional'
        self.has_crashed = False
        logger.info("Initialized !")

    def loop(self, progress:bool=True):
        r"""Launch the simulation loop.

        This method launch the simulation for the number of steps to simulate
        described by the attribute ``n_days_to_sim``, calling the
        :meth:`next_step` method. For convenience, it dumps the
        parameters used in the logs just before running the loop. Once the loop
        is completed, it flushes the different memmaps generated.

        Parameters
        ----------

        progress: bool, default: True
            If True show a progress bar of the loop in the console.
        """
        logger.info("Starting model loop for at most {} steps".format(self.n_days_to_sim//self.model.n_days_by_step+1))
        logger.info("One step is {} day(s)".format(self.model.n_days_by_step))
        tmp = logging.FileHandler(self.results_storage/"simulation.log")
        tmp.setLevel(logging.DEBUG)
        tmp.setFormatter(DEBUGFORMATTER)
        logger.addHandler(tmp)
        with (pathlib.Path(self.params["output_dir"]+"/"+self.params['results_storage'])/"simulated_events.json").open('w') as f:
            json.dump(self.events, f, indent=4)
        if progress:
            widgets = [
                'Processed: ', progressbar.Counter('Step: %(value)d'), ' ~ ', progressbar.Percentage(), ' ', progressbar.ETA(),
            ]
            bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True)
            for t in bar(range(0,self.n_days_to_sim,math.floor(self.params['model_time_step']))):
                #assert self.current_day == t
                step_res = self.next_step()
                self.n_steps_simulated = self.current_day
                if step_res == 1:
                    self.has_crashed = True
                    logger.warning(f"""Economy seems to have crashed.
                    - At step : {self.current_day}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    logger.warning(f"""Economy seems to have found an equilibrium
                    - At step : {self.current_day}
                    """
                    )
                    break
        else:
            for t in range(0,self.n_days_to_sim,math.floor(self.params['model_time_step'])):
                #assert self.current_day == t
                step_res = self.next_step()
                self.n_steps_simulated = self.current_day
                if step_res == 1:
                    self.has_crashed = True
                    logger.warning(f"""Economy seems to have crashed.
                    - At step : {self.current_day}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    logger.warning(f"""Economy seems to have found an equilibrium
                    - At step : {self.current_day}
                    """
                    )
                    break

        self.model.rebuild_demand_evolution.flush()
        self.model.final_demand_unmet_evolution.flush()
        self.model.classic_demand_evolution.flush()
        self.model.production_evolution.flush()
        self.model.limiting_stocks_evolution.flush()
        self.model.rebuild_production_evolution.flush()
        if self.params['register_stocks']:
            self.model.stocks_evolution.flush()
        self.model.overproduction_evolution.flush()
        self.model.production_cap_evolution.flush()
        self.params['n_timesteps_simulated'] = self.n_steps_simulated
        self.params['has_crashed'] = self.has_crashed
        with (pathlib.Path(self.params["output_dir"]+"/"+self.params['results_storage'])/"simulated_params.json").open('w') as f:
            json.dump(self.params, f, indent=4)
        logger.info('Loop complete')
        if progress:
            bar.finish() # type: ignore (bar possibly unbound but actually not possible)

    def next_step(self, check_period : int = 10, min_steps_check : int = None, min_failing_regions : int = None):
        """Advance the model run by one step.

        This method wraps all computations and logging to proceed to the next
        step of the simulation run. First it checks if an event is planned to
        occur at the current step and if so, shocks the model with the
        corresponding event. Then it :

        1) Computes the production required by demand (using :meth:`~boario.model_base.ARIOBaseModel.calc_prod_reqby_demand`)

        2) Computes the production capacity vector of the current step (using :meth:`~boario.model_base.ARIOBaseModel.calc_production_cap`)

        3) Computes the actual production vector for the step (using :meth:`~boario.model_base.ARIOBaseModel.calc_production`)

        4) Distribute the actual production towards the different demands (intermediate, final, rebuilding) and the changes in the stocks matrix (using :meth:`~boario.model_base.ARIOBaseModel.distribute_production`)

        5) Computes the orders matrix for the next step (using :meth:`~boario.model_base.ARIOBaseModel.calc_orders`)

        6) Computes the new overproduction vector for the next step (using :meth:`~boario.model_base.ARIOBaseModel.calc_overproduction`)

        See :ref:`Mathematical background <boario-math>` section for more in depth information.

        Parameters
        ----------

        check_period : int, default: 10
            [Deprecated] Number of steps between each crash/equilibrium checking.

        min_steps_check : int, default: None
            [Deprecated] Minimum number of steps before checking for crash/equilibrium. If none, it is set to a fifth of the number of steps to simulate.

        min_failing_regions : int, default: None
            [Deprecated] Minimum number of 'failing regions' required to consider the economy has 'crashed' (see :func:`~ario3.mriosystem.MrioSystem.check_crash`:).

        """
        if min_steps_check is None:
            min_steps_check = self.n_days_to_sim // 5
        if min_failing_regions is None:
            min_failing_regions = self.model.n_regions*self.model.n_sectors // 3

        new_events = [(e_id,e) for e_id, e in enumerate(self.events) if ((self.current_day-self.params['model_time_step']) <= e.occurence_time <= self.current_day)]
        for (e_id,e) in new_events:
            # print(e)
            if e not in self.current_events:
                self.current_events.append(e)
                self.shock(e_id)

        if self.current_events != []:
            self.update_events()
            self.model.update_system_from_events(self.current_events)
        if self.params['register_stocks']:
            self.model.write_stocks(self.current_day)
        self.model.calc_prod_reqby_demand(self.current_events)
        if self.current_day > 1:
                self.model.calc_overproduction()
        self.model.write_overproduction(self.current_day)
        self.model.write_rebuild_demand(self.current_day)
        self.model.write_classic_demand(self.current_day)
        self.model.calc_production_cap()
        constraints = self.model.calc_production(self.current_day)
        self.model.write_limiting_stocks(self.current_day, constraints)
        self.model.write_production(self.current_day)
        self.model.write_production_max(self.current_day)
        try:
            events_to_remove = self.model.distribute_production(self.current_day, self.current_events, self.scheme)
        except RuntimeError as e:
            logger.exception("This exception happened:",e)
            return 1
        if events_to_remove != []:
            self.current_events = [e for e in self.current_events if e not in events_to_remove]
            for e in events_to_remove:
                logger.info("Day : {} ~ Event named {} that occured at {} in {} for {} damages is completely rebuilt".format(self.current_day, e.name,e.occurence_time, e.aff_regions, e.q_damages))
        self.model.calc_orders(self.current_events)
        # TODO : Redo this properly
        if False:
            if self.current_day > min_steps_check and (self.current_day % check_period == 0):
                if self.model.check_crash() >  min_failing_regions:
                    return 1
                if self.current_day >= self.params['min_duration'] and self.model.rebuilding_demand.sum() == 0 and self.model.check_production_eq_soft(self.current_day, period = check_period):
                    self._monotony_checker +=1
                else:
                    self._monotony_checker = 0
        self.current_day+= self.params['model_time_step']
        return 0

    def update_events(self):
        """Update events status

        This method cycles through the events defines in the ``current_events`` attribute and sets their ``rebuildable`` attribute.
        An event is considered rebuildable if its ``duration`` is over (i.e. the number of days elapsed since it shocked the model is greater than ``occurence_time`` + ``duration``).
        This method also logs the moment an event starts rebuilding.
        """
        for e in self.current_events:
            already_rebuilding = e.rebuildable
            e.rebuildable = (e.occurence_time + e.duration) <= self.current_day
            if e.rebuildable and not already_rebuilding:
                logger.info("Day : {} ~ Event named {} that occured at {} in {} for {} damages has started rebuilding".format(self.current_day,e.name,e.occurence_time, e.aff_regions, e.q_damages))

    def read_events_from_list(self, events_list : list[dict]):
        """Import a list of events (as a list of dictionaries) into the model.

        Also performs various checks on the events to avoid badly written events.
        See :ref:`How to define Events <boario-events>` to understand how to write events dictionaries or JSON files.

        Parameters
        ----------
        events_list :
            List of events as dictionnaries.

        Examples
        --------

        .. include:: ../../../../api-examples/simulation/example-read_events_from_list.rstinc

        """
        logger.info("Reading events from given list and adding them to the model")
        for ev_dic in events_list:
            if ev_dic['aff-sectors'] == 'all':
                ev_dic['aff-sectors'] = list(self.model.sectors)
            ev = Event(ev_dic,self.model)
            ev.check_values(self)
            self.events.append(ev)
            self.events_timings.add(ev_dic['occur'])

    def shock(self, event_to_add_id:int):
        """Shocks the model with an event.

        Sets the rebuilding demand and the share of production allocated toward
        it in the model.

        First, if multiple regions are affected, it computes the vector of how damages are distributed across these,
        using the ``dmg_distrib_across_regions`` attribute of the :class:`~boario.event.Event` object.
        Then it computes the vector of how regional damages are distributed across affected sectors using
        the ``dmg_distrib_across_sector`` and ``dmg_distrib_across_sector_type`` attributes.
        This ``n_regions`` * ``n_sectors`` sized vector hence stores the damage (i.e. capital destroyed) for all industries.

        This method also computes the `rebuilding demand` matrix, the demand addressed to the rebuilding
        industries consequent to the shock, and sets the initial vector of production share dedicated to rebuilding.

        See :ref:`How to define Events <boario-events>` for further detail on how to parameter these distribution.

        Parameters
        ----------
        event_to_add_id : int
            The id (rank it the ``events`` list) of the event to shock the model with.

        Raises
        ------
        ValueError
            Raised if the production share allocated to rebuilding (in either
            the impacted regions or the others) is not in [0,1].
        """
        logger.info("Day : {} ~ Shocking model with new event".format(self.current_day))
        logger.info("Affected regions are : {}".format(self.events[event_to_add_id].aff_regions))
        impacted_region_prod_share = self.params['impacted_region_base_production_toward_rebuilding']
        RoW_prod_share = self.params['row_base_production_toward_rebuilding']
        self.events[event_to_add_id].check_values(self)
        if (impacted_region_prod_share > 1.0 or impacted_region_prod_share < 0.0):
            raise ValueError("Impacted production share should be in [0.0,1.0], (%f)", impacted_region_prod_share)
        if (RoW_prod_share > 1.0 or RoW_prod_share < 0.0):
            raise ValueError("RoW production share should be in [0.0,1.0], (%f)", RoW_prod_share)
        regions_idx = np.arange(self.model.regions.size)
        aff_regions_idx = np.searchsorted(self.model.regions, self.events[event_to_add_id].aff_regions)
        n_regions_aff = aff_regions_idx.size
        aff_sectors_idx = np.searchsorted(self.model.sectors, self.events[event_to_add_id].aff_sectors)
        n_sectors_aff = aff_sectors_idx.size
        aff_industries_idx = np.array([self.model.n_sectors * ri + si for ri in aff_regions_idx for si in aff_sectors_idx]) # type: ignore (aff_regions_idx not considered as array)
        q_dmg = self.events[event_to_add_id].q_damages / self.model.monetary_unit
        logger.info("Damages are {} times {} [unit (ie $/€/£)]".format(q_dmg,self.model.monetary_unit))
        # print(f'''
        # regions_idx = {regions_idx},
        # aff_regions_idx = {aff_regions_idx}
        # n_regions_aff = {aff_regions_idx.size}
        # aff_sectors_idx = {aff_sectors_idx}
        # n_sectors_aff = {aff_sectors_idx.size}
        # aff_industries_idx = {aff_industries_idx}
        # q_dmg = {q_dmg}
        # ''')

        # DAMAGE DISTRIBUTION ACROSS REGIONS
        if self.events[event_to_add_id].dmg_distrib_across_regions is None:
            q_dmg_regions = np.array([1.0]) * q_dmg
        elif type(self.events[event_to_add_id].dmg_distrib_across_regions) == list:
            q_dmg_regions = np.array(self.events[event_to_add_id].dmg_distrib_across_regions) * q_dmg
        elif type(self.events[event_to_add_id].dmg_distrib_across_regions) == str and self.events[event_to_add_id].dmg_distrib_across_regions == 'shared':
            q_dmg_regions = np.full(shape=aff_regions_idx.shape,fill_value=q_dmg/aff_regions_idx.size)
        else:
            raise ValueError("This should not happen")
        q_dmg_regions = q_dmg_regions.reshape((n_regions_aff,1))

        # DAMAGE DISTRIBUTION ACROSS SECTORS
        if self.events[event_to_add_id].dmg_distrib_across_sectors_type == "gdp":
            shares = self.model.gdp_share_sector.reshape((self.model.n_regions,self.model.n_sectors))
            if shares[aff_regions_idx][:,aff_sectors_idx].sum(axis=1)[0] == 0:
                raise ValueError("The sum of the affected sectors value added is 0 (meaning they probably don't exist in this regions)")
            q_dmg_regions_sectors = q_dmg_regions * (shares[aff_regions_idx][:,aff_sectors_idx]/shares[aff_regions_idx][:,aff_sectors_idx].sum(axis=1)[:,np.newaxis])
        elif self.events[event_to_add_id].dmg_distrib_across_sectors is None:
            q_dmg_regions_sectors = q_dmg_regions
        elif type(self.events[event_to_add_id].dmg_distrib_across_sectors) == list:
            q_dmg_regions_sectors = q_dmg_regions * np.array(self.events[event_to_add_id].dmg_distrib_across_sectors)
        elif type(self.events[event_to_add_id].dmg_distrib_across_sectors) == str and self.events[event_to_add_id].dmg_distrib_across_sectors == 'GDP':
            shares = self.model.gdp_share_sector.reshape((self.model.n_regions,self.model.n_sectors))
            q_dmg_regions_sectors = q_dmg_regions * (shares[aff_regions_idx][:,aff_sectors_idx]/shares[aff_regions_idx][:,aff_sectors_idx].sum(axis=1)[:,np.newaxis])
        else:
            raise ValueError("damage <-> sectors distribution %s not implemented", self.events[event_to_add_id].dmg_distrib_across_sectors)

        rebuilding_sectors_idx = np.searchsorted(self.model.sectors, np.array(list(self.events[event_to_add_id].rebuilding_sectors.keys())))
        rebuilding_industries_idx = np.array([self.model.n_sectors * ri + si for ri in aff_regions_idx for si in rebuilding_sectors_idx]) # type: ignore (aff_regions_idx not considered as array)
        rebuilding_industries_RoW_idx = np.array([self.model.n_sectors * ri + si for ri in regions_idx if ri not in aff_regions_idx for si in rebuilding_sectors_idx])
        rebuild_share = np.array([self.events[event_to_add_id].rebuilding_sectors[k] for k in sorted(self.events[event_to_add_id].rebuilding_sectors.keys())])
        rebuilding_demand = np.outer(rebuild_share, q_dmg_regions_sectors)
        new_rebuilding_demand = np.full(self.model.Z_0.shape, 0.0)
        # build the mask of rebuilding sectors (worldwide)
        mask = np.ix_(np.union1d(rebuilding_industries_RoW_idx, rebuilding_industries_idx), aff_industries_idx)
        new_rebuilding_demand[mask] = self.model.Z_distrib[mask] * np.tile(rebuilding_demand, (self.model.n_regions,1)) # type: ignore (strange __getitem__ warning)
        #new_rebuilding_demand[] = q_dmg_regions_sectors * rebuild_share.reshape(rebuilding_sectors_idx.size,1)
        new_prod_max_toward_rebuilding = np.full(self.model.production.shape, 0.0)
        new_prod_max_toward_rebuilding[rebuilding_industries_idx] = impacted_region_prod_share
        new_prod_max_toward_rebuilding[rebuilding_industries_RoW_idx] = RoW_prod_share
        # TODO : Differentiate industry losses and households losses
        self.events[event_to_add_id].industry_rebuild = new_rebuilding_demand

        # Currently unused :
        #self.events[event_to_add_id].production_share_allocated = new_prod_max_toward_rebuilding

        #self.model.update_kapital_lost()

    def reset_sim_with_same_events(self):
        """Resets the model to its initial status (without removing the events).
        """

        logger.info('Resetting model to initial status (with same events)')
        self.current_day = 0
        self._monotony_checker = 0
        self.n_steps_simulated = 0
        self.has_crashed = False
        self.model.reset_module(self.params)

    def reset_sim_full(self):
        """Resets the model to its initial status and remove all events.
        """

        self.reset_sim_with_same_events()
        logger.info('Resetting events')
        self.events = []
        self.events_timings = set()

    def update_params(self, new_params:dict):
        """Update the parameters of the model.

        Replace the ``params`` attribute with ``new_params`` and logs the update.
        This method also checks if the directory specified to save the results exists and create it otherwise.

        .. warning::
            Be aware this method calls :meth:`~boario.model_base.ARIOBaseModel.update_params`, which resets the memmap files located in the results directory !

        Parameters
        ----------
        new_params : dict
            New dictionnary of parameters to use.

        """
        logger.info('Updating model parameters')
        self.params = new_params
        results_storage = pathlib.Path(self.params['output_dir']+"/"+self.params['results_storage'])
        if not results_storage.exists():
            results_storage.mkdir(parents=True)
        self.model.update_params(self.params)

    def write_index(self, index_file:Union[str, pathlib.Path]):
        """Write the index of the dataframes used in the model in a json file.

        See :meth:`~boario.model_base.ARIOBaseModel.write_index` for a more detailed documentation.

        Parameters
        ----------
        index_file : Union[str, pathlib.Path]
            name of the file to save the indexes to.

        """

        self.model.write_index(index_file)

    def read_events(self, events_file:Union[str, pathlib.Path]):
        """Read events from a json file.

        Parameters
        ----------
        events_file :
            path to a json file

        Raises
        ------
        FileNotFoundError
            If file does not exist

        """
        logger.info("Reading events from {} and adding them to the model".format(events_file))
        if isinstance(events_file,str):
            events_file = pathlib.Path(events_file)
        elif not isinstance(events_file,pathlib.Path):
            raise TypeError("Given index file is not an str or a Path")
        if not events_file.exists():
            raise FileNotFoundError("This file does not exist: ",events_file)
        else:
            with events_file.open('r') as f:
                events = json.load(f)
        if isinstance(events,list):
            self.read_events_from_list(events)
        else:
            self.read_events_from_list([events])
