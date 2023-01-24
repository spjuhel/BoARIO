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
import json
import pickle
import pathlib
from typing import Optional, Union
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
from boario import DEBUGFORMATTER

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

    current_temporal_unit : int
        Tracks the number of `temporal_units` elapsed since simulation start.
        This may differs from the number of `steps` if the parameter `temporal_units_by_step` differs from 1 temporal_unit as `current_temporal_unit` is actually `step` * `temporal_units_by_step`.

    n_temporal_units_to_sim : int
        The total number of `temporal_units` to simulate.

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
    def __init__(self, params: Union[dict, str, pathlib.Path], mrio_system: Union[IOSystem, str, pathlib.Path, None] = None, mrio_params: Optional[dict] = None, modeltype:str = "ARIOPsi") -> None:
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

        # SIMULATION PARAMETER LOADING
        simulation_params = None
        params_path = None

        # CASE DICT
        if isinstance(params, dict):
            logger.info("Loading simulation parameters from dict")
            simulation_params = params

        # CASE STR|PATH
        elif isinstance(params, str):
            params_path=pathlib.Path(params)
        elif isinstance(params, pathlib.Path):
            params_path=params

        # OTHERWISE ERROR
        else:
            raise TypeError("params must be either a dict, a str or a pathlib.Path, not a {}".format(type(params)))

        # IF NOT DEFINED, then its a path to the file, or a directory with the file.
        if simulation_params is None:
            if params_path is None:
                raise ValueError("Parameters path is set yet, this is a problem.")
            if params_path.is_dir():
                simulation_params_path = params_path / 'params.json'
                if not simulation_params_path.exists():
                    raise FileNotFoundError("Simulation parameters file not found, it should be here: ",simulation_params_path.absolute())
                else:
                    with simulation_params_path.open() as f:
                        logger.info("Loading simulation parameters from {}".format(simulation_params_path))
                        simulation_params = json.load(f)
            else:
                if not params_path.exists():
                    raise FileNotFoundError("Simulation parameters file not found, it should be here: ",params_path.absolute())
                else:
                    with params_path.open() as f:
                        logger.info("Loading simulation parameters from {}".format(params_path))
                        simulation_params = json.load(f)

        # MRIO PARAMETERS LOADING
        mrio_params_loaded = None
        mrio_params_path = None
        # CASE: given as parameter
        if mrio_params is not None:
            # Is it directly a dict ?
            if isinstance(mrio_params, dict):
                logger.info("Loading MRIO parameters from dict")
                mrio_params_loaded = mrio_params
            # If it is a str, convert to path
            elif isinstance(mrio_params, str):
                mrio_params_path=pathlib.Path(mrio_params)
            # Already a path
            elif isinstance(mrio_params, pathlib.Path):
                mrio_params_path=mrio_params
            # Else error !
            else:
                raise TypeError("params must be either a dict, a str or a pathlib.Path, not a {}".format(type(params)))

            # Still in the case where it is an argument
            if mrio_params_loaded is None:
                if params_path is not None and params_path.is_dir():
                    mrio_params_path = params_path / 'mrio_params.json'
                    if not mrio_params_path.exists():
                        raise FileNotFoundError("MRIO parameters file not found, it should be here: ",mrio_params_path.absolute())
                    else:
                        with mrio_params_path.open() as f:
                            logger.info("Loading MRIO parameters from {}".format(mrio_params_path))
                            mrio_params_loaded = json.load(f)
                else:
                    if mrio_params_path is None:
                        raise ValueError("MRIO parameters are not set a this point and this is a problem.")
                    if not mrio_params_path.exists():
                        raise FileNotFoundError("MRIO parameters file not found, it should be here: ",mrio_params_path.absolute())
                    else:
                        with mrio_params_path.open() as f:
                            logger.info("Loading MRIO parameters from {}".format(mrio_params_path))
                            mrio_params_loaded = json.load(f)

        # In case it is not given as an argument :
        elif 'mrio_params_file' not in simulation_params.keys():
            raise FileNotFoundError("Unable to load MRIO parameters file from arguments and the path to it is not set in the simulation parameters file")
        else:
            mrio_params_path = pathlib.Path(simulation_params['mrio_params_file'])
            if not mrio_params_path.exists():
                logger.warning('MRIO parameters file specified in simulation params was not found: {}'.format(mrio_params_path))
            else:
                with mrio_params_path.open() as f:
                    mrio_params_loaded = json.load(f)

        # Probably useless now
        if mrio_params_loaded is None:
            raise FileNotFoundError("Couldn't load the MRIO params (tried with simulation params and the default file)")

        # LOAD MRIO system
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

        self.params = simulation_params
        self.results_storage = results_storage = pathlib.Path(self.params['output_dir']).resolve()/pathlib.Path(self.params['results_storage'])
        if not results_storage.exists():
            results_storage.mkdir(parents=True)
        if modeltype == "ARIOBase":
            logger.warning("Be aware you are using an ARIOBaseModel instance, which is mostly intended for testing and developping and not using for economic modeling (it has numerous numeric instabilities among other things). You should use ARIOModelPsi for this purpose.")
            self.model = ARIOBaseModel(mrio, mrio_params_loaded, simulation_params, results_storage)
        elif modeltype == "ARIOPsi":
            self.model = ARIOModelPsi(mrio, mrio_params_loaded, simulation_params, results_storage)
        else:
            raise NotImplementedError("""This model type is not implemented : {}
Available types are {}
            """.format(modeltype,str(["ARIOBase","ARIOPsi"])))
        self.all_events = []
        self.currently_happening_events = []
        self.events_timings = set()
        self.n_temporal_units_to_sim = simulation_params['n_temporal_units_to_sim']
        self.current_temporal_unit = 0
        self.equi = {
            (int(0),int(0),"production"):"equi",
            (int(0),int(0),"stocks"):"equi",
            (int(0),int(0),"rebuilding"):"equi"
        }
        self.n_temporal_units_simulated = 0
        self._n_checks = 0
        self._monotony_checker = 0
        self.scheme = 'proportional'
        self.has_crashed = False

        logger.info("Initialized !")

    def loop(self, progress:bool=True):
        r"""Launch the simulation loop.

        This method launch the simulation for the number of steps to simulate
        described by the attribute ``n_temporal_units_to_sim``, calling the
        :meth:`next_step` method. For convenience, it dumps the
        parameters used in the logs just before running the loop. Once the loop
        is completed, it flushes the different memmaps generated.

        Parameters
        ----------

        progress: bool, default: True
            If True show a progress bar of the loop in the console.
        """
        logger.info("Starting model loop for at most {} steps".format(self.n_temporal_units_to_sim//self.model.n_temporal_units_by_step+1))
        logger.info("One step is {}/{} of a year".format(self.model.n_temporal_units_by_step, self.model.iotable_year_to_temporal_unit_factor))
        tmp = logging.FileHandler(self.results_storage/"simulation.log")
        tmp.setLevel(logging.DEBUG)
        tmp.setFormatter(DEBUGFORMATTER)
        logger.addHandler(tmp)
        logger.info("Events : {}".format(self.all_events))
        (pathlib.Path(self.params["output_dir"]+"/"+self.params['results_storage'])/"jsons").mkdir(parents=True, exist_ok=True)
        with (pathlib.Path(self.params["output_dir"]+"/"+self.params['results_storage'])/"jsons"/"simulated_events.json").open('w') as f:
            event_dicts = [ev.event_dict for ev in self.all_events]
            json.dump(event_dicts, f, indent=4)
        if progress:
            widgets = [
                'Processed: ', progressbar.Counter('Step: %(value)d'), ' ~ ', progressbar.Percentage(), ' ', progressbar.ETA(),
            ]
            bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True)
            for _ in bar(range(0,self.n_temporal_units_to_sim,math.floor(self.params['temporal_units_by_step']))):
                #assert self.current_temporal_unit == t
                step_res = self.next_step()
                self.n_temporal_units_simulated = self.current_temporal_unit
                if step_res == 1:
                    self.has_crashed = True
                    logger.warning(f"""Economy seems to have crashed.
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    logger.warning(f"""Economy seems to have found an equilibrium
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
        else:
            for _ in range(0,self.n_temporal_units_to_sim,math.floor(self.params['temporal_units_by_step'])):
                #assert self.current_temporal_unit == t
                step_res = self.next_step()
                self.n_temporal_units_simulated = self.current_temporal_unit
                if step_res == 1:
                    self.has_crashed = True
                    logger.warning(f"""Economy seems to have crashed.
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    logger.warning(f"""Economy seems to have found an equilibrium
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break

        self.model.rebuild_demand_evolution.flush()
        self.model.final_demand_unmet_evolution.flush()
        self.model.final_demand_evolution.flush()
        self.model.io_demand_evolution.flush()
        self.model.production_evolution.flush()
        self.model.limiting_stocks_evolution.flush()
        self.model.rebuild_production_evolution.flush()
        if self.params['register_stocks']:
            self.model.stocks_evolution.flush()
        self.model.overproduction_evolution.flush()
        self.model.production_cap_evolution.flush()
        self.params['n_temporal_units_simulated'] = self.n_temporal_units_simulated
        self.params['has_crashed'] = self.has_crashed
        with (pathlib.Path(self.params["output_dir"]+"/"+self.params['results_storage'])/"jsons"/"simulated_params.json").open('w') as f:
            json.dump(self.params, f, indent=4)
        with (pathlib.Path(self.params["output_dir"]+"/"+self.params['results_storage'])/"jsons"/"equilibrium_checks.json").open('w') as f:
            json.dump({str(k): v for k, v in self.equi.items()}, f, indent=4)
        logger.info('Loop complete')
        if progress:
            bar.finish() # type: ignore (bar possibly unbound but actually not possible)

    def next_step(self, check_period : int = 182, min_steps_check : Optional[int] = None, min_failing_regions : Optional[int] = None):
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
            min_steps_check = self.n_temporal_units_to_sim // 5
        if min_failing_regions is None:
            min_failing_regions = self.model.n_regions*self.model.n_sectors // 3

        # Check if there are new events to add,
        # if some happening events can start rebuilding (if rebuildable),
        # and updates the internal model production_cap decrease and rebuild_demand
        self.check_happening_events()

        if self.params['register_stocks']:
            self.model.write_stocks(self.current_temporal_unit)
        if self.current_temporal_unit > 1:
                self.model.calc_overproduction()
        self.model.write_overproduction(self.current_temporal_unit)
        self.model.write_rebuild_demand(self.current_temporal_unit)
        self.model.write_final_demand(self.current_temporal_unit)
        self.model.write_io_demand(self.current_temporal_unit)
        constraints = self.model.calc_production(self.current_temporal_unit)
        self.model.write_limiting_stocks(self.current_temporal_unit, constraints)
        self.model.write_production(self.current_temporal_unit)
        self.model.write_kapital_lost(self.current_temporal_unit)
        self.model.write_production_max(self.current_temporal_unit)
        try:
            rebuildable_events = [ev for ev in self.currently_happening_events if ev.rebuildable]
            events_to_remove = self.model.distribute_production(self.current_temporal_unit, rebuildable_events, self.scheme)
        except RuntimeError as e:
            logger.exception("This exception happened:",e)
            return 1
        events_to_remove = events_to_remove + [ev for ev in self.currently_happening_events if ev.over]
        if events_to_remove != []:
            self.currently_happening_events = [e for e in self.currently_happening_events if e not in events_to_remove]
            for e in events_to_remove:
                logger.info("Temporal_Unit : {} ~ Event named {} that occured at {} in {} for {} damages is completely rebuilt/recovered".format(self.current_temporal_unit, e.name,e.occurrence, e.aff_regions, e.total_kapital_destroyed))

        self.model.calc_orders()
        # TODO : Redo this properly
        n_checks = self.current_temporal_unit // check_period
        if (n_checks > self._n_checks):
            self.check_equilibrium(n_checks)
            self._n_checks+=1

        self.current_temporal_unit += self.params['temporal_units_by_step']
        return 0

    def check_equilibrium(self,n_checks:int):
        if np.greater_equal(self.model.production,self.model.X_0).all():
            self.equi[(n_checks,self.current_temporal_unit,"production")] = "greater"
        elif np.allclose(self.model.production,self.model.X_0,atol=0.01):
            self.equi[(n_checks,self.current_temporal_unit,"production")] = "equi"
        else:
            self.equi[(n_checks,self.current_temporal_unit,"production")] = "not equi"

        if np.greater_equal(self.model.matrix_stock,self.model.matrix_stock_0).all():
            self.equi[(n_checks,self.current_temporal_unit,"stocks")] = "greater"
        elif np.allclose(self.model.production,self.model.X_0,atol=0.01):
            self.equi[(n_checks,self.current_temporal_unit,"stocks")] = "equi"
        else:
            self.equi[(n_checks,self.current_temporal_unit,"stocks")] = "not equi"

        if self.model.tot_rebuild_demand is None or not self.model.tot_rebuild_demand.any():
            self.equi[(n_checks,self.current_temporal_unit,"rebuilding")] = "finished"
        else:
            self.equi[(n_checks,self.current_temporal_unit,"rebuilding")] = "not finished"

    def read_events_from_list(self, events_list : list[dict]):
        """Import a list of events (as a list of dictionaries) into the model.

        Also performs various checks on the events to avoid badly written events.
        See :ref:`How to define Events <boario-events>` to understand how to write events dictionaries or JSON files.

        Parameters
        ----------
        events_list :
            List of events as dictionaries.

        """
        logger.info("Reading events from given list and adding them to the model")
        if not isinstance(events_list, list):
            if isinstance(events_list, dict):
                raise TypeError("read_events_from_list() takes a list of event dicts as an argument, not a single event dict, you might want to use read_event(your_event) instead.")
            else:
                raise TypeError("read_events_from_list() takes a list of event dicts as an argument, not a {}".format(type(events_list)))
        for ev_dic in events_list:
            self.read_event(ev_dic)

    def read_event(self, ev_dic:dict):
        if ev_dic['aff_sectors'] == 'all':
            ev_dic['aff_sectors'] = list(self.model.sectors)
        ev = Event(ev_dic)
        self.all_events.append(ev)
        self.events_timings.add(ev.occurrence)


    def reset_sim_with_same_events(self):
        """Resets the model to its initial status (without removing the events).
        """

        logger.info('Resetting model to initial status (with same events)')
        self.current_temporal_unit = 0
        self._monotony_checker = 0
        self.n_temporal_units_simulated = 0
        self.has_crashed = False
        self.model.reset_module(self.params)

    def reset_sim_full(self):
        """Resets the model to its initial status and remove all events.
        """

        self.reset_sim_with_same_events()
        logger.info('Resetting events')
        self.all_events = []
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

    def check_happening_events(self) -> None:
        for ev in self.all_events:
            if not ev.happened:
                if ((self.current_temporal_unit-self.params['temporal_units_by_step']) <= ev.occurrence <= self.current_temporal_unit):
                    logger.info("Temporal_Unit : {} ~ Shocking model with new event".format(self.current_temporal_unit))
                    logger.info("Affected regions are : {}".format(ev.aff_regions))
                    ev.happened = True
                    self.currently_happening_events.append(ev)
        for ev in self.currently_happening_events:
            ev.rebuildable = self.current_temporal_unit
            ev.recoverable = self.current_temporal_unit
            if ev.recoverable:
                ev.recovery(self.current_temporal_unit)
        self.model.update_system_from_events(self.currently_happening_events)


#    def check_current_events(self)-> None:
#        if self.current_events != []:
#            for ev in self.current_events:
#                pass
