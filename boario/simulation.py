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

"""
Simulation module

This module defines the Simulation object, which represent a BoARIO simulation environment.

"""

from __future__ import annotations

import json
import logging
import math
import pathlib
import tempfile
from pprint import pformat
from typing import Optional, Union

import numpy as np
import pandas as pd
import progressbar

from boario import DEBUGFORMATTER
from boario import logger
from boario.event import *
from boario.extended_models import *
from boario.model_base import ARIOBaseModel
from boario.utils.misc import CustomNumpyEncoder, TempMemmap, sizeof_fmt, print_summary

__all__ = ["Simulation"]


class Simulation:
    """Defines a simulation object with a set of parameters and an IOSystem.

    This class wraps a :class:`~boario.model_base.ARIOBaseModel` or descendant, and create the context for
    simulations using this model. It stores execution parameters as well as events perturbing
    the model.
    """

    __possible_records = [
        "production_realised",
        "production_capacity",
        "final_demand",
        "intermediate_demand",
        "rebuild_demand",
        "overproduction",
        "final_demand_unmet",
        "rebuild_prod",
        "inputs_stocks",
        "limiting_inputs",
        "productive_capital_to_recover",
    ]

    __file_save_array_specs = {
        "production_realised": (
            "float64",
            "_production_evolution",
            "industries",
            np.nan,
        ),
        "production_capacity": (
            "float64",
            "_production_cap_evolution",
            "industries",
            np.nan,
        ),
        "final_demand": ("float64", "_final_demand_evolution", "industries", np.nan),
        "intermediate_demand": (
            "float64",
            "_io_demand_evolution",
            "industries",
            np.nan,
        ),
        "rebuild_demand": (
            "float64",
            "_rebuild_demand_evolution",
            "industries",
            np.nan,
        ),
        "overproduction": (
            "float64",
            "_overproduction_evolution",
            "industries",
            np.nan,
        ),
        "final_demand_unmet": (
            "float64",
            "_final_demand_unmet_evolution",
            "industries",
            np.nan,
        ),
        "rebuild_prod": (
            "float64",
            "_rebuild_production_evolution",
            "industries",
            np.nan,
        ),
        "inputs_stocks": ("float64", "_inputs_evolution", "stocks", np.nan),
        "limiting_inputs": ("byte", "_limiting_inputs_evolution", "stocks", -1),
        "productive_capital_to_recover": (
            "float64",
            "_regional_sectoral_productive_capital_destroyed_evolution",
            "industries",
            np.nan,
        ),
    }

    def __init__(
        self,
        model: Union[ARIOBaseModel, ARIOPsiModel],
        register_stocks: bool = False,
        n_temporal_units_to_sim: int = 365,
        events_list: Optional[list[Event]] = None,
        separate_sims: bool = False,
        save_events: bool = False,
        save_params: bool = False,
        save_index: bool = False,
        save_records: list | str = [],
        boario_output_dir: str | pathlib.Path = tempfile.mkdtemp(prefix="boario"),
        results_dir_name: Optional[str] = None,
    ) -> None:
        """A Simulation instance can be initialized with the following parameters:

        Parameters
        ----------
        model : Union[ARIOBaseModel, ARIOPsiModel, ARIOClimadaModel]
            The model to run the simulation with.
        register_stocks : bool, default False
            A boolean stating if stocks evolution should be registered in a file.
            Be aware that such arrays have timesteps*sectors*sectors*regions size
            which can rapidly lead to very large files.
        n_temporal_units_to_sim : int, default 365
            The number of temporal units to simulates.
        events_list : list[Event], optional
            An optional list of events to run the simulation with [WIP].
        separate_sims : bool, default False
            Whether to run each event separately or during the same simulation [WIP].
        boario_output_dir : str | pathlib.Path
            An optional directory where to save files generated by the simulation.
        results_dir_name : str, default 'results'
            The name of the folder where simulation results will be stored.

        Examples
        --------

        See #add link to example page.

        """

        if events_list is None:
            events_list = []
        logger.info("Initializing new simulation instance")
        self._save_events = save_events
        self._save_params = save_params
        self._save_index = save_index
        self._register_stocks = register_stocks
        self.output_dir = pathlib.Path(boario_output_dir)
        """pathlib.Path, optional: Optional path to the directory where output are stored."""

        # Pre-init record variables
        self._production_evolution = np.array([])
        self._production_cap_evolution = np.array([])
        self._final_demand_evolution = np.array([])
        self._io_demand_evolution = np.array([])
        self._rebuild_demand_evolution = np.array([])
        self._overproduction_evolution = np.array([])
        self._final_demand_unmet_evolution = np.array([])
        self._rebuild_production_evolution = np.array([])
        self._inputs_evolution = np.array([])
        self._limiting_inputs_evolution = np.array([])
        self._regional_sectoral_productive_capital_destroyed_evolution = np.array([])

        if save_records != [] or save_events or save_params or save_index:
            self.output_dir.resolve().mkdir(parents=True, exist_ok=True)

        self.results_storage = (
            self.output_dir.resolve()
            if not results_dir_name
            else self.output_dir.resolve() / results_dir_name
        )
        """str: Name of the folder in `output_dir` where the results will be stored if saved."""

        if save_records != []:
            if not self.results_storage.exists():
                self.results_storage.mkdir()

        if not self.results_storage.exists():
            self.results_storage.mkdir(parents=True)

        self.model = model
        """Union[ARIOBaseModel, ARIOPsiModel] : The model to run the simulation with.
        See :class:`~boario.model_base.ARIOBaseModel`."""

        self.all_events: list[Event] = events_list
        """list[Event]: A list containing all events associated with the simulation."""

        self.currently_happening_events: list[Event] = []
        """list[Event]: A list containing all events that are happening at the current timestep of the simulation."""

        self.events_timings = set()
        self.n_temporal_units_to_sim = n_temporal_units_to_sim
        """int: The total number of `temporal_units` to simulate."""

        Event.temporal_unit_range = self.n_temporal_units_to_sim
        self.current_temporal_unit = 0
        """int: Tracks the number of `temporal_units` elapsed since simulation start.
        This may differs from the number of `steps` if the parameter `n_temporal_units_by_step`
        differs from 1 temporal_unit as `current_temporal_unit` is actually `step` * `n_temporal_units_by_step`."""

        self.equi = {
            (int(0), int(0), "production"): "equi",
            (int(0), int(0), "stocks"): "equi",
            (int(0), int(0), "rebuilding"): "equi",
        }
        self.n_temporal_units_simulated = 0
        self._n_checks = 0
        self._monotony_checker = 0
        self.scheme = "proportional"
        self.has_crashed = False
        # RECORDS FILES

        self.records_storage: pathlib.Path = self.results_storage / "records"
        """Place where records are stored if stored"""

        self._files_to_record = []

        if save_records != []:
            if isinstance(save_records, str):
                if save_records == "all":
                    save_records = self.__possible_records
                else:
                    raise ValueError(
                        f'save_records argument has to be either "all" or a sublist of {self.__possible_records}'
                    )

            impossible_records = set(save_records).difference(
                set(self.__possible_records)
            )
            if not len(impossible_records) == 0:
                raise ValueError(
                    f"{impossible_records} are not possible records ({self.__possible_records})"
                )
            logger.info(f"Will save {save_records} records")
            logger.info("Records storage is: {}".format(self.records_storage))
            self.records_storage.mkdir(parents=True, exist_ok=True)
            self._save_index = True
            self._save_events = True
            self._save_params = True

        self._init_records(save_records)

        Event.temporal_unit_range = self.n_temporal_units_to_sim
        self.params_dict = {
            "n_temporal_units_to_sim": self.n_temporal_units_to_sim,
            "output_dir": str(self.output_dir)
            if hasattr(self, "output_dir")
            else "none",
            "results_storage": self.results_storage.stem
            if hasattr(self, "results_storage")
            else "none",
            "model_type": self.model.__class__.__name__,
            "psi_param": self.model.psi
            if isinstance(self.model, ARIOPsiModel)
            else None,
            "order_type": self.model.order_type,
            "n_temporal_units_by_step": self.model.n_temporal_units_by_step,
            "year_to_temporal_unit_factor": self.model.iotable_year_to_temporal_unit_factor,
            "inventory_restoration_tau": list(self.model.restoration_tau)
            if isinstance(self.model, ARIOPsiModel)
            else None,
            "alpha_base": self.model.overprod_base,
            "alpha_max": self.model.overprod_max,
            "alpha_tau": self.model.overprod_tau,
            "rebuild_tau": self.model.rebuild_tau,
        }
        """dict: A dictionary saving the parameters the simulation was run with."""

        logger.info("Initialized !")
        formatted_params_dict = {
            key: print_summary(value) if key == "inventory_restoration_tau" else value
            for key, value in self.params_dict.items()
        }
        logger.info(
            "Simulation parameters:\n{}".format(
                pformat(formatted_params_dict, compact=True)
            )
        )

    def loop(self, progress: bool = True):
        r"""Launch the simulation loop.

        This method launch the simulation for the number of steps to simulate
        described by the attribute ``self.n_temporal_units_to_sim``, calling the
        :meth:`next_step` method. For convenience, it dumps the
        parameters used in the logs just before running the loop. Once the loop
        is completed, it flushes the different memmaps generated.

        Parameters
        ----------

        progress: bool, default: True
            If True, shows a progress bar of the loop in the console.
        """
        logger.info(
            "Starting model loop for at most {} steps".format(
                self.n_temporal_units_to_sim // self.model.n_temporal_units_by_step + 1
            )
        )
        logger.info(
            "One step is {}/{} of a year".format(
                self.model.n_temporal_units_by_step,
                self.model.iotable_year_to_temporal_unit_factor,
            )
        )
        tmp = logging.FileHandler(self.results_storage / "simulation.log")
        tmp.setLevel(logging.INFO)
        tmp.setFormatter(DEBUGFORMATTER)
        logger.addHandler(tmp)
        logger.info("Events : {}".format(self.all_events))

        run_range = range(
            0,
            self.n_temporal_units_to_sim,
            math.floor(self.model.n_temporal_units_by_step),
        )

        if self._save_events:
            (pathlib.Path(self.results_storage) / "jsons").mkdir(
                parents=True, exist_ok=True
            )
            with (
                pathlib.Path(self.results_storage) / "jsons" / "simulated_events.json"
            ).open("w") as f:
                event_dicts = [ev.event_dict for ev in self.all_events]
                json.dump(event_dicts, f, indent=4, cls=CustomNumpyEncoder)
        if progress:
            widgets = [
                "Processed: ",
                progressbar.Counter("Step: %(value)d"),
                " ~ ",
                progressbar.Percentage(),
                " ",
                progressbar.ETA(),
            ]
            bar = progressbar.ProgressBar(widgets=widgets, redirect_stdout=True)
            for _ in bar(run_range):
                step_res = self.next_step()
                self.n_temporal_units_simulated = self.current_temporal_unit
                if step_res == 1:
                    self.has_crashed = True
                    logger.warning(
                        f"""Economy seems to have crashed.
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    logger.warning(
                        f"""Economy seems to have found an equilibrium
                    - At step : {self.current_temporal_unit}
                    """
                    )
            bar.finish()
        else:
            for _ in run_range:
                step_res = self.next_step()
                self.n_temporal_units_simulated = self.current_temporal_unit
                if step_res == 1:
                    self.has_crashed = True
                    logger.warning(
                        f"""Economy or model seems to have crashed.
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    logger.warning(
                        f"""Economy seems to have found an equilibrium
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break

        if self._files_to_record != []:
            self._flush_memmaps()

        if self._save_index:
            self.model.write_index(self.results_storage / "jsons" / "indexes.json")

        self.params_dict["n_temporal_units_simulated"] = self.n_temporal_units_simulated
        self.has_crashed = self.has_crashed

        if self._save_params:
            with (
                pathlib.Path(self.results_storage) / "jsons" / "simulated_params.json"
            ).open("w") as f:
                json.dump(self.params_dict, f, indent=4, cls=CustomNumpyEncoder)
            with (
                pathlib.Path(self.results_storage) / "jsons" / "equilibrium_checks.json"
            ).open("w") as f:
                json.dump(
                    {str(k): v for k, v in self.equi.items()},
                    f,
                    indent=4,
                    cls=CustomNumpyEncoder,
                )
        logger.info("Loop complete")

    def next_step(
        self,
        check_period: int = 182,
        min_steps_check: Optional[int] = None,
        min_failing_regions: Optional[int] = None,
    ):
        """Advance the model to the next temporal step.

        This method wraps all computation required to advance to the next step
        of the simulation.

        First it checks if an event is planned to
        occur at the current step and if so, shocks the model with the
        corresponding event. Then it :

        0) If at least one step elapsed, it computes the new overproduction vector for the next step (using :meth:`~boario.model_base.ARIOBaseModel.calc_overproduction`)

        1) Computes production for the current step. (See :meth:`~boario.model_base.ARIOBaseModel.calc_production`)

        2) Distribute the `realised` production towards the different demands (intermediate, final, rebuilding) and compute the changes in the inputs stock matrix (see :meth:`~boario.model_base.ARIOBaseModel.distribute_production`)

        Note that it is during this step that the model checks if an event is completely rebuild/recovered.

        3) Computes the orders matrix (i.e. the intermediate demand) for the next step (see :meth:`~boario.model_base.ARIOBaseModel.calc_orders`)

        Additionally, once every `check_period` steps elapsed, it checks for crash or equilibrium of the economy (see :meth:`~boario.simulation.check_equilibrium`).

        Parameters
        ----------
        check_period : int
            The time period in number of temporal units to wait between each "crash/equilibrium" check.
        min_steps_check : Optional[int]
            The minimum wait before the first check.
        min_failing_regions : Optional[int]
            The minimum number of failing regions required to consider economy has crashed.

        """
        try:
            if min_steps_check is None:
                min_steps_check = self.n_temporal_units_to_sim // 5
            if min_failing_regions is None:
                min_failing_regions = self.model.n_regions * self.model.n_sectors // 3

            # Check if there are new events to add,
            # if some happening events can start rebuilding (if rebuildable),
            # and updates the internal model production_cap decrease and rebuild_demand
            self._check_happening_events()

            if "_inputs_evolution" in self._files_to_record:
                self._write_stocks()

            if self.current_temporal_unit > 1:
                self.model.calc_overproduction()

            if "_overproduction_evolution" in self._files_to_record:
                self._write_overproduction()
            if "_rebuild_demand_evolution" in self._files_to_record:
                self._write_rebuild_demand()
            if "_final_evolution" in self._files_to_record:
                self._write_final_demand()
            if "_io_demand_evolution" in self._files_to_record:
                self._write_io_demand()

            constraints = self.model.calc_production(self.current_temporal_unit)

            if "_limiting_inputs_evolution" in self._files_to_record:
                self._write_limiting_stocks(constraints)
            if "_production_evolution" in self._files_to_record:
                self._write_production()
            if "_production_cap_evolution" in self._files_to_record:
                self._write_production_max()
            if (
                "_regional_sectoral_productive_capital_destroyed_evolution"
                in self._files_to_record
            ):
                self._write_productive_capital_lost()

            try:
                rebuildable_events = [
                    ev
                    for ev in self.currently_happening_events
                    if isinstance(ev, EventKapitalRebuild) and ev.rebuildable
                ]
                events_to_remove = self.model.distribute_production(
                    rebuildable_events, self.scheme
                )
                if "_final_demand_unmet_evolution" in self._files_to_record:
                    self._write_final_demand_unmet()
                if "_rebuild_production_evolution" in self._files_to_record:
                    self._write_rebuild_prod()
            except RuntimeError as e:
                logger.exception("This exception happened:", e)
                self.model.matrix_stock.dump(
                    self.results_storage / "matrix_stock_dump.pkl"
                )
                logger.error(
                    "Negative values in the stocks, matrix has been dumped in the results dir : \n {}".format(
                        self.results_storage / "matrix_stock_dump.pkl"
                    )
                )
                return 1
            events_to_remove = events_to_remove + [
                ev for ev in self.currently_happening_events if ev.over
            ]
            if events_to_remove:
                self.currently_happening_events = [
                    e
                    for e in self.currently_happening_events
                    if e not in events_to_remove
                ]
                for e in events_to_remove:
                    if isinstance(e, EventKapitalDestroyed):
                        logger.info(
                            "Temporal_Unit : {} ~ Event named {} that occured at {} in {} for {} damages is completely rebuilt/recovered".format(
                                self.current_temporal_unit,
                                e.name,
                                e.occurrence,
                                e.aff_regions,
                                e.total_productive_capital_destroyed,
                            )
                        )

            self.model.calc_orders()

            n_checks = self.current_temporal_unit // check_period
            if n_checks > self._n_checks:
                self.check_equilibrium(n_checks)
                self._n_checks += 1

            self.current_temporal_unit += self.model.n_temporal_units_by_step
            return 0
        except Exception as e:
            logger.exception(f"The following exception happened: {e}")
            return 1

    def check_equilibrium(self, n_checks: int):
        """Checks the status of production, stocks and rebuilding demand.

        This methods checks and store the status of production, inputs stocks
        and rebuilding demand and store the information in ``self.equi``.

        At the moment, the following status are implemented:

        - `production` and `stocks` can be `greater` (ie all industries are producing more), `equi` (ie all industries produce almost the same as at initial equilibrium (0.01 atol)), `not equi` (ie neither of the previous case)
        - `rebuilding demand` can be `finished` or `not finished` depending if some events still have some rebuilding demand unanswered or if all are completely rebuilt.

        Parameters
        ----------
        n_checks : int
            The number of checks counter.

        """
        if np.greater_equal(self.model.production, self.model.X_0).all():
            self.equi[(n_checks, self.current_temporal_unit, "production")] = "greater"
        elif np.allclose(self.model.production, self.model.X_0, atol=0.01):
            self.equi[(n_checks, self.current_temporal_unit, "production")] = "equi"
        else:
            self.equi[(n_checks, self.current_temporal_unit, "production")] = "not equi"

        if np.greater_equal(self.model.matrix_stock, self.model.matrix_stock_0).all():
            self.equi[(n_checks, self.current_temporal_unit, "stocks")] = "greater"
        elif np.allclose(self.model.production, self.model.X_0, atol=0.01):
            self.equi[(n_checks, self.current_temporal_unit, "stocks")] = "equi"
        else:
            self.equi[(n_checks, self.current_temporal_unit, "stocks")] = "not equi"

        if (
            self.model.tot_rebuild_demand is None
            or not self.model.tot_rebuild_demand.any()
        ):
            self.equi[(n_checks, self.current_temporal_unit, "rebuilding")] = "finished"
        else:
            self.equi[
                (n_checks, self.current_temporal_unit, "rebuilding")
            ] = "not finished"

    def read_events_from_list(self, events_list: list[dict]):
        raise NotImplementedError("I have to redo this one")

    #     """Import a list of events (as a list of dictionaries) into the model.

    #     Also performs various checks on the events to avoid badly written events.
    #     See :ref:`How to define Events <boario-events>` to understand how to write events dictionaries or JSON files.

    #     Parameters
    #     ----------
    #     events_list :
    #         List of events as dictionaries.

    #     """
    #     logger.info("Reading events from given list and adding them to the model")
    #     if not isinstance(events_list, list):
    #         if isinstance(events_list, dict):
    #             raise TypeError(
    #                 "read_events_from_list() takes a list of event dicts as an argument, not a single event dict, you might want to use read_event(your_event) instead."
    #             )
    #         else:
    #             raise TypeError(
    #                 "read_events_from_list() takes a list of event dicts as an argument, not a {}".format(
    #                     type(events_list)
    #                 )
    #             )
    #     for ev_dic in events_list:
    #         self.read_event(ev_dic)

    def read_event(self, ev_dic: dict):
        raise NotImplementedError("I have to redo this one")

    #     if ev_dic["aff_sectors"] == "all":
    #         ev_dic["aff_sectors"] = list(self.model.sectors)
    #     ev = Event(ev_dic)
    #     self.all_events.append(ev)
    #     self.events_timings.add(ev.occurrence)

    def add_event(self, ev: Event):
        """Add an event to the simulation.

        Parameters
        ----------
        ev : Event
            The event to add.
        """

        self.all_events.append(ev)
        self.events_timings.add(ev.occurrence)

    def reset_sim_with_same_events(self):
        """Resets the model to its initial status (without removing the events). [WIP]"""
        logger.info("Resetting model to initial status (with same events)")
        self.current_temporal_unit = 0
        self._monotony_checker = 0
        self._n_checks = 0
        self.n_temporal_units_simulated = 0
        self.has_crashed = False
        self._reset_records()
        self.model.reset_module()

    def reset_sim_full(self):
        """Resets the model to its initial status and remove all events."""

        self.reset_sim_with_same_events()
        logger.info("Resetting events")
        self.all_events = []
        self.currently_happening_events = []
        self.events_timings = set()

    def update_params(self, new_params: dict):
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
        raise NotImplementedError("To fix")
        logger.info("Updating model parameters")
        self.params_dict = new_params
        results_storage = pathlib.Path(self.results_storage)
        if not results_storage.exists():
            results_storage.mkdir(parents=True)
        self.model.update_params(self.params_dict)

    def write_index(self, index_file: Union[str, pathlib.Path]):
        """Write the index of the dataframes used in the model in a json file.

        See :meth:`~boario.model_base.ARIOBaseModel.write_index` for a more detailed documentation.

        Parameters
        ----------
        index_file : Union[str, pathlib.Path]
            name of the file to save the indexes to.

        """
        self.model.write_index(index_file)

    def read_events(self, events_file: Union[str, pathlib.Path]):
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
        raise NotImplementedError("I have to redo this one")
        # logger.info(
        #     "Reading events from {} and adding them to the model".format(events_file)
        # )
        # if isinstance(events_file, str):
        #     events_file = pathlib.Path(events_file)
        # elif not isinstance(events_file, pathlib.Path):
        #     raise TypeError("Given index file is not an str or a Path")
        # if not events_file.exists():
        #     raise FileNotFoundError("This file does not exist: ", events_file)
        # else:
        #     with events_file.open("r") as f:
        #         events = json.load(f)
        # if isinstance(events, list):
        #     self.read_events_from_list(events)
        # else:
        #     self.read_events_from_list([events])

    def _check_happening_events(self) -> None:
        """Updates the status of all events.

        Check the `all_events` attribute and `current_temporal_unit` and
        updates the events accordingly (ie if they happened, if they can start
        rebuild/recover)

        """
        for ev in self.all_events:
            if not ev.happened:
                if (
                    (self.current_temporal_unit - self.model.n_temporal_units_by_step)
                    <= ev.occurrence
                    <= self.current_temporal_unit
                ):
                    logger.info(
                        "Temporal_Unit : {} ~ Shocking model with new event".format(
                            self.current_temporal_unit
                        )
                    )
                    logger.info("Affected regions are : {}".format(ev.aff_regions))
                    ev.happened = True
                    self.currently_happening_events.append(ev)
        for ev in self.currently_happening_events:
            if isinstance(ev, EventKapitalRebuild):
                ev.rebuildable = self.current_temporal_unit
            if isinstance(ev, EventKapitalRecover):
                ev.recoverable = self.current_temporal_unit
                if ev.recoverable:
                    ev.recovery(self.current_temporal_unit)
        self.model.update_system_from_events(self.currently_happening_events)

    def _write_production(self) -> None:
        """Saves the current production vector to the memmap."""
        self._production_evolution[self.current_temporal_unit] = self.model.production

    def _write_rebuild_prod(self) -> None:
        """Saves the current rebuilding production vector to the memmap."""
        logger.debug(
            f"self._rebuild_production_evolution shape : {self._rebuild_production_evolution.shape}, self.model.rebuild_prod shape : {self.model.rebuild_prod.shape}"
        )
        self._rebuild_production_evolution[
            self.current_temporal_unit
        ] = self.model.rebuild_prod

    def _write_productive_capital_lost(self) -> None:
        """Saves the current remaining productive_capital to rebuild vector to the memmap."""
        self._regional_sectoral_productive_capital_destroyed_evolution[
            self.current_temporal_unit
        ] = self.model.productive_capital_lost

    def _write_production_max(self) -> None:
        """Saves the current production capacity vector to the memmap."""
        self._production_cap_evolution[
            self.current_temporal_unit
        ] = self.model.production_cap

    def _write_io_demand(self) -> None:
        """Saves the current (total per industry) intermediate demand vector to the memmap."""
        self._io_demand_evolution[
            self.current_temporal_unit
        ] = self.model.matrix_orders.sum(axis=1)

    def _write_final_demand(self) -> None:
        """Saves the current (total per industry) final demand vector to the memmap."""
        self._final_demand_evolution[
            self.current_temporal_unit
        ] = self.model.final_demand.sum(axis=1)

    def _write_rebuild_demand(self) -> None:
        """Saves the current (total per industry) rebuilding demand vector to the memmap."""
        to_write = np.full(self.model.n_regions * self.model.n_sectors, 0.0)
        if (r_dem := self.model.tot_rebuild_demand) is not None:
            self._rebuild_demand_evolution[self.current_temporal_unit] = r_dem  # type: ignore
        else:
            self._rebuild_demand_evolution[self.current_temporal_unit] = to_write  # type: ignore

    def _write_overproduction(self) -> None:
        """Saves the current overproduction vector to the memmap."""
        self._overproduction_evolution[self.current_temporal_unit] = self.model.overprod

    def _write_final_demand_unmet(self) -> None:
        """Saves the unmet final demand (for this step) vector to the memmap."""
        self._final_demand_unmet_evolution[
            self.current_temporal_unit
        ] = self.model.final_demand_not_met

    def _write_stocks(self) -> None:
        """Saves the current inputs stock matrix to the memmap."""
        self._inputs_evolution[self.current_temporal_unit] = self.model.matrix_stock

    def _write_limiting_stocks(self, limiting_stock: np.ndarray) -> None:
        """Saves the current limiting inputs matrix to the memmap."""
        self._limiting_inputs_evolution[self.current_temporal_unit] = limiting_stock  # type: ignore

    def _flush_memmaps(self) -> None:
        """Saves files to record"""
        for at in self._files_to_record:
            if not hasattr(self, at):
                raise RuntimeError(
                    f"{at} should be a member yet it isn't. This shouldn't happen."
                )
            else:
                getattr(self, at).flush()

    def _init_records(self, save_records):
        for rec in self.__possible_records:
            if rec == "inputs_stocks" and not self._register_stocks:
                logger.debug("Will not save inputs stocks")
                pass
            else:
                if rec == "inputs_stocks":
                    logger.info(
                        f"Simulation will save inputs stocks. Estimated size is {sizeof_fmt(self.n_temporal_units_to_sim * self.model.n_sectors * self.model.n_sectors * self.model.n_regions * 64)}"
                    )
                save = rec in save_records
                filename = rec
                dtype, attr_name, shapev, fillv = self.__file_save_array_specs[rec]
                if shapev == "industries":
                    shape = (
                        self.n_temporal_units_to_sim,
                        self.model.n_sectors * self.model.n_regions,
                    )
                elif shapev == "stocks":
                    shape = (
                        self.n_temporal_units_to_sim,
                        self.model.n_sectors,
                        self.model.n_sectors * self.model.n_regions,
                    )
                else:
                    raise RuntimeError(f"shapev {shapev} unrecognised")
                memmap_array = TempMemmap(
                    filename=(self.records_storage / filename),
                    dtype=dtype,
                    mode="w+",
                    shape=shape,
                    save=save,
                )
                memmap_array.fill(fillv)
                self._files_to_record.append(attr_name)
                setattr(self, attr_name, memmap_array)

    def _reset_records(
        self,
    ):
        for rec in self.__possible_records:
            _, attr_name, _, fillv = self.__file_save_array_specs[rec]
            if rec == "input_stocks" and not self._register_stocks:
                pass
            else:
                memmap_array = getattr(self, attr_name)
                memmap_array.fill(fillv)
                setattr(self, attr_name, memmap_array)

    @property
    def production_realised(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._production_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def production_capacity(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._production_cap_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def final_demand(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._final_demand_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def intermediate_demand(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._io_demand_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def rebuild_demand(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._rebuild_demand_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def overproduction(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._overproduction_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def final_demand_unmet(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._final_demand_unmet_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def rebuild_prod(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._rebuild_production_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def inputs_stocks(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._inputs_evolution,
            columns=self.model.industries,
            copy=True,
            index=pd.MultiIndex.from_product(
                [list(range(self.n_temporal_units_to_sim)), self.model.sectors],
                names=["step", "input"],
            ),
        )

    @property
    def limiting_inputs(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._limiting_inputs_evolution,
            columns=self.model.industries,
            copy=True,
            index=pd.MultiIndex.from_product(
                [list(range(self.n_temporal_units_to_sim)), self.model.sectors],
                names=["step", "input"],
            ),
        )

    @property
    def productive_capital_to_recover(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._regional_sectoral_productive_capital_destroyed_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")
