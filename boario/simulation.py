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

from functools import cached_property, partial
import json
import logging
import warnings
import math
import pathlib
import tempfile
from pprint import pformat
from typing import Callable, Literal, Optional, Union, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import progressbar

from boario import DEBUG_TRACE, DEBUGFORMATTER, logger
from boario.event import (
    Event,
    EventArbitraryProd,
    EventKapitalDestroyed,
    EventKapitalRebuild,
    EventKapitalRecover,
    RegionsList,
    SectorsList,
)

from boario.extended_models import ARIOPsiModel
from boario.model_base import ARIOBaseModel
from boario.utils.misc import CustomNumpyEncoder, TempMemmap, print_summary, sizeof_fmt

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
        save_events: bool = False,
        save_params: bool = False,
        save_index: bool = False,
        save_records: list | str = [],
        boario_output_dir: str | pathlib.Path = tempfile.mkdtemp(prefix="boario"),
        results_dir_name: Optional[str] = None,
        show_progress: bool = False,
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
        save_events: bool, default False
            If True, saves a json file of the list of events to simulate when starting the model loop.
        save_params: bool, default False
            If True, saves a json file of the parameters used when starting the model loop.
        save_index: bool, default False
            If True, saves a json file of the list of industries in the model when starting the model loop (convenience).
        save_records: list | str, default []
            The list of simulation variable records to save in corresponding files.
        boario_output_dir : str | pathlib.Path
            An optional directory where to save files generated by the simulation. Defaults to a temporary directory prefixed by "boario".
        results_dir_name : str, default 'results'
            The name of the folder where simulation results will be stored.
        show_progress: bool, default: False
            If True, shows a progress bar in the console during the simulation.

        """
        self.output_dir = pathlib.Path(boario_output_dir)
        """pathlib.Path, optional: Optional path to the directory where output are stored."""
        self.results_storage = (
            self.output_dir.resolve()
            if not results_dir_name
            else self.output_dir.resolve() / results_dir_name
        )
        """str: Name of the folder in `output_dir` where the results will be stored if saved."""

        if not self.results_storage.exists():
            self.results_storage.mkdir(parents=True)

        tmp = logging.FileHandler(self.results_storage / "simulation.log")
        tmp.setLevel(logging.INFO)
        tmp.setFormatter(DEBUGFORMATTER)
        logger.addHandler(tmp)

        if events_list is None:
            events_list = []
        logger.info("Initializing new simulation instance")
        self._save_events = save_events
        self._save_params = save_params
        self._save_index = save_index
        self._register_stocks = register_stocks
        self._show_progress = show_progress

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

        if save_records != []:
            if not self.results_storage.exists():
                self.results_storage.mkdir()

        self.model = model
        """Union[ARIOBaseModel, ARIOPsiModel] : The model to run the simulation with.
        See :class:`~boario.model_base.ARIOBaseModel`."""

        self._event_tracking: list[EventTracker] = []
        self._events_to_rebuild = 0

        self.all_events: list[Event] = []
        """list[Event]: A list containing all events associated with the simulation."""

        if events_list is not None:
            self.add_events(events_list)

        self.n_temporal_units_to_sim = n_temporal_units_to_sim
        """int: The total number of `temporal_units` to simulate."""

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
        """pathlib.Path: Place where records are stored if stored"""

        self._vars_to_record = []
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
            logger.info("Records storage is: {self.records_storage}")
            self.records_storage.mkdir(parents=True, exist_ok=True)
            self._save_index = True
            self._save_events = True
            self._save_params = True

        self._init_records(save_records)

        self.params_dict = {
            "n_temporal_units_to_sim": self.n_temporal_units_to_sim,
            "output_dir": (
                str(self.output_dir) if hasattr(self, "output_dir") else "none"
            ),
            "results_storage": (
                self.results_storage.stem
                if hasattr(self, "results_storage")
                else "none"
            ),
            "model_type": self.model.__class__.__name__,
            "psi_param": (
                self.model.psi if isinstance(self.model, ARIOPsiModel) else None
            ),
            "order_type": self.model.order_type,
            "n_temporal_units_by_step": self.model.n_temporal_units_by_step,
            "year_to_temporal_unit_factor": self.model.iotable_year_to_temporal_unit_factor,
            "inventory_restoration_tau": (
                list(np.reciprocal(self.model.restoration_tau))
                if isinstance(self.model, ARIOPsiModel)
                else None
            ),
            "alpha_base": self.model.overprod_base,
            "alpha_max": self.model.overprod_max,
            "alpha_tau": self.model.overprod_tau
            * self.model.iotable_year_to_temporal_unit_factor,
            "rebuild_tau": self.model.rebuild_tau,
        }
        """dict: A dictionary saving the parameters the simulation was run with."""

        logger.info("Initialized !")
        formatted_params_dict = {
            key: print_summary(value) if key == "inventory_restoration_tau" else value
            for key, value in self.params_dict.items()
        }
        logger.info(
            f"Simulation parameters:\n{pformat(formatted_params_dict, compact=True)}"
        )

    def loop(self):
        r"""Launch the simulation loop.

        This method launch the simulation for the number of steps to simulate
        described by the attribute ``self.n_temporal_units_to_sim``, calling the
        :meth:`next_step` method. For convenience, it dumps the
        parameters used in the logs just before running the loop. Once the loop
        is completed, it flushes the different memmaps generated.

        """
        logger.info(
            f"Starting model loop for at most {self.n_temporal_units_to_sim // self.model.n_temporal_units_by_step + 1} steps"
        )
        logger.info(
            f"One step is {self.model.n_temporal_units_by_step}/{self.model.iotable_year_to_temporal_unit_factor} of a year"
        )
        logger.info("Events : {self.all_events}")

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
            ).open("w") as ffile:
                event_dicts = [ev.event_dict for ev in self.all_events]
                json.dump(event_dicts, ffile, indent=4, cls=CustomNumpyEncoder)

        if self._show_progress:
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
                    warnings.warn(
                        f"""Economy seems to have crashed.
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    warnings.warn(
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
                    warnings.warn(
                        f"""Economy or model seems to have crashed.
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break
                elif self._monotony_checker > 3:
                    warnings.warn(
                        f"""Economy seems to have found an equilibrium
                    - At step : {self.current_temporal_unit}
                    """
                    )
                    break

        if self._files_to_record:
            self._flush_memmaps()

        if self._save_index:
            self.model.write_index(self.results_storage / "jsons" / "indexes.json")

        self.params_dict["n_temporal_units_simulated"] = self.n_temporal_units_simulated
        self.params_dict["has_crashed"] = self.has_crashed

        if self._save_params:
            with (
                pathlib.Path(self.results_storage) / "jsons" / "simulated_params.json"
            ).open("w") as ffile:
                json.dump(self.params_dict, ffile, indent=4, cls=CustomNumpyEncoder)
            with (
                pathlib.Path(self.results_storage) / "jsons" / "equilibrium_checks.json"
            ).open("w") as ffile:
                json.dump(
                    {str(k): v for k, v in self.equi.items()},
                    ffile,
                    indent=4,
                    cls=CustomNumpyEncoder,
                )
        if self.has_crashed:
            logger.info("Loop crashed before completion.")
        else:
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
        corresponding event. Then:

        0) If at least one step elapsed, it computes the new overproduction vector for the next step (using :meth:`~boario.model_base.ARIOBaseModel.calc_overproduction`)

        1) Computes production for the current step. (See :meth:`~boario.model_base.ARIOBaseModel.calc_production`)

        2) Distribute the `realised` production towards the different demands (intermediate, final, rebuilding) and compute the changes in the inputs stock matrix (see :meth:`~boario.model_base.ARIOBaseModel.distribute_production`)

        Note that it is during this step that the model checks if an event is completely rebuild/recovered and removes it from the list the case being.

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

            if ("_inputs_evolution" in self._files_to_record) or (
                "_inputs_evolution" in self._vars_to_record
            ):
                self._write_stocks()

            # 0)
            if self.current_temporal_unit > 1:
                self.model.calc_overproduction()

            if ("_overproduction_evolution" in self._files_to_record) or (
                "_overproduction_evolution" in self._files_to_record
            ):
                self._write_overproduction()
            if ("_rebuild_demand_evolution" in self._files_to_record) or (
                "_rebuild_demand_evolution" in self._vars_to_record
            ):
                self._write_rebuild_demand()
            if ("_final_demand_evolution" in self._files_to_record) or (
                "_final_demand_evolution" in self._vars_to_record
            ):
                self._write_final_demand()
            if ("_io_demand_evolution" in self._files_to_record) or (
                "_io_demand_evolution" in self._vars_to_record
            ):
                self._write_io_demand()

            # 1)
            constraints = self.model.calc_production(self.current_temporal_unit)

            if ("_limiting_inputs_evolution" in self._files_to_record) or (
                "_limiting_inputs_evolution" in self._vars_to_record
            ):
                self._write_limiting_stocks(constraints)
            if ("_production_evolution" in self._files_to_record) or (
                "_production_evolution" in self._vars_to_record
            ):
                self._write_production()
            if ("_production_cap_evolution" in self._files_to_record) or (
                "_production_cap_evolution" in self._vars_to_record
            ):
                self._write_production_max()
            if (
                "_regional_sectoral_productive_capital_destroyed_evolution"
                in self._files_to_record
            ) or (
                "_regional_sectoral_productive_capital_destroyed_evolution"
                in self._vars_to_record
            ):
                self._write_productive_capital_lost()

            # 2)
            try:
                self.model.distribute_production(self.scheme)
                if ("_final_demand_unmet_evolution" in self._files_to_record) or (
                    "_final_demand_unmet_evolution" in self._vars_to_record
                ):
                    self._write_final_demand_unmet()
                if ("_rebuild_production_evolution" in self._files_to_record) or (
                    "_rebuild_production_evolution" in self._vars_to_record
                ):
                    self._write_rebuild_prod()
                self.rebuild_events()
                self.recover_events()

            except RuntimeError:
                logger.exception("An exception happened: ")
                self.model.inputs_stock.dump(
                    self.results_storage / "matrix_stock_dump.pkl"
                )
                logger.error(
                    f"Negative values in the stocks, matrix has been dumped in the results dir : \n {self.results_storage / 'matrix_stock_dump.pkl'}"
                )
                return 1
            self.model.calc_orders()

            n_checks = self.current_temporal_unit // check_period
            if n_checks > self._n_checks:
                self.check_equilibrium(n_checks)
                self._n_checks += 1

            self.current_temporal_unit += self.model.n_temporal_units_by_step
            return 0
        except Exception as err:
            logger.exception("The following exception happened:")
            raise RuntimeError("An exception happened:") from err

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

        if np.greater_equal(self.model.inputs_stock, self.model.inputs_stock_0).all():
            self.equi[(n_checks, self.current_temporal_unit, "stocks")] = "greater"
        elif np.allclose(self.model.production, self.model.X_0, atol=0.01):
            self.equi[(n_checks, self.current_temporal_unit, "stocks")] = "equi"
        else:
            self.equi[(n_checks, self.current_temporal_unit, "stocks")] = "not equi"

        if (
            self.model.rebuild_demand_tot is None
            or self.model.rebuild_demand is None
            or not self.model.rebuild_demand.any()
        ):
            self.equi[(n_checks, self.current_temporal_unit, "rebuilding")] = "finished"
        else:
            self.equi[(n_checks, self.current_temporal_unit, "rebuilding")] = (
                "not finished"
            )

    def add_events(self, events: list[Event]):
        """Add a list of events to the simulation.

        Parameters
        ----------
        events : list[Event]
            The events to add.
        """
        if not isinstance(events, list):
            raise TypeError(f"list[Event] expected, {type(events)} received.")
        for ev in events:
            self.add_event(ev)

    def add_event(self, ev: Event):
        """Add an event to the simulation.

        Parameters
        ----------
        ev : Event
            The event to add.
        """
        if not isinstance(ev, Event):
            raise ValueError(f"Event expected, {type(ev)} received.")
        self.event_compatibility(ev)
        self.all_events.append(ev)
        self._event_tracking.append(EventTracker(self, ev))

    def event_compatibility(self, ev: Event):
        """Checks if an event is compatible with current simulation environment

        Parameters
        ----------
        ev : Event
            The event to checks.

        Raises
        ------

        ValueError
            If one attribute of the event is not consistent with the simulation context.


        """
        if not 0 < ev.occurrence <= self.n_temporal_units_to_sim:
            raise ValueError(
                f"Occurrence of event is not in the range of simulation steps (cannot be 0) : {ev.occurrence} not in ]0-{self.n_temporal_units_to_sim}]"
            )
        if not 0 < ev.occurrence + ev.duration <= self.n_temporal_units_to_sim:
            raise ValueError(
                f"Occurrence + duration of event is not in the range of simulation steps (cannot be 0) : {ev.occurrence} not in ]0-{self.n_temporal_units_to_sim}]"
            )
        if (impossible_regions := self.regions_compatible(ev.aff_regions)).size > 0:
            raise ValueError(
                "Some affected sectors of the event are not in the model : {}".format(
                    impossible_regions
                )
            )
        if (impossible_sectors := self.sectors_compatible(ev.aff_sectors)).size > 0:
            raise ValueError(
                "Some affected sectors of the event are not in the model : {}".format(
                    impossible_sectors
                )
            )
        if isinstance(ev, EventKapitalRebuild):
            if (
                impossible_sectors := self.sectors_compatible(
                    ev.rebuilding_sectors.index
                )
            ).size > 0:
                raise ValueError(
                    "Some affected sectors of the event are not in the model : {}".format(
                        impossible_sectors
                    )
                )
        if isinstance(ev, EventKapitalDestroyed):
            if ev.event_monetary_factor != self.model.monetary_factor:
                warnings.warn(
                    f"Event monetary factors ({ev.event_monetary_factor}), differs from model monetary factor ({self.model.monetary_factor}). Will automatically adjust."
                )

    def regions_compatible(self, regions: RegionsList):
        """Checks if given regions are all present in the simulation context.

        Parameters
        ----------
        regions : RegionsList
            The regions to checks.

        Returns
        -------

        set
            The set of regions not present in the model's regions.

        """
        return np.setdiff1d(regions, self.model.regions)

    def sectors_compatible(self, sectors: SectorsList):
        """Checks if given sectors are all present in the simulation context.

        Parameters
        ----------
        sectors : SectorsList
            The sectors to checks.

        Returns
        -------

        set
            The set of sectors not present in the model's regions.

        """
        return np.setdiff1d(sectors, self.model.sectors)

    def _check_happening_events(self) -> None:
        """Updates the status of all events.

        Check the `all_events` attribute and `current_temporal_unit` and
        updates the events accordingly (ie if they happened, if they can start
        rebuild/recover)

        """
        new_reb_event = False
        for event_tracker in self._event_tracking:
            if event_tracker.status == "pending":
                if (
                    (self.current_temporal_unit - self.model.n_temporal_units_by_step)
                    <= event_tracker.event.occurrence
                    <= self.current_temporal_unit
                ):
                    logger.info(
                        f"Temporal_Unit : {self.current_temporal_unit} ~ Shocking model with new event"
                    )
                    logger.info(
                        f"Affected regions are : {event_tracker.event.aff_regions.to_list()}"
                    )
                    event_tracker._status = "happening"
        for event_tracker in self._event_tracking:
            if event_tracker.status == "happening":
                if self.current_temporal_unit >= (
                    event_tracker.event.occurrence + event_tracker.event.duration
                ):
                    if isinstance(event_tracker.event, EventKapitalRebuild):
                        new_reb_event = True
                        self.model._n_rebuilding_events += 1
                        event_tracker._rebuild_id = self.model._n_rebuilding_events - 1
                        event_tracker._status = "rebuilding"
                        logger.info(
                            "Temporal_Unit : {} ~ Event named {} that occurred at {} in {} for {} damages has started rebuilding".format(
                                self.current_temporal_unit,
                                event_tracker.event.name,
                                event_tracker.event.occurrence,
                                event_tracker.event.aff_regions.to_list(),
                                event_tracker.impact_vector.sum(),
                            )
                        )

                    elif isinstance(
                        event_tracker.event, (EventKapitalRecover, EventArbitraryProd)
                    ):
                        event_tracker._status = "recovering"
                        logger.info(
                            "Temporal_Unit : {} ~ Event named {} that occurred at {} in {} for {} damages has started recovering".format(
                                self.current_temporal_unit,
                                event_tracker.event.name,
                                event_tracker.event.occurrence,
                                event_tracker.event.aff_regions.to_list(),
                                event_tracker.impact_vector.sum(),
                            )
                        )

                    else:
                        event_tracker._status = "finished"
                        logger.info(
                            "Temporal_Unit : {} ~ Event named {} that occurred at {} in {} is now considered finished".format(
                                self.current_temporal_unit,
                                event_tracker.event.name,
                                event_tracker.event.occurrence,
                                event_tracker.event.aff_regions.to_list(),
                            )
                        )

        self.update_prod_cap_delta_tot()
        if new_reb_event:
            self.model._chg_events_number()

        self.update_rebuild_demand()

    def rebuild_events(self):
        """Updates rebuilding events from model's production dedicated to rebuilding.

        This method loops through the event trackers and update the events depending
        on their allocated rebuilding production. If the received production is sufficient,
        then the events are flagged as "finished", and removed from the tracker.

        Raises
        ------
        RuntimeError
            Raised if an event tracker has no rebuid_id (should not happen).

        """

        events_rebuilt_ids = []
        for event_tracker in self._event_tracking:
            if event_tracker.status == "rebuilding":
                if event_tracker._rebuild_id is None:
                    raise RuntimeError(
                        "Rebuilding event has no rebuilding id, which should not happen."
                    )
                assert self.model.rebuild_prod is not None
                event_tracker.receive_indus_rebuilding(
                    self.model.rebuild_prod_indus_event(event_tracker._rebuild_id)
                )

                if event_tracker.rebuild_demand_house is not None:
                    event_tracker.receive_house_rebuilding(
                        self.model.rebuild_prod_house_event(event_tracker._rebuild_id)
                    )
                if (
                    event_tracker._house_dmg is None
                    and event_tracker._indus_dmg is None
                ):
                    event_tracker._status = "finished"
                    events_rebuilt_ids.append(event_tracker._rebuild_id)
                    self._events_to_rebuild -= 1

        if len(events_rebuilt_ids) > 0:
            events_rebuilt_ids = sorted(events_rebuilt_ids)
            non_reb_events = sorted([ev for ev in self._event_tracking if ev._rebuild_id is not None and ev._rebuild_id not in events_rebuilt_ids], key=lambda x: x._rebuild_id)  # type: ignore # Because lsp cannot understand rebuild_id is not none here.
            for ev_to_rm in events_rebuilt_ids:
                for evnt_trck in non_reb_events:
                    if evnt_trck._rebuild_id > ev_to_rm:
                        evnt_trck._rebuild_id -= 1

    def recover_events(self):
        """Updates recovering events with their recovery function.

        This method loops through the event trackers and update the events depending
        on their recovery function.

        Raises
        ------
        RuntimeError
            Raised if an event tracker has no recovery function (should not happen).

        """
        for event_tracker in self._event_tracking:
            if event_tracker.status == "recovering":
                if event_tracker.event.recovery_function is None:
                    raise RuntimeError(
                        "Recovering event has no recovery function, which should not happen."
                    )
                event_tracker.recover()

    def update_rebuild_demand(self):
        r"""Computes and updates total rebuilding demand based on a list of events.

        Computes and updates the model rebuilding demand from the event tracker. Only events
        tagged as rebuildable are accounted for. Both `house_rebuild_demand` and
        `indus_rebuild_demand` are updated.

        """

        if not isinstance(self._event_tracking, list):
            ValueError(
                f"Setting tot_rebuild_demand can only be done with a list of events self._event_tracking, not a {type(self._event_tracking)}"
            )
        if "rebuilding" in [
            event_tracker.status for event_tracker in self._event_tracking
        ]:
            _rebuilding_demand = np.zeros(
                shape=(
                    self.model.n_regions * self.model.n_sectors,
                    (
                        self.model.n_regions * self.model.n_sectors
                        + self.model.n_regions * self.model.n_fd_cat
                    )
                    * self.model._n_rebuilding_events,
                )
            )
            for evnt_trck in self._event_tracking:
                if evnt_trck._rebuild_id is not None:
                    _rebuilding_demand[
                        :,
                        (self.model.n_regions * self.model.n_sectors)
                        * (evnt_trck._rebuild_id) : (
                            self.model.n_regions * self.model.n_sectors
                        )
                        * (evnt_trck._rebuild_id + 1),
                    ] = evnt_trck.distributed_reb_dem_indus_tau
                    if evnt_trck.households_damages is not None:
                        _rebuilding_demand[
                            :,
                            (self.model.n_regions * self.model.n_sectors)
                            * self.model._n_rebuilding_events : (
                                self.model.n_regions * self.model.n_sectors
                                + self.model.n_regions * self.model.n_fd_cat
                            )
                            * (evnt_trck._rebuild_id + 1),
                        ] = evnt_trck.distributed_reb_dem_house_tau

            self.model.rebuild_demand = _rebuilding_demand

    def update_productive_capital_lost(self):
        r"""Computes current capital lost and updates production delta accordingly.

        Computes and sets the current stock of capital lost by each industry of
        the model due to the events. Also update the production
        capacity lost accordingly, by computing the ratio of capital lost to
        capital stock.

        """

        if DEBUG_TRACE:
            logger.debug("Updating productive_capital lost from list of events")
        source = [
            ev
            for ev in self._event_tracking
            if (ev.status in ["happening", "rebuilding", "recovering"])
        ]
        productive_capital_lost = np.add.reduce(
            np.array([e._indus_dmg for e in source if e is not None])
        )
        if not isinstance(productive_capital_lost, np.number):
            self.model.productive_capital_lost = productive_capital_lost
        else:
            self.model.productive_capital_lost = np.zeros_like(self.model.X_0)

    def update_prod_cap_delta_arb(self):
        r"""Computes and sets the loss of production capacity from "arbitrary" sources.

        .. warning::
           If multiple events impact the same industry, only the maximum loss is
           accounted for.

        """
        source = [
            ev
            for ev in self._event_tracking
            if (ev.status in ["happening", "recovering"])
        ]
        event_arb = np.array(
            [
                ev._prod_delta_from_arb
                for ev in source
                if ev._prod_delta_from_arb is not None
            ]
        )
        if event_arb.size == 0:
            self.model._prod_cap_delta_arbitrary = np.zeros_like(self.model.X_0)
        else:
            self.model._prod_cap_delta_arbitrary = np.maximum.reduce(event_arb)

    def update_prod_cap_delta_tot(self):
        r"""Computes and sets the loss of production capacity from both "arbitrary" sources and
        capital destroyed sources.

        """
        if DEBUG_TRACE:
            logger.debug("Updating total production delta")

        self.update_productive_capital_lost()
        self.update_prod_cap_delta_arb()

    @property
    def production_realised(self) -> pd.DataFrame:
        """Returns the evolution of the production realised by each industry (region,sector) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the production realised, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._production_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def production_capacity(self) -> pd.DataFrame:
        """Returns the evolution of the production capacity of each industry (region,sector) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the production capacity, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._production_cap_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def final_demand(self) -> pd.DataFrame:
        """Return the evolution of final demand asked of each industry as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the final demand asked, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._final_demand_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def intermediate_demand(self) -> pd.DataFrame:
        """Returns the evolution of intermediate demand asked of each industry (Total orders) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the intermediate demand asked, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._io_demand_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def rebuild_demand(self) -> pd.DataFrame:
        """Returns the evolution of rebuild demand asked of each industry (Total orders) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the rebuild demand asked, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._rebuild_demand_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def overproduction(self) -> pd.DataFrame:
        """Returns the evolution of the overproduction factor of each industry (region,sector) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the overproduction factor, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._overproduction_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def final_demand_unmet(self) -> pd.DataFrame:
        """Returns the evolution of the final demand that could not be answered by industries as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the final demand not met, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._final_demand_unmet_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def rebuild_prod(self) -> pd.DataFrame:
        """Returns the production allocated for the rebuilding demand by each industry (region,sector) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the production allocated, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._rebuild_production_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    @property
    def inputs_stocks(self) -> pd.DataFrame:
        """Returns the evolution of the inventory amount of each input for each industry (region,sector) as a DataFrame. Not this is not available if "record_stocks" is not set to True,
        as the DataFrame can be quite large for "classic" MRIOTs.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the amount in inventory, the columns are the industries
            and the index are the step and input considered (MultiIndex).

        """
        return pd.DataFrame(
            self._inputs_evolution.reshape(
                self.n_temporal_units_to_sim * self.model.n_sectors, -1
            ),
            columns=self.model.industries,
            copy=True,
            index=pd.MultiIndex.from_product(
                [list(range(self.n_temporal_units_to_sim)), self.model.sectors],
                names=["step", "input"],
            ),
        )

    @property
    def limiting_inputs(self) -> pd.DataFrame:
        """Returns the evolution of the inputs lacking for each industry (region,sector) as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is a boolean set to 1 if considered input constrains production, the columns are the industries
            and the index are the step and input considered (MultiIndex).

        """
        return pd.DataFrame(
            self._limiting_inputs_evolution.reshape(
                self.n_temporal_units_to_sim * self.model.n_sectors, -1
            ),
            columns=self.model.industries,
            copy=True,
            index=pd.MultiIndex.from_product(
                [list(range(self.n_temporal_units_to_sim)), self.model.sectors],
                names=["step", "input"],
            ),
        )

    @property
    def productive_capital_to_recover(self) -> pd.DataFrame:
        """Retursn the evolution of remaining capital destroyed/to recover for each industry (region,sector) if it exists as a DataFrame.

        Returns
        -------
            pd.DataFrame: A pandas DataFrame where the value is the amount of capital (still) destroyed, the columns are the industries
            and the index is the step considered.

        """
        return pd.DataFrame(
            self._regional_sectoral_productive_capital_destroyed_evolution,
            columns=self.model.industries,
            copy=True,
        ).rename_axis("step")

    def reset_sim_with_same_events(self):
        """Resets the model to its initial status (without removing the events). [WIP]"""

        raise NotImplementedError(
            "This methods has not been reimplemented for the updated version. Simplest way to reset is to recreate the Simulation and Model objects."
        )

        logger.info("Resetting model to initial status (with same events)")
        self.current_temporal_unit = 0
        self._monotony_checker = 0
        self._n_checks = 0
        self.n_temporal_units_simulated = 0
        self.has_crashed = False
        self.equi = {
            (int(0), int(0), "production"): "equi",
            (int(0), int(0), "stocks"): "equi",
            (int(0), int(0), "rebuilding"): "equi",
        }
        self._reset_records()
        self.model.reset_module()

    def reset_sim_full(self):
        """Resets the model to its initial status and remove all events."""

        raise NotImplementedError(
            "This methods has not been reimplemented for the updated version. Simplest way to reset is to recreate the Simulation and Model objects."
        )

        self.reset_sim_with_same_events()
        logger.info("Resetting events")
        self.all_events = []
        self.currently_happening_events = []
        self.events_timings = set()

    def write_index(self, index_file: Union[str, pathlib.Path]):
        """Writes the index of the dataframes used in the model in a json file.

        See :meth:`~boario.model_base.ARIOBaseModel.write_index` for a more detailed documentation.

        Parameters
        ----------
        index_file : Union[str, pathlib.Path]
            name of the file to save the indexes to.

        """
        self.model.write_index(index_file)

    def _write_production(self) -> None:
        """Saves the current production vector to the memmap."""
        self._production_evolution[self.current_temporal_unit] = self.model.production

    def _write_rebuild_prod(self) -> None:
        """Saves the current rebuilding production vector to the memmap."""
        to_write = np.full(self.model.n_regions * self.model.n_sectors, 0.0)
        if self.model.rebuild_prod_tot is not None:
            self._rebuild_production_evolution[self.current_temporal_unit] = (
                self.model.rebuild_prod_tot
            )
        else:
            self._rebuild_production_evolution[self.current_temporal_unit] = to_write

    def _write_productive_capital_lost(self) -> None:
        """Saves the current remaining productive_capital to rebuild vector to the memmap."""
        self._regional_sectoral_productive_capital_destroyed_evolution[
            self.current_temporal_unit
        ] = self.model.productive_capital_lost

    def _write_production_max(self) -> None:
        """Saves the current production capacity vector to the memmap."""
        self._production_cap_evolution[self.current_temporal_unit] = (
            self.model.production_cap
        )

    def _write_io_demand(self) -> None:
        """Saves the current (total per industry) intermediate demand vector to the memmap."""
        self._io_demand_evolution[self.current_temporal_unit] = (
            self.model.intermediate_demand_tot
        )

    def _write_final_demand(self) -> None:
        """Saves the current (total per industry) final demand vector to the memmap."""
        self._final_demand_evolution[self.current_temporal_unit] = (
            self.model.final_demand_tot
        )

    def _write_rebuild_demand(self) -> None:
        """Saves the current (total per industry) rebuilding demand vector to the memmap."""
        to_write = np.full(self.model.n_regions * self.model.n_sectors, 0.0)
        if len(r_dem := self.model.rebuild_demand_tot) > 0:
            self._rebuild_demand_evolution[self.current_temporal_unit] = r_dem  # type: ignore
        else:
            self._rebuild_demand_evolution[self.current_temporal_unit] = to_write  # type: ignore

    def _write_overproduction(self) -> None:
        """Saves the current overproduction vector to the memmap."""
        self._overproduction_evolution[self.current_temporal_unit] = self.model.overprod

    def _write_final_demand_unmet(self) -> None:
        """Saves the unmet final demand (for this step) vector to the memmap."""
        self._final_demand_unmet_evolution[self.current_temporal_unit] = (
            self.model.final_demand_not_met
        )

    def _write_stocks(self) -> None:
        """Saves the current inputs stock matrix to the memmap."""
        self._inputs_evolution[self.current_temporal_unit] = self.model.inputs_stock

    def _write_limiting_stocks(self, limiting_stock: np.ndarray) -> None:
        """Saves the current limiting inputs matrix to the memmap."""
        self._limiting_inputs_evolution[self.current_temporal_unit] = limiting_stock  # type: ignore

    def _flush_memmaps(self) -> None:
        """Saves files to records."""
        for attr in self._files_to_record:
            if not hasattr(self, attr):
                raise RuntimeError(
                    f"{attr} should be a member yet it isn't. This shouldn't happen."
                )
            else:
                getattr(self, attr).flush()

    def _init_records(self, save_records):
        for rec in self.__possible_records:
            if rec == "inputs_stocks" and not self._register_stocks:
                logger.debug("Will not save inputs stocks")
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
                if save:
                    memmap_array = TempMemmap(
                        filename=(self.records_storage / filename),
                        dtype=dtype,
                        mode="w+",
                        shape=shape,
                        save=save,
                    )
                    self._files_to_record.append(attr_name)
                else:
                    memmap_array = np.ndarray(shape=shape, dtype=dtype, order="C")
                    self._vars_to_record.append(attr_name)
                memmap_array.fill(fillv)
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


@overload
def _thin_to_wide(
    thin: pd.Series, long_index: pd.Index, long_columns: None = None
) -> pd.Series: ...


@overload
def _thin_to_wide(
    thin: pd.DataFrame,
    long_index: pd.Index,
    long_columns: pd.Index,
) -> pd.DataFrame: ...


def _thin_to_wide(
    thin: pd.Series | pd.DataFrame,
    long_index: pd.Index,
    long_columns: pd.Index | None = None,
) -> pd.Series | pd.DataFrame:
    """
    Converts a "thin" (sparse) DataFrame or Series into a "wide" (dense) format with a specified index
    and optional column structure, filling missing values with zeros.

    Parameters
    ----------
    thin : pd.Series or pd.DataFrame
        The sparse data to convert to a wide format. If `thin` is a DataFrame, `long_columns`
        must be specified.
    long_index : pd.Index
        The index to use for the resulting wide-format data. All values in `thin` are
        realigned to this index.
    long_columns : pd.Index, optional
        The columns to use for the resulting wide-format DataFrame. This parameter is
        required if `thin` is a DataFrame and ignored if `thin` is a Series.

    Returns
    -------
    pd.Series or pd.DataFrame
        The wide-format representation of `thin`, with `long_index` as its index and
        `long_columns` as its columns (if applicable). Missing values are filled with zeros.

    Raises
    ------
    ValueError
        If `thin` is a DataFrame and `long_columns` is not provided.

    Examples
    --------
    >>> thin_series = pd.Series([1, 2], index=[0, 1])
    >>> long_index = pd.Index([0, 1, 2, 3])
    >>> _thin_to_wide(thin_series, long_index)
    0    1.0
    1    2.0
    2    0.0
    3    0.0
    dtype: float64

    >>> thin_df = pd.DataFrame([[1, 2], [3, 4]], index=[0, 1], columns=['A', 'B'])
    >>> long_columns = pd.Index(['A', 'B', 'C'])
    >>> _thin_to_wide(thin_df, long_index, long_columns)
         A    B    C
    0  1.0  2.0  0.0
    1  3.0  4.0  0.0
    2  0.0  0.0  0.0
    3  0.0  0.0  0.0

    """
    if isinstance(thin, pd.Series):
        wide = pd.Series(index=long_index, dtype=thin.dtype)
    elif isinstance(thin, pd.DataFrame):
        if long_columns is None:
            raise ValueError(
                "long_columns argument cannot be None when widening a DataFrame."
            )
        wide = pd.DataFrame(
            index=long_index, columns=long_columns, dtype=thin.dtypes.iloc[0]
        )
    wide.fillna(0, inplace=True)
    if isinstance(thin, pd.DataFrame):
        wide.loc[thin.index, thin.columns] = thin.values
    else:
        wide.loc[thin.index] = thin.values
    return wide


def _equal_distribution(
    affected: pd.Index | None, addressed_to: pd.Index
) -> pd.DataFrame:
    """
    Creates a DataFrame representing an equal distribution of values across
    specified indices.

    Parameters
    ----------
    affected : pd.Index or None
        The index for the columns, representing the entities affected by the distribution.
    addressed_to : pd.Index
        The index for the rows, representing the entities receiving the distribution.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each element in `addressed_to` receives an equal share
        of 1 divided by the length of `addressed_to` for each `affected` column.

    Examples
    --------
    >>> affected = pd.Index(['A', 'B'])
    >>> addressed_to = pd.Index(['X', 'Y', 'Z'])
    >>> _equal_distribution(affected, addressed_to)
          A         B
    X  0.333333  0.333333
    Y  0.333333  0.333333
    Z  0.333333  0.333333
    """
    if addressed_to.empty:
        raise ValueError("addressed_to index cannot be empty.")
    ret = pd.DataFrame(0.0, index=addressed_to, columns=affected)
    ret.loc[addressed_to, affected] = 1 / len(addressed_to)
    return ret


def _normalize_distribution(
    dist: pd.DataFrame | pd.Series, affected: pd.Index | None, addressed_to: pd.Index
) -> pd.DataFrame:
    """
    Normalizes a distribution so that values across specified indices sum to 1 for each
    entity in `addressed_to`.

    Parameters
    ----------
    dist : pd.DataFrame or pd.Series
        The initial distribution to normalize. If a DataFrame, it should align with both
        `affected` and `addressed_to`. If a Series, it should align with `addressed_to`.
    affected : pd.Index or None
        The index for the columns, representing entities affected by the distribution.
    addressed_to : pd.Index
        The index for the rows, representing entities receiving the distribution.

    Returns
    -------
    pd.DataFrame
        A DataFrame where values across `addressed_to` entities are normalized to sum to 1
        across each `affected` column.

    Raises
    ------
    ValueError
        If `dist` is not a Series or DataFrame.

    Examples
    --------
    >>> dist = pd.Series([2, 3, 5], index=['X', 'Y', 'Z'])
    >>> affected = pd.Index(['A'])
    >>> addressed_to = pd.Index(['X', 'Y', 'Z'])
    >>> _normalize_distribution(dist, affected, addressed_to)
          A
    X  0.2
    Y  0.3
    Z  0.5
    """
    ret = pd.DataFrame(0.0, index=addressed_to, columns=affected)
    if isinstance(dist, pd.DataFrame):
        dist_sq = dist.squeeze()
    else:
        dist_sq = dist
    if isinstance(dist_sq, pd.Series):
        ret.loc[addressed_to, :] = (
            dist_sq.loc[addressed_to]
            .groupby(level=1)
            .transform(lambda x: x / sum(x))
            .values[:, None]
        )
        return ret
    elif isinstance(dist_sq, pd.DataFrame):
        ret.loc[addressed_to, affected] = (
            dist_sq.loc[addressed_to, affected]
            .groupby(level=1)
            .transform(lambda x: x / sum(x))
        )
        return ret
    else:
        raise ValueError("given distribution should be a Series or a DataFrame.")


class EventTracker:
    """
    Tracks the state and progression of an event within a simulation, including
    damages, recovery, and rebuilding demands.

    Parameters
    ----------
    sim : Simulation
        The simulation object where the event is tracked.
    source_event : Event
        The source event to be tracked.
    indus_rebuild_distribution : pd.DataFrame, None, or Literal["equal"], optional
        Specifies the distribution of rebuilding demand across industries, default is None.
    house_rebuild_distribution : pd.DataFrame, None, or Literal["equal"], optional
        Specifies the distribution of rebuilding demand across households, default is None.

    Attributes
    ----------
    sim : Simulation
        Reference to the simulation where the event takes place.
    event : Event
        The event being tracked.
    _status : {"pending", "happening", "recovering", "rebuilding", "finished"}
        Current status of the event.
    _prod_delta_from_arb_0 : pd.Series or None
        Initial production delta for arbitrary production events.
    _prod_delta_from_arb : pd.Series or None
        Current production delta for arbitrary production events.
    _indus_dmg_0 : pd.Series or None
        Initial industrial damages caused by the event.
    _indus_dmg : pd.Series or None
        Current industrial damages from the event.
    _rebuildable : bool
        Indicates if the event is eligible for rebuilding.
    _rebuild_id : int or None
        Unique identifier for rebuilding actions.
    _rebuild_shares : pd.Series or None
        Distribution shares for rebuilding demands.
    _rebuild_demand_indus_0 : pd.DataFrame or None
        Initial rebuilding demand for industries.
    _rebuild_demand_house_0 : pd.DataFrame or None
        Initial rebuilding demand for households.
    _rebuild_demand_indus : pd.DataFrame or None
        Current rebuilding demand for industries.
    _distributed_reb_dem_indus : pd.DataFrame or None
        Distributed demand for rebuilding industries.
    _rebuild_demand_house : pd.DataFrame or None
        Current rebuilding demand for households.
    _distributed_reb_dem_house : pd.DataFrame or None
        Distributed demand for rebuilding households.
    _recovery_function : Callable or None
        Function to compute recovery over time.
    _house_dmg_0 : pd.Series or None
        Initial household damages from the event.
    _house_dmg : pd.Series or None
        Current household damages.
    """

    def __init__(
        self,
        sim: Simulation,
        source_event: Event,
        indus_rebuild_distribution: pd.DataFrame | None | Literal["equal"] = None,
        house_rebuild_distribution: pd.DataFrame | None | Literal["equal"] = None,
    ):
        self.sim: Simulation = sim
        self.event = source_event
        self._status: (
            Literal["pending"]
            | Literal["happening"]
            | Literal["recovering"]
            | Literal["rebuilding"]
            | Literal["finished"]
        ) = "pending"

        self._prod_delta_from_arb_0: pd.Series | None = None
        self._prod_delta_from_arb: pd.Series | None = None
        self._indus_dmg_0: pd.Series | None = None
        self._indus_dmg: pd.Series | None = None
        self._rebuildable: bool = False
        self._rebuild_id: int | None = None
        self._rebuild_shares: pd.Series | None = None
        self._rebuild_demand_indus_0: pd.DataFrame | None = None
        self._rebuild_demand_house_0: pd.DataFrame | None = None
        self._rebuild_demand_indus: pd.DataFrame | None = None
        # self._reb_dem_indus_distribution: npt.NDArray | None = None
        self._distributed_reb_dem_indus: pd.DataFrame | None = None
        self._rebuild_demand_house: pd.DataFrame | None = None
        # self._reb_dem_house_distribution: pd.DataFrame | None = None
        self._distributed_reb_dem_house: pd.DataFrame | None = None
        self._recovery_function: Callable | None = None
        self._house_dmg_0: pd.Series | None = None
        self._house_dmg: pd.Series | None = None

        impact = source_event.impact.copy()
        if isinstance(source_event, EventKapitalDestroyed):
            if source_event.event_monetary_factor != sim.model.monetary_factor:
                impact = impact * (
                    source_event.event_monetary_factor / sim.model.monetary_factor
                )

        if isinstance(source_event, EventKapitalDestroyed):
            self._indus_dmg_0 = _thin_to_wide(impact, self.sim.model.industries)
            self._indus_dmg = self._indus_dmg_0.copy()
            if source_event.impact_households is not None:
                impact_house = source_event.impact_households.copy()
                if source_event.event_monetary_factor != sim.model.monetary_factor:
                    impact_house = impact_house * (
                        source_event.event_monetary_factor / sim.model.monetary_factor
                    )
                self._house_dmg_0 = _thin_to_wide(
                    impact_house, self.sim.model.all_regions_fd
                )
                self._house_dmg = self._house_dmg_0.copy()

        if isinstance(source_event, EventKapitalRebuild):
            self._init_distrib("indus", indus_rebuild_distribution)
            assert self._indus_dmg_0 is not None

            self._rebuild_demand_indus_0 = pd.DataFrame(
                source_event.rebuilding_sectors.values[:, None]
                * source_event.impact.values,
                index=source_event.rebuilding_sectors.index,
                columns=source_event.impact.index,
            )
            self._rebuild_demand_indus_0.rename_axis(
                index="rebuilding sector", inplace=True
            )
            self._rebuild_demand_indus_0 *= source_event.rebuilding_factor
            self._rebuild_demand_indus = self._rebuild_demand_indus_0.copy()
            self.rebuild_dem_indus_distribution = self._reb_dem_indus_distribution

            if source_event.impact_households is not None:
                self._init_distrib("house", house_rebuild_distribution)
                self._rebuild_demand_house_0 = pd.DataFrame(
                    source_event.rebuilding_sectors.values[:, None]
                    * impact_house.values,
                    index=source_event.rebuilding_sectors.index,
                    columns=source_event.impact_households.index,
                )
                self._rebuild_demand_house_0.rename_axis(
                    index="rebuilding sector", inplace=True
                )
                self._rebuild_demand_house_0 *= source_event.rebuilding_factor
                self._rebuild_demand_house = self._rebuild_demand_house_0.copy()
                self.rebuild_dem_house_distribution = self._reb_dem_house_distribution

        if isinstance(self.event, EventKapitalRecover):
            self._recovery_function_indus = partial(
                self.event.recovery_function,
                init_impact_stock=self._indus_dmg_0,
                recovery_tau=self.event.recovery_tau,
            )
            if self._house_dmg_0 is not None:
                self._recovery_function_house = partial(
                    self.event.recovery_function,
                    init_impact_stock=self._house_dmg_0,
                    recovery_tau=self.event.recovery_tau,
                )

        if isinstance(self.event, EventArbitraryProd):
            self._prod_delta_from_arb_0 = _thin_to_wide(
                source_event.impact.copy(), self.sim.model.industries
            )
            self._prod_delta_from_arb = self._prod_delta_from_arb_0.copy()
            self._recovery_function_arb_delta = partial(
                self.event.recovery_function,
                init_impact_stock=self._prod_delta_from_arb_0,
                recovery_tau=self.event.recovery_tau,
            )

    def _init_distrib(
        self,
        dtype: Literal["indus"] | Literal["house"],
        distrib: pd.DataFrame | None | Literal["equal"],
    ):
        """
        Initializes the distribution for rebuilding demand based on type.

        Parameters
        ----------
        dtype : {"indus", "house"}
             Specifies the type of distribution to initialize, either "indus" for industries or "house" for households.
        distrib : pd.DataFrame, None, or "equal"
             The distribution data for rebuilding demand. If None, distribution is based on intermediate demand; if "equal," distribution is equal across all sectors.
        """
        if dtype == "indus":
            if distrib is None:
                self._reb_dem_indus_distribution = _normalize_distribution(
                    self.sim.model.mriot.Z,
                    affected=self.event.aff_industries,
                    addressed_to=pd.MultiIndex.from_product(
                        [self.sim.model.regions, self.event.rebuilding_sectors.index],
                        names=["region", "rebuilding sector"],
                    ),
                )
            elif distrib == "equal":
                self._reb_dem_indus_distribution = _equal_distribution(
                    affected=self.event.aff_industries,
                    addressed_to=pd.MultiIndex.from_product(
                        [self.sim.model.regions, self.event.rebuilding_sectors.index],
                        names=["region", "rebuilding sector"],
                    ),
                )
            else:
                self._reb_dem_indus_distribution = _normalize_distribution(
                    distrib,
                    affected=self.event.aff_industries,
                    addressed_to=pd.MultiIndex.from_product(
                        [self.sim.model.regions, self.event.rebuilding_sectors.index],
                        names=["region", "rebuilding sector"],
                    ),
                )

        if dtype == "house":
            if distrib is None:
                self._reb_dem_house_distribution = _normalize_distribution(
                    self.sim.model.mriot.Y,
                    affected=self.event._aff_final_demands,
                    addressed_to=pd.MultiIndex.from_product(
                        [self.sim.model.regions, self.event.rebuilding_sectors.index],
                        names=["region", "rebuilding sector"],
                    ),
                )
            elif distrib == "equal":
                self._reb_dem_house_distribution = _equal_distribution(
                    affected=self.event._aff_final_demands,
                    addressed_to=pd.MultiIndex.from_product(
                        [self.sim.model.regions, self.event.rebuilding_sectors.index],
                        names=["region", "rebuilding sector"],
                    ),
                )
            else:
                self._reb_dem_house_distribution = _normalize_distribution(
                    distrib,
                    affected=self.event._aff_final_demands,
                    addressed_to=pd.MultiIndex.from_product(
                        [self.sim.model.regions, self.event.rebuilding_sectors.index],
                        names=["region", "rebuilding sector"],
                    ),
                )

    @cached_property
    def impact_vector(self) -> pd.Series:
        """
        Returns the impact vector of the event.

        Returns
        -------
        pd.Series
             The event impact, formatted as a series with all industries as index.
        """
        return _thin_to_wide(self.event.impact, self.sim.model.industries)

    @property
    def productive_capital_dmg_init(self) -> pd.Series | None:
        """
        Gets the initial productive capital damage from the event.

        Returns
        -------
        pd.Series or None
            Series of initial damages to productive capital; None if not available.
        """
        return self._indus_dmg_0

    @property
    def productive_capital_dmg(self) -> pd.Series | None:
        """
        Gets the current productive capital damage.

        Returns
        -------
        pd.Series or None
            Series of current damages to productive capital; None if not available.
        """
        return self._indus_dmg

    @property
    def households_damages_init(self) -> pd.Series | None:
        """
        Gets the initial household damage for the event.

        Returns
        -------
        pd.Series or None
            Series of initial damages to households; None if not available.
        """
        return self._house_dmg_0

    @property
    def households_damages(self) -> pd.Series | None:
        """
        Gets the current household damage for the event.

        Returns
        -------
        pd.Series or None
            Series of initial damages to households; None if not available.
        """
        return self._house_dmg

    @property
    def rebuild_demand_indus(self) -> pd.DataFrame | None:
        """
        Gets the current rebuilding demand for industries.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of rebuilding demands for industries; None if not applicable.
        """
        return self._rebuild_demand_indus

    @property
    def rebuild_demand_house(self) -> pd.DataFrame | None:
        """
        Gets the current rebuilding demand for households.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of rebuilding demands for households; None if not applicable.
        """
        return self._rebuild_demand_house

    @property
    def prod_delta_arbitrary(self) -> pd.Series | None:
        """
        Gets the current production delta for arbitrary production loss events.

        Returns
        -------
        pd.Series or None
            Series representing production changes; None if not available.
        """
        return self._prod_delta_from_arb

    @prod_delta_arbitrary.setter
    def prod_delta_arbitrary(self, value: pd.Series | None):
        self._prod_delta_from_arb = value

    @property
    def status(
        self,
    ) -> (
        Literal["pending"]
        | Literal["happening"]
        | Literal["recovering"]
        | Literal["rebuilding"]
        | Literal["finished"]
    ):
        """
        Gets the current status of the event.

        Returns
        -------
        {"pending", "happening", "recovering", "rebuilding", "finished"}
            The status indicating the stage of the event.
        """
        return self._status

    def _compute_distributed_demand(
        self, demand_by_sectors, distribution, rebuilding_industries
    ):
        """
        Computes the distributed demand based on sector demand and distribution.

        Parameters
        ----------
        demand_by_sectors : pd.DataFrame
            The demand data for sectors.
        distribution : pd.DataFrame
            Distribution weights for sectors.
        rebuilding_industries : pd.Index
            Index of rebuilding industries.

        Returns
        -------
        pd.DataFrame
            DataFrame of distributed demand across regions and sectors.
        """
        multi_index = pd.MultiIndex.from_product(
            [self.sim.model.regions, rebuilding_industries],
            names=["region", "rebuilding sector"],
        )
        demand_by_sectors = demand_by_sectors.reindex(multi_index, level=1)
        if (distribution.index.sort_values() != multi_index.sort_values()).any():
            distribution = distribution.reindex(multi_index, level=0)
        return demand_by_sectors.mul(distribution)

    @property
    def rebuild_dem_indus_distribution(self) -> pd.DataFrame | None:
        """
        Gets the distribution for rebuilding industry demand.

        Returns
        -------
        pd.DataFrame or None
            Distribution DataFrame for rebuilding industry demand; None if not set.
        """
        return self._reb_dem_indus_distribution

    @rebuild_dem_indus_distribution.setter
    def rebuild_dem_indus_distribution(self, value: pd.DataFrame):
        self._rebuild_dem_indus_distribution = value
        if self.rebuild_demand_indus is not None:
            self._distributed_reb_dem_indus = _thin_to_wide(
                self._compute_distributed_demand(
                    self.rebuild_demand_indus,
                    self._rebuild_dem_indus_distribution,
                    self.event.rebuilding_sectors.index,
                ),
                long_index=self.sim.model.industries,
                long_columns=self.sim.model.industries,
            )

    @property
    def distributed_reb_dem_indus_tau(self) -> pd.DataFrame | None:
        """
        Gets the current rebuilding demand for industries.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of rebuilding demands for industries; None if not applicable.
        """
        if self.event.rebuild_tau:
            reb_tau = self.event.rebuild_tau
        else:
            reb_tau = self.sim.model.rebuild_tau
        return self._distributed_reb_dem_indus * (
            self.sim.model.n_temporal_units_by_step / reb_tau
        )

    @property
    def distributed_reb_dem_house_tau(self) -> pd.DataFrame | None:
        """
        Gets the current rebuilding demand for households.

        Returns
        -------
        pd.DataFrame or None
            DataFrame of rebuilding demands for households; None if not applicable.
        """
        if self.event.rebuild_tau:
            reb_tau = self.event.rebuild_tau
        else:
            reb_tau = self.sim.model.rebuild_tau
        return self._distributed_reb_dem_house * (
            self.sim.model.n_temporal_units_by_step / reb_tau
        )

    @property
    def distributed_reb_dem_indus(self) -> pd.DataFrame | None:
        """
        Gets the distributed rebuilding demand for industries.

        Returns
        -------
        pd.DataFrame or None
            Distributed DataFrame for rebuilding industry demand; None if not set.
        """
        return self._distributed_reb_dem_indus

    def receive_indus_rebuilding(self, reb_prod: pd.DataFrame | npt.ArrayLike | None):
        """
        Processes and updates the industry rebuilding demand based on received production.

        Parameters
        ----------
        reb_prod : pd.DataFrame or npt.ArrayLike
            Rebuilding production to apply to industry demands.

        Raises
        ------
        ValueError
            If reb_prod is None, or if rebuilding demand does not exist or event is not a rebuilding event.
        """
        if reb_prod is None:
            raise ValueError("Trying to rebuild with None rebuilding prod.")
        if self._distributed_reb_dem_indus is None:
            raise ValueError("The rebuilding demand of this event does not exist.")
        if not isinstance(self.event, EventKapitalRebuild):
            raise ValueError("The event is not a rebuilding event.")
        self._distributed_reb_dem_indus -= reb_prod
        precision = int(math.log10(self.event.event_monetary_factor)) + 1
        self._distributed_reb_dem_indus = self._distributed_reb_dem_indus.round(
            precision
        )
        self._distributed_reb_dem_indus[self._distributed_reb_dem_indus < 0] = 0.0
        if not self._distributed_reb_dem_indus.values.any():
            self._distributed_reb_dem_indus = None
            self._rebuild_demand_indus = None
            self._indus_dmg = None
        else:
            self._rebuild_demand_indus = self._distributed_reb_dem_indus.groupby(
                "region"
            ).sum()
            self._indus_dmg = (
                self._rebuild_demand_indus.sum(axis=0) / self.event.rebuilding_factor
            )

    @property
    def rebuild_dem_house_distribution(self) -> pd.DataFrame | None:
        """
        Gets the distribution for rebuilding household demand.

        Returns
        -------
        pd.DataFrame or None
            Distribution DataFrame for rebuilding household demand; None if not set.
        """
        return self._reb_dem_house_distribution

    @rebuild_dem_house_distribution.setter
    def rebuild_dem_house_distribution(self, value: pd.DataFrame):
        self._rebuild_dem_house_distribution = value
        if self.rebuild_demand_house is not None:
            self._distributed_reb_dem_house = _thin_to_wide(
                self._compute_distributed_demand(
                    self.rebuild_demand_house,
                    self._rebuild_dem_house_distribution,
                    self.event.rebuilding_sectors.index,
                ),
                long_index=self.sim.model.industries,
                long_columns=self.sim.model.all_regions_fd,
            )

    @property
    def distributed_reb_dem_house(self) -> pd.DataFrame | None:
        """
        Gets the distributed rebuilding demand for households.

        Returns
        -------
        pd.DataFrame or None
            Distributed DataFrame for rebuilding household demand; None if not set.
        """
        return self._distributed_reb_dem_house

    def receive_house_rebuilding(self, reb_prod: pd.DataFrame | npt.ArrayLike | None):
        """
        Processes and updates the household rebuilding demand based on received production.

        Parameters
        ----------
        reb_prod : pd.DataFrame or npt.ArrayLike
            Rebuilding production to apply to household demands.

        Raises
        ------
        ValueError
            If reb_prod is None, or if household rebuilding demand does not exist or event is not a rebuilding event.
        """
        if reb_prod is None:
            raise ValueError("Trying to rebuild with None rebuilding prod.")

        if self._distributed_reb_dem_house is None:
            raise ValueError(
                "The household rebuilding demand of this event does not exist."
            )
        if not isinstance(self.event, EventKapitalRebuild):
            raise ValueError("The event is not a rebuilding event.")
        self._distributed_reb_dem_house -= reb_prod
        precision = int(math.log10(self.event.event_monetary_factor)) + 1
        self._distributed_reb_dem_house = self._distributed_reb_dem_house.round(
            precision
        )
        self._distributed_reb_dem_house[self._distributed_reb_dem_house < 0] = 0.0
        if not self._distributed_reb_dem_house.values.any():
            self._distributed_reb_dem_house = None
            self._rebuild_demand_house = None
            self._house_dmg = None
        else:
            self._rebuild_demand_house = self._distributed_reb_dem_house.groupby(
                "region"
            ).sum()
            self._house_dmg = (
                self._rebuild_demand_house.sum(axis=0) / self.event.rebuilding_factor
            )

    def recover(self):
        """
        Applies the recovery function to update the damages over time based on the recovery rate of the event.

        Raises
        ------
        ValueError
            If the event is not a recoverable type (i.e., not a capital or production recovery event).
        """
        if not isinstance(self.event, (EventKapitalRecover, EventArbitraryProd)):
            raise ValueError("The event is not a recoverable event.")
        if isinstance(self.event, EventKapitalRecover):
            precision = int(math.log10(self.event.event_monetary_factor)) + 1
            if self._indus_dmg is not None:
                self._indus_dmg = self._recovery_function_indus(
                    self.sim.current_temporal_unit
                    - (self.event.occurrence + self.event.duration)
                ).round(precision)
                if not self._indus_dmg.any():
                    self._indus_dmg = None
            if self._house_dmg is not None:
                self._house_dmg = self._recovery_function_house(
                    self.sim.current_temporal_unit
                    - (self.event.occurrence + self.event.duration)
                ).round(precision)
                if not self._house_dmg.any():
                    self._house_dmg = None
        if self._prod_delta_from_arb is not None:
            self._prod_delta_from_arb = self._recovery_function_arb_delta(
                self.sim.current_temporal_unit
                - (self.event.occurrence + self.event.duration)
            ).round(6)
            if not self._prod_delta_from_arb.any():
                self._prod_delta_from_arb = None
        if (
            self._indus_dmg is None
            and self._house_dmg is None
            and self._prod_delta_from_arb is None
        ):
            self._status = "finished"

    @property
    def recovery_function(self):
        """
        Gets the recovery function associated with the event.

        Returns
        -------
        Callable or None
            Function to calculate recovery over time, if applicable.
        """
        return self._recovery_function

    @cached_property
    def affected_sectors_idx(self) -> npt.NDArray:
        """
        Gets the index of sectors affected by the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing affected sectors.
        """
        return np.searchsorted(self.sim.model.sectors, self.event.aff_sectors)

    @cached_property
    def affected_regions_idx(self) -> npt.NDArray:
        """
        Gets the index of regions affected by the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing affected regions.

        Raises
        ------
        ValueError
            If some affected regions are not found in the model.
        """
        impossible_regions = np.setdiff1d(
            self.event.aff_regions, self.sim.model.regions
        )
        if impossible_regions.size > 0:
            raise ValueError(
                "Some affected regions of the event are not in the model : {}".format(
                    impossible_regions
                )
            )
        return np.searchsorted(self.sim.model.regions, self.event.aff_regions)

    @cached_property
    def affected_industries_idx(self) -> npt.NDArray:
        """
        Gets the index of industries affected by the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing affected industries.
        """
        return np.array(
            [
                np.size(self.sim.model.sectors) * ri + si
                for ri in self.affected_regions_idx
                for si in self.affected_sectors_idx
            ]
        )

    @cached_property
    def rebuilding_sectors_idx(self) -> npt.NDArray:
        """
        Gets the index of sectors involved in rebuilding for the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing rebuilding sectors.

        Raises
        ------
        ValueError
            If the event is not a rebuilding event or if some rebuilding sectors are not found in the model.
        """
        if not isinstance(self.event, EventKapitalRebuild):
            raise ValueError(
                "This event is not a rebuilding event and has no rebuilding sectors."
            )
        else:
            impossible_sectors = np.setdiff1d(
                self.event.rebuilding_sectors, self.sim.model.sectors
            )
        if impossible_sectors.size > 0:
            raise ValueError(
                "Some rebuilding sectors of the event are not in the model : {}".format(
                    impossible_sectors
                )
            )
        return np.searchsorted(self.sim.model.sectors, self.event.rebuilding_sectors)

    @cached_property
    def rebuilding_industries_idx_impacted(self) -> npt.NDArray:
        """
        Gets the index of rebuilding industries that were impacted by the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing rebuilding industries impacted by the event.
        """
        rebuilding_sectors_idx = self.rebuilding_sectors_idx
        affected_region_idx = self.affected_regions_idx
        return np.array(
            [
                np.size(self.sim.model.sectors) * ri + si
                for ri in affected_region_idx
                for si in rebuilding_sectors_idx
            ],
            dtype="int64",
        )

    @cached_property
    def rebuilding_industries_idx_not_impacted(self) -> npt.NDArray:
        """
        Gets the index of rebuilding industries not impacted by the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing rebuilding industries not impacted by the event.
        """
        rebuilding_sectors_idx = self.rebuilding_sectors_idx
        affected_region_idx = self.affected_regions_idx
        return np.array(
            [
                np.size(self.sim.model.sectors) * ri + si
                for ri in range(np.size(self.sim.model.regions))
                if ri not in affected_region_idx
                for si in rebuilding_sectors_idx
            ],
            dtype="int64",
        )

    @cached_property
    def rebuilding_industries_idx_all(self) -> npt.NDArray:
        """
        Gets the index of all rebuilding industries involved in the event.

        Returns
        -------
        npt.NDArray
            Array of indices representing all rebuilding industries for the event.
        """
        rebuilding_sectors_idx = self.rebuilding_sectors_idx
        return np.array(
            [
                np.size(self.sim.model.sectors) * ri + si
                for ri in range(np.size(self.sim.model.regions))
                for si in rebuilding_sectors_idx
            ],
            dtype="int64",
        )
