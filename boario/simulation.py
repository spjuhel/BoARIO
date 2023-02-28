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
import pathlib
from typing import Optional, Union
import math
import numpy as np
import progressbar

from boario.event import *
from boario.model_base import ARIOBaseModel
from boario.extended_models import *
from boario import logger
from boario import DEBUGFORMATTER
from pprint import pformat
from boario.utils.misc import CustomNumpyEncoder

__all__ = ["Simulation"]


class Simulation:
    """Defines a simulation object with a set of parameters and an IOSystem.

    This class wraps an :class:`~boario.model_base.ARIOBaseModel` or :class:`~boario.extended_models.ARIOPsiModel`, and create the context for
    simulations using this model. It stores execution parameters as well as events perturbing
    the model.

    Attributes
    ----------
    params_dict : dict
        Parameters to run the simulation with. If str or Path, it must lead
        to a json file containing a dictionary of the parameters.

    results_storage : pathlib.Path
        Path to store the results to.

    model : Union[ARIOBaseModel, ARIOPsiModel]
        The model to run the simulation with.

    current_temporal_unit : int
        Tracks the number of `temporal_units` elapsed since simulation start.
        This may differs from the number of `steps` if the parameter `n_temporal_units_by_step` differs from 1 temporal_unit as `current_temporal_unit` is actually `step` * `n_temporal_units_by_step`.

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

    __possible_records = [
        "production_realised",
        "production_capacity",
        "final_demand",
        "intermediate_demand",
        "rebuild_demand",
        "overproduction",
        "final_demand_unmet",
        "rebuild_prod",
        "input_stocks",
        "limiting_inputs",
        "kapital_to_recover",
    ]

    __file_save_array_specs = {
        "production_realised": (
            "float64",
            "production_evolution",
            "industries",
            np.nan,
        ),
        "production_capacity": (
            "float64",
            "production_cap_evolution",
            "industries",
            np.nan,
        ),
        "final_demand": ("float64", "final_demand_evolution", "industries", np.nan),
        "intermediate_demand": ("float64", "io_demand_evolution", "industries", np.nan),
        "rebuild_demand": ("float64", "rebuild_demand_evolution", "industries", np.nan),
        "overproduction": ("float64", "overproduction_evolution", "industries", np.nan),
        "final_demand_unmet": (
            "float64",
            "final_demand_unmet_evolution",
            "industries",
            np.nan,
        ),
        "rebuild_prod": (
            "float64",
            "rebuild_production_evolution",
            "industries",
            np.nan,
        ),
        "input_stocks": ("float64", "inputs_evolution", "stocks", np.nan),
        "limiting_inputs": ("byte", "limiting_inputs_evolution", "stocks", -1),
        "kapital_to_recover": (
            "float64",
            "regional_sectoral_kapital_destroyed_evolution",
            "industries",
            np.nan,
        ),
    }

    def __init__(
        self,
        model: Union[ARIOBaseModel, ARIOPsiModel, ARIOClimadaModel],
        n_temporal_units_to_sim=365,
        events_list: list = [],
        separate_sims: bool = False,
        save_events: bool = False,
        save_params: bool = False,
        save_index: bool = False,
        save_records: list = [],
        boario_output_dir: str | pathlib.Path = "/tmp/boario",
        results_dir_name: str = "results",
    ) -> None:
        """
        #TODO Update this one

        Initialisation of a Simulation object uses these parameters

        Parameters
        ----------
        """
        logger.info("Initializing new simulation instance")
        self._save_events = save_events
        self._save_params = save_params
        self._save_index = save_index
        if save_records != [] or save_events or save_params:
            self.output_dir = pathlib.Path(boario_output_dir)
            self.output_dir.resolve().mkdir(parents=True, exist_ok=True)
        if save_records != []:
            self.results_storage = self.output_dir.resolve() / results_dir_name
            if not self.results_storage.exists():
                self.results_storage.mkdir()

        self.model = model
        self.all_events: list[Event] = events_list
        self.currently_happening_events: list[Event] = []
        self.events_timings = set()
        self._files_to_record = []
        self.n_temporal_units_to_sim = n_temporal_units_to_sim
        Event.temporal_unit_range = self.n_temporal_units_to_sim
        self.current_temporal_unit = 0
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

        if save_records != []:
            impossible_records = set(save_records).difference(
                set(self.__possible_records)
            )
            if impossible_records != []:
                raise ValueError(
                    f"{impossible_records} are not possible records ({self.__possible_records})"
                )
            logger.info(f"Will save {save_records} records")
            self.records_storage: pathlib.Path = self.results_storage / "records"
            logger.info("Records storage is: {}".format(self.records_storage))
            self.records_storage.mkdir(parents=True, exist_ok=True)

        for rec in save_records:
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
            memmap_array = np.memmap(
                self.records_storage / filename, dtype=dtype, mode="w+", shape=shape
            )
            memmap_array.fill(fillv)
            self._files_to_record.append(attr_name)
            setattr(self, attr_name, memmap_array)

        Event.temporal_unit_range = self.n_temporal_units_to_sim
        self.params_dict = {
            "n_temporal_units_to_sim": self.n_temporal_units_to_sim,
            "output_dir": self.output_dir if hasattr(self, "output_dir") else "none",
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
        logger.info("Initialized !")
        logger.info(
            "Simulation parameters:\n{}".format(pformat(self.params_dict, compact=True))
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
            If True show a progress bar of the loop in the console.
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
        logger.info("Events : {}".format(self.all_events))

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
            for _ in bar(
                range(
                    0,
                    self.n_temporal_units_to_sim,
                    math.floor(self.model.n_temporal_units_by_step),
                )
            ):
                # assert self.current_temporal_unit == t
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
                    break
        else:
            for _ in range(
                0,
                self.n_temporal_units_to_sim,
                math.floor(self.model.n_temporal_units_by_step),
            ):
                # assert self.current_temporal_unit == t
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
                    break

        if self._files_to_record != []:
            self.flush_memmaps()

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
        if progress:
            bar.finish()  # type: ignore (bar possibly unbound but actually not possible)

    def next_step(
        self,
        check_period: int = 182,
        min_steps_check: Optional[int] = None,
        min_failing_regions: Optional[int] = None,
    ):
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
            min_failing_regions = self.model.n_regions * self.model.n_sectors // 3

        # Check if there are new events to add,
        # if some happening events can start rebuilding (if rebuildable),
        # and updates the internal model production_cap decrease and rebuild_demand
        self.check_happening_events()

        if "inputs_evolution" in self._files_to_record:
            self.write_stocks()

        if self.current_temporal_unit > 1:
            self.model.calc_overproduction()

        if "overproduction_evolution" in self._files_to_record:
            self.write_overproduction()
        if "rebuild_demand_evolution" in self._files_to_record:
            self.write_rebuild_demand()
        if "final_evolution" in self._files_to_record:
            self.write_final_demand()
        if "io_demand_evolution" in self._files_to_record:
            self.write_io_demand()

        constraints = self.model.calc_production(self.current_temporal_unit)

        if "limiting_inputs_evolution" in self._files_to_record:
            self.write_limiting_stocks(constraints)
        if "production_evolution" in self._files_to_record:
            self.write_production()
        if "production_cap_evolution" in self._files_to_record:
            self.write_production_max()
        if "regional_sectoral_kapital_destroyed_evolution" in self._files_to_record:
            self.write_kapital_lost()

        try:
            rebuildable_events = [
                ev
                for ev in self.currently_happening_events
                if isinstance(ev, EventKapitalRebuild) and ev.rebuildable
            ]
            events_to_remove = self.model.distribute_production(
                rebuildable_events, self.scheme
            )
            if "final_demand_unmet_evolution" in self._files_to_record:
                self.write_final_demand_unmet()
            if "rebuild_production_evolution" in self._files_to_record:
                self.write_rebuild_prod()
        except RuntimeError as e:
            logger.exception("This exception happened:", e)
            return 1
        events_to_remove = events_to_remove + [
            ev for ev in self.currently_happening_events if ev.over
        ]
        if events_to_remove != []:
            self.currently_happening_events = [
                e for e in self.currently_happening_events if e not in events_to_remove
            ]
            for e in events_to_remove:
                if isinstance(e, EventKapitalDestroyed):
                    logger.info(
                        "Temporal_Unit : {} ~ Event named {} that occured at {} in {} for {} damages is completely rebuilt/recovered".format(
                            self.current_temporal_unit,
                            e.name,
                            e.occurrence,
                            e.aff_regions,
                            e.total_kapital_destroyed,
                        )
                    )

        self.model.calc_orders()

        n_checks = self.current_temporal_unit // check_period
        if n_checks > self._n_checks:
            self.check_equilibrium(n_checks)
            self._n_checks += 1

        self.current_temporal_unit += self.model.n_temporal_units_by_step
        return 0

    def check_equilibrium(self, n_checks: int):
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
        self.all_events.append(ev)
        self.events_timings.add(ev.occurrence)

    def reset_sim_with_same_events(self):
        """Resets the model to its initial status (without removing the events)."""

        logger.info("Resetting model to initial status (with same events)")
        self.current_temporal_unit = 0
        self._monotony_checker = 0
        self.n_temporal_units_simulated = 0
        self.has_crashed = False
        self.model.reset_module(self.params_dict)

    def reset_sim_full(self):
        """Resets the model to its initial status and remove all events."""

        self.reset_sim_with_same_events()
        logger.info("Resetting events")
        self.all_events = []
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

    def check_happening_events(self) -> None:
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

    def write_production(self) -> None:
        self.production_evolution[self.current_temporal_unit] = self.model.production  # type: ignore

    def write_kapital_lost(self) -> None:
        self.regional_sectoral_kapital_destroyed_evolution[  # type: ignore
            self.current_temporal_unit
        ] = self.model._kapital_lost

    def write_production_max(self) -> None:
        self.production_cap_evolution[  # type: ignore
            self.current_temporal_unit
        ] = self.model.production_cap

    def write_io_demand(self) -> None:
        self.io_demand_evolution[  # type: ignore
            self.current_temporal_unit
        ] = self.model.matrix_orders.sum(axis=1)

    def write_final_demand(self) -> None:
        self.final_demand_evolution[  # type: ignore
            self.current_temporal_unit
        ] = self.model.final_demand.sum(axis=1)

    def write_rebuild_demand(self) -> None:
        to_write = np.full(self.model.n_regions * self.model.n_sectors, 0.0)
        if (r_dem := self.model.tot_rebuild_demand) is not None:
            self.rebuild_demand_evolution[self.current_temporal_unit] = r_dem  # type: ignore
        else:
            self.rebuild_demand_evolution[self.current_temporal_unit] = to_write  # type: ignore

    def write_rebuild_prod(self) -> None:
        logger.debug(
            f"self.rebuild_production_evolution shape : {self.rebuild_production_evolution.shape}, self.model.rebuild_prod shape : {self.model.rebuild_prod.shape}"  # type: ignore
        )
        self.rebuild_production_evolution[  # type: ignore
            self.current_temporal_unit
        ] = self.model.rebuild_prod

    def write_overproduction(self) -> None:
        self.overproduction_evolution[self.current_temporal_unit] = self.model.overprod  # type: ignore

    def write_final_demand_unmet(self) -> None:
        self.final_demand_unmet_evolution[  # type: ignore
            self.current_temporal_unit
        ] = self.model.final_demand_not_met

    def write_stocks(self) -> None:
        self.inputs_evolution[self.current_temporal_unit] = self.model.matrix_stock  # type: ignore

    def write_limiting_stocks(self, limiting_stock: np.ndarray) -> None:
        self.limiting_inputs_evolution[self.current_temporal_unit] = limiting_stock  # type: ignore

    def flush_memmaps(self) -> None:
        for at in self._files_to_record:
            if not hasattr(self, at):
                raise RuntimeError(f"{at} should be a member yet it isn't. This shouldn't happen.")
            else:
                getattr(self,at).flush()
