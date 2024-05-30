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

"""This module defines the core mechanisms of the model."""

from __future__ import annotations

import json
import pathlib
import typing
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymrio.core.mriosystem import IOSystem

from boario import logger, warn_once
from boario.event import (
    Event,
    EventArbitraryProd,
    EventKapitalDestroyed,
    EventKapitalRebuild,
)
from boario.utils.misc import lexico_reindex

__all__ = [
    "ARIOBaseModel",
    "INV_THRESHOLD",
    "VALUE_ADDED_NAMES",
    "VA_idx",
    "lexico_reindex",
]

INV_THRESHOLD = 0  # 20 #temporal_units

TECHNOLOGY_THRESHOLD = (
    0.00001  # Do not care about input if producing requires less than this value
)

VALUE_ADDED_NAMES = [
    "VA",
    "Value Added",
    "value added",
    "factor inputs",
    "factor_inputs",
    "Factors Inputs",
    "Satellite Accounts",
    "satellite accounts",
    "satellite_accounts",
    "satellite",
]

VA_idx = np.array(
    [
        "Taxes less subsidies on products purchased: Total",
        "Other net taxes on production",
        "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
        "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
        "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
        "Operating surplus: Consumption of fixed capital",
        "Operating surplus: Rents on land",
        "Operating surplus: Royalties on resources",
        "Operating surplus: Remaining net operating surplus",
    ],
    dtype=object,
)


class ARIOBaseModel:
    def __init__(
        self,
        pym_mrio: IOSystem,
        *,
        order_type: str = "alt",
        alpha_base: float = 1.0,
        alpha_max: float = 1.25,
        alpha_tau: int = 365,
        rebuild_tau: int = 60,
        main_inv_dur: int = 90,
        monetary_factor: int = 10**6,
        temporal_units_by_step: int = 1,
        iotable_year_to_temporal_unit_factor: int = 365,
        infinite_inventories_sect: Optional[list] = None,
        inventory_dict: Optional[dict] = None,
        productive_capital_vector: Optional[
            pd.Series | npt.NDArray | pd.DataFrame
        ] = None,
        productive_capital_to_VA_dict: Optional[dict] = None,
    ) -> None:
        r"""The core of an ARIO model.  Handles the different arrays containing the mrio tables.

        `ARIOBaseModel` wrap all the data and functions used in the core of the most basic version of the ARIO
        model (based on :cite:`2013:hallegatte` and :cite:`2020:guan`).

        An ARIOBaseModel instance based is build on the given IOSystem.
        Most default parameters are the same as in :cite:`2013:hallegatte` and
        default order mechanisms comes from :cite:`2020:guan`. By default each
        industry capital stock is 4 times its value added (:cite:`2008:hallegatte`).

        Parameters
        ----------
        pym_mrio : IOSystem
            The IOSystem to base the model on.
        order_type : str, default "alt"
            The type of orders mechanism to use. Currently, can be "alt" or
            "noalt". See :ref:`boario-math`
        alpha_base : float, default 1.0
            Base value of overproduction factor :math:`\alpha^{b}` (Default to 1.0).
        alpha_max : float, default 1.25
            Maximum factor of overproduction :math:`\alpha^{\textrm{max}}` (default should be 1.25).
        alpha_tau : int, default 365
            Characteristic time of overproduction :math:`\tau_{\alpha}` in ``n_temporal_units_by_step`` (default should be 365 days).
        rebuild_tau : int, default 60
            Rebuilding characteristic time :math:`\tau_{\textrm{REBUILD}}` (see :ref:`boario-math`). Overwritten by per event value if it exists.
        main_inv_dur : int, default 90
            The default numbers of days for inputs inventory to use if it is not defined for an input.
        monetary_factor : int, default 10**6
            Monetary unit factor (i.e. if the tables unit is 10^6 € instead of 1 €, it should be set to 10^6).
        temporal_units_by_step: int, default 1
            The number of temporal_units between each step. (Current version of the model showed inconsistencies with values other than `1`).
        iotable_year_to_temporal_unit_factor : int, default 365
            The (inverse of the) factor by which MRIO data should be multiplied in order to get "per temporal unit value", assuming IO data is yearly.
        infinite_inventories_sect: list, optional
            List of inputs (sector) considered never constraining for production.
        inventory_dict: dict, optional
            Dictionary in the form input:initial_inventory_size, (where input is a sector, and initial_inventory_size is in "temporal_unit" (defaults to a day))
        productive_capital_vector: ArrayLike, optional
            Array of capital per industry if you need to give it exogenously.
        productive_capital_to_VA_dict: dict, optional
            Dictionary in the form sector:ratio. Where ratio is used to estimate capital stock based on the value added of the sector.

        Notes
        -----

        It is recommended to use ``productive_capital_to_VA_dict`` if you have a more precise estimation of
        the ratio of (Capital Stock / Value Added) per sectors than the default 4/1 ratio.
        You may also feed in directly a ``productive_capital_vector`` if you did your estimation before-hand.
        (This is especially useful if you have events based of an exposure layer for instance)

        Regarding inventories, they default to 90 days for all inputs (ie sectors).
        You can set some inputs to be never constraining for production by listing them
        in ``infinite_inventories_sect`` or directly feed in a dictionary of the inventory
        duration for each input.

        """

        logger.debug("Initiating new ARIOBaseModel instance")
        super().__init__()
        # ############### Parameters variables #######################
        try:
            logger.info(
                "IO system metadata :\n{}\n{}\n{}\n{}".format(
                    str(pym_mrio.meta.description),
                    str(pym_mrio.meta.name),
                    str(pym_mrio.meta.system),
                    str(pym_mrio.meta.version),
                )
            )
        except AttributeError:
            logger.warning(
                "It seems the MRIOT you loaded doesn't have metadata to print."
            )

        pym_mrio = lexico_reindex(pym_mrio)
        self.main_inv_dur: int = main_inv_dur
        r"""int, default 90: The default numbers of days for inputs inventory to use if it is not defined for an input."""

        reg = pym_mrio.get_regions()
        reg = typing.cast("list[str]", reg)
        self.regions = np.array(sorted(reg))
        r"""numpy.ndarray of str : An array of the regions of the model."""

        self.n_regions = len(reg)
        r"""int : The number :math:`m` of regions."""

        sec = pym_mrio.get_sectors()
        sec = typing.cast("list[str]", sec)
        self.sectors = np.array(sorted(sec))
        r"""numpy.ndarray of str: An array of the sectors of the model."""

        self.n_sectors = len(sec)
        r"""int : The number :math:`n` of sectors."""

        self.industries = pd.MultiIndex.from_product(
            [self.regions, self.sectors], names=["region", "sector"]
        )
        r"""pandas.MultiIndex : A pandas MultiIndex of the industries (region,sector) of the model."""

        self.n_industries = len(self.industries)
        r"""int : The number :math:`m * n` of industries."""

        try:
            self.final_demand_cat = np.array(sorted(list(pym_mrio.get_Y_categories())))  # type: ignore
            r"""numpy.ndarray of str: An array of the final demand categories of the model (``["Final demand"]`` if there is only one)"""

            self.n_fd_cat = len(pym_mrio.get_Y_categories())  # type: ignore
            r"""int: The numbers of final demand categories."""

        except KeyError:
            self.n_fd_cat = 1
            self.final_demand_cat = np.array(["Final demand"])
        except IndexError:
            self.n_fd_cat = 1
            self.final_demand_cat = np.array(["Final demand"])

        if hasattr(pym_mrio, "monetary_factor"):
            logger.warning(
                f"Custom monetary factor found in the mrio pickle file, continuing with this one ({getattr(pym_mrio,'monetary_factor')})"
            )
            self.monetary_factor = getattr(pym_mrio, "monetary_factor")
        else:
            self.monetary_factor = monetary_factor
        r"""int, default 10^6: Monetary unit factor (i.e. if the tables unit is 10^6 € instead of 1 €, it should be set to 10^6)."""

        self.n_temporal_units_by_step = temporal_units_by_step
        r"""int, default 1: The number of temporal_units between each step. (Current version of the model was not tested with values other than `1`)."""

        self.iotable_year_to_temporal_unit_factor = iotable_year_to_temporal_unit_factor
        r"""int, default 365: The (inverse of the) factor by which MRIO data should be multiplied in order to get "per temporal unit value", assuming IO data is yearly."""

        if self.iotable_year_to_temporal_unit_factor != 365:
            logger.warning(
                "iotable_to_daily_step_factor is not set to 365 (days). This should probably not be the case if the IO tables you use are on a yearly basis."
            )

        self.steply_factor = (
            self.n_temporal_units_by_step / self.iotable_year_to_temporal_unit_factor
        )
        self.rebuild_tau = rebuild_tau
        r"""int: Rebuilding characteristic time :math:`\tau_{\textrm{REBUILD}}` (see :ref:`boario-math`)."""

        self.overprod_max = alpha_max
        r"""float: Maximum factor of overproduction :math:`\alpha^{\textrm{max}}` (default should be 1.25)."""

        self.overprod_tau = self.n_temporal_units_by_step / alpha_tau
        r"""float: Characteristic time of overproduction :math:`\tau_{\alpha}` in ``n_temporal_units_by_step`` (default should be 365 days)."""

        self.overprod_base = alpha_base
        r"""float: Base value of overproduction factor :math:`\alpha^{b}` (Default to 1.0)."""

        if inventory_dict is None:
            infinite_inventories_sect = (
                [] if infinite_inventories_sect is None else infinite_inventories_sect
            )
            self.inventories = [
                np.inf if sector in infinite_inventories_sect else main_inv_dur
                for sector in self.sectors
            ]
            r"""numpy.ndarray of int: Array :math:`\mathbf{s}` of size :math:`n` (sectors), setting for each input the initial number of ``n_temporal_units_by_step`` of stock for the input. (see :ref:`boario-math`)."""

        else:
            infinite_inventories_sect = (
                [] if infinite_inventories_sect is None else infinite_inventories_sect
            )
            self.inventories = [
                (
                    np.inf
                    if inventory_dict[k] in ["inf", "Inf", "Infinity", "infinity"]
                    or k in infinite_inventories_sect
                    else inventory_dict[k]
                )
                for k in sorted(inventory_dict.keys())
            ]

        logger.debug(f"inventories: {self.inventories}")
        logger.debug(f"n_temporal_units_by_step: {self.n_temporal_units_by_step}")
        self.inv_duration = np.array(self.inventories) / self.n_temporal_units_by_step

        if (self.inv_duration <= 1).any():
            logger.warning(
                f"At least one product has inventory duration lower than the numbers of temporal units in one step ({self.n_temporal_units_by_step}), model will set it to 2 by default, but you should probably check this !"
            )
            self.inv_duration[self.inv_duration <= 1] = 2
        #################################################################

        # ####### INITIAL MRIO STATE (in step temporality) ###############
        if pym_mrio.Z is None:
            raise ValueError(
                "Z attribute of given MRIO doesn't exist, this is a problem"
            )
        if pym_mrio.Y is None:
            raise ValueError(
                "Y attribute of given MRIO doesn't exist, this is a problem"
            )
        if pym_mrio.x is None:
            raise ValueError(
                "x attribute of given MRIO doesn't exist, this is a problem"
            )
        self._matrix_id = np.eye(self.n_sectors)
        self._matrix_I_sum = np.tile(self._matrix_id, self.n_regions)
        self.Z_0 = pym_mrio.Z.to_numpy()
        r"""numpy.ndarray of float: 2-dim square matrix array :math:`\ioz` of size :math:`(n \times m, n \times m)` representing the daily intermediate (transaction) matrix (see :ref:`boario-math-init`)."""

        self.Z_C = self._matrix_I_sum @ self.Z_0
        r"""numpy.ndarray of float: 2-dim matrix array :math:`\ioz^{\sectorsset}` of size :math:`(n, n \times m)` representing the intermediate (transaction) matrix aggregated by inputs (see :ref:`here <boario-math-z-agg>`)."""

        with np.errstate(divide="ignore", invalid="ignore"):
            self.Z_distrib = np.divide(
                self.Z_0, (np.tile(self.Z_C, (self.n_regions, 1)))
            )
            r"""numpy.ndarray of float: math:`\ioz` normalised by :math:`\ioz^{\sectorsset}`, i.e. representing for each input the share of the total ordered transiting from an industry to another (see :ref:`here <boario-math-z-agg>`)."""

        self.Z_distrib = np.nan_to_num(self.Z_distrib)
        self.Z_0 = pym_mrio.Z.to_numpy() * self.steply_factor

        self.Y_0 = pym_mrio.Y.to_numpy()
        self.Y_C = self._matrix_I_sum @ self.Y_0
        r"""numpy.ndarray of float: 2-dim matrix array :math:`\ioy^{\sectorsset}` of size :math:`(n, m \times \text{number of final demand categories})` representing the final demand matrix aggregated by inputs (see :ref:`here <boario-math-z-agg>`)."""

        with np.errstate(divide="ignore", invalid="ignore"):
            self.Y_distrib = np.divide(
                self.Y_0, (np.tile(self.Y_C, (self.n_regions, 1)))
            )
            r"""numpy.ndarray of float: math:`\ioy` normalised by :math:`\ioy^{\sectorsset}`, i.e. representing for each input the share of the total ordered transiting from an industry to final demand (see :ref:`here <boario-math-z-agg>`)."""

        self.Y_distrib = np.nan_to_num(self.Y_distrib)

        self.Y_0 = pym_mrio.Y.to_numpy() * self.steply_factor
        r"""numpy.ndarray of float: 2-dim array :math:`\ioy` of size :math:`(n \times m, m \times \text{number of final demand categories})` representing the daily final demand matrix."""

        tmp = np.array(pym_mrio.x.T)
        self.X_0 = tmp.flatten() * self.steply_factor
        r"""numpy.ndarray of float: Array :math:`\iox(0)` of size :math:`n \times m` representing the initial daily gross production."""

        del tmp
        value_added = pym_mrio.x.T - pym_mrio.Z.sum(axis=0)
        value_added = value_added.reindex(sorted(value_added.index), axis=0)
        value_added = value_added.reindex(sorted(value_added.columns), axis=1)
        if (value_added < 0).any().any():
            logger.warning(
                f"""Found negative values in the value added, will set to 0. Note that sectors with null value added are not impacted by capital destruction.
                industries with negative VA: {(value_added[value_added < 0].dropna(axis=1)).columns}
                """
            )

        value_added[value_added < 0] = 0.0
        self.gdp_df = value_added.T.groupby("region").sum().T
        r"""pandas.DataFrame: Dataframe of the total GDP of each region of the model"""

        self.VA_0 = value_added.to_numpy().flatten()
        r"""numpy.ndarray of float: Array :math:`\iov` of size :math:`n \times m` representing the total value added for each sectors."""

        self.tech_mat = (self._matrix_I_sum @ pym_mrio.A).to_numpy()  # type: ignore #to_numpy is not superfluous !
        r"""numpy.ndarray of float: 2-dim array :math:`\ioa` of size :math:`(n \times m, n \times m)` representing the technical coefficients matrix."""

        if productive_capital_vector is not None:
            self.k_stock = productive_capital_vector
            """numpy.ndarray of float: Array of size :math:`n \times m` representing the estimated stock of capital of each industry."""

            if isinstance(self.k_stock, pd.DataFrame):
                self.k_stock = self.k_stock.squeeze().sort_index().to_numpy()
        elif productive_capital_to_VA_dict is None:
            logger.warning("No capital to VA dictionary given, considering 4/1 ratio")
            self.kstock_ratio_to_VA = 4
            self.k_stock = self.VA_0 * self.kstock_ratio_to_VA
        else:
            kratio = productive_capital_to_VA_dict
            kratio_ordered = [kratio[k] for k in sorted(kratio.keys())]
            self.kstock_ratio_to_VA = np.tile(np.array(kratio_ordered), self.n_regions)
            self.k_stock = self.VA_0 * self.kstock_ratio_to_VA
        if value_added.ndim > 1:
            self.gdp_share_sector = (
                self.VA_0
                / value_added.sum(axis=0).groupby("region").transform("sum").to_numpy()
            )
        else:
            self.gdp_share_sector = (
                self.VA_0 / value_added.groupby("region").transform("sum").to_numpy()
            )
        self.gdp_share_sector = self.gdp_share_sector.flatten()
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the estimated share of a sector in its regional economy."""

        self.matrix_share_thresh = (
            self.Z_C > np.tile(self.X_0, (self.n_sectors, 1)) * TECHNOLOGY_THRESHOLD
        )  # [n_sectors, n_regions*n_sectors]
        r"""numpy.ndarray of float: 2-dim square matrix array of size :math:`(n , n \times m)` representing the threshold under which an input is not considered being an input (0.00001)."""
        #################################################################

        # ###### SIMULATION VARIABLES ####################################
        self.overprod = np.full(
            (self.n_regions * self.n_sectors), self.overprod_base, dtype=np.float64
        )
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the overproduction coefficients vector :math:`\mathbf{\alpha}(t)`."""

        with np.errstate(divide="ignore", invalid="ignore"):
            self.matrix_stock = (
                np.tile(self.X_0, (self.n_sectors, 1)) * self.tech_mat
            ) * self.inv_duration[:, np.newaxis]
        self.matrix_stock = np.nan_to_num(self.matrix_stock, nan=np.inf, posinf=np.inf)
        r"""numpy.ndarray of float: 2-dim square matrix array :math:`\ioinv` of size :math:`(n \times m, n \times m)` representing the stock of inputs (see :ref:`boario-math-init`)."""

        self.matrix_stock_0 = self.matrix_stock.copy()
        self.matrix_orders = self.Z_0.copy()
        r"""numpy.ndarray of float: 2-dim square matrix array :math:`\ioorders` of size :math:`(n \times m, n \times m)` representing the matrix of orders."""

        self.production = self.X_0.copy()
        r"""numpy.ndarray of float: Array :math:`\iox(t)` of size :math:`n \times m` representing the current gross production."""

        self.intmd_demand = self.Z_0.copy()
        self.final_demand = self.Y_0.copy()
        self.rebuilding_demand = None
        self.productive_capital_lost = np.zeros(self.production.shape)
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the estimated stock of capital currently destroyed for each industry."""

        self.order_type = order_type
        #################################################################

        # ################# SIMULATION TRACKING VARIABLES ################
        self.in_shortage = False
        r"""Boolean stating if at least one industry is in shortage (i.e.) if at least one of its inputs inventory is low enough to reduce production."""

        self.had_shortage = False
        r"""Boolean stating if at least one industry was in shortage at some point."""

        self.rebuild_prod = np.zeros(shape=self.X_0.shape)
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the remaining stock of rebuilding demand asked of each industry."""

        self.final_demand_not_met = np.zeros(self.Y_0.shape)
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the final demand that could not be met at current step for each industry."""

        #################################################################

        # ## Internals
        self._prod_delta_type = None

        # ### POST INIT ####
        # ## Event Class Attribute setting
        logger.debug(
            f"Setting possible regions (currently: {Event.possible_regions}) to: {self.regions}"
        )
        Event.possible_regions = pd.CategoricalIndex(
            self.regions, name="region", copy=True
        )
        logger.debug(f"Possible regions is now {Event.possible_regions}")
        Event.regions_idx = np.arange(self.n_regions)
        Event.possible_sectors = pd.CategoricalIndex(
            self.sectors, name="sector", copy=True
        )
        Event.possible_final_demand_cat = pd.CategoricalIndex(
            pym_mrio.get_Y_categories(), name="final demand", copy=True
        )
        Event.sectors_idx = np.arange(self.n_sectors)
        Event.z_shape = self.Z_0.shape
        Event.y_shape = self.Y_0.shape
        Event.x_shape = self.X_0.shape
        Event.model_monetary_factor = monetary_factor
        Event.sectors_gva_shares = self.gdp_share_sector.copy()
        Event.Z_distrib = self.Z_distrib.copy()
        Event.Y_distrib = self.Y_distrib.copy()
        # logger.warning(f"{value_added}")
        Event.gva_df = value_added.iloc[0].copy()

        meta = pym_mrio.meta.metadata
        assert isinstance(meta, dict)
        # meta = {"name":"unnamed", "description":"", "system":"unknown","version":"unknown"}
        try:
            Event.mrio_name = "_".join(
                [
                    str(meta["name"]),
                    str(meta["description"]),
                    str(meta["system"]),
                    str(meta["version"]),
                ]
            )
        except TypeError:
            Event.mrio_name = "custom - method WIP"

        # initialize those (it's not very nice, but otherwise python complains)
        self._indus_rebuild_demand_tot: npt.NDArray = np.array([])
        self._house_rebuild_demand_tot: npt.NDArray = np.array([])
        self._indus_rebuild_demand: npt.NDArray = np.array([])
        self._house_rebuild_demand: npt.NDArray = np.array([])
        self._tot_rebuild_demand: npt.NDArray = np.array([])
        self._productive_capital_lost: npt.NDArray = np.zeros(self.VA_0.shape)
        self._prod_cap_delta_productive_capital: npt.NDArray = np.zeros(
            len(self.industries)
        )
        self._prod_cap_delta_arbitrary: npt.NDArray = np.zeros(len(self.industries))
        self._prod_cap_delta_tot: npt.NDArray = np.zeros(len(self.industries))

    # # Properties

    @property
    def tot_rebuild_demand(self) -> npt.NDArray:
        r"""Returns current total rebuilding demand (as the sum of rebuilding demand addressed to each industry)"""
        tmp = []
        logger.debug("Trying to return tot_rebuilding demand")
        if np.any(self._indus_rebuild_demand_tot > 0):
            tmp.append(self._indus_rebuild_demand_tot)
        if np.any(self._house_rebuild_demand_tot > 0):
            tmp.append(self._house_rebuild_demand_tot)
        if tmp:
            ret = np.concatenate(tmp, axis=1).sum(axis=1)
            self._tot_rebuild_demand = ret
        else:
            self._tot_rebuild_demand = np.array([])
        return self._tot_rebuild_demand

    @tot_rebuild_demand.setter
    def tot_rebuild_demand(self, source: list[EventKapitalRebuild]):
        r"""Computes and updates total rebuilding demand based on a list of events.

        Compute and update rebuilding demand for the given list of events. Only events
        tagged as rebuildable are accounted for. Both `house_rebuild_demand` and
        `indus_rebuild_demand` are updated.

        Parameters
        ----------
        events : 'list[EventKapitalRebuild]'
            A list of EventKapitalRebuild objects
        """
        logger.debug(f"Trying to set tot_rebuilding demand from {source}")
        if not isinstance(source, list):
            ValueError(
                f"Setting tot_rebuild_demand can only be done with a list of events, not a {type(source)}"
            )
        self.house_rebuild_demand = source
        self.indus_rebuild_demand = source

    @property
    def house_rebuild_demand(self) -> npt.NDArray:
        r"""Returns household rebuilding demand matrix.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\ioy`, containing the sum of all currently
            rebuildable final demand stock.
        """

        return self._house_rebuild_demand

    @property
    def house_rebuild_demand_tot(self) -> npt.NDArray:
        r"""Returns total household rebuilding demand vector.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the sum of all currently
            rebuildable households demands.
        """

        return self._house_rebuild_demand_tot

    @house_rebuild_demand.setter
    def house_rebuild_demand(self, source: list[EventKapitalRebuild]):
        r"""Computes rebuild demand from household from a list of events

        Computes and sets rebuilding final households demand for the given list of events
        by summing the `rebuilding_demand_house member` of each event and multiplying it
        by the characteristic rebuilding time (inverse of its value). Only events
        tagged as rebuildable are accounted for.

        If the event has its own rebuilding characteristic time :math:`\tau_{\textrm{REBUILD}}` it is applied,
        else the model rebuilding characteristic time is used.

        Parameters
        ----------
        events : 'list[Event]'
            A list of Event objects

        """
        tmp = []
        for evnt in source:
            if evnt.rebuildable:
                if evnt.rebuild_tau is None:
                    rebuild_tau = self.rebuild_tau
                else:
                    rebuild_tau = evnt.rebuild_tau
                    warn_once(logger, "Event has a custom rebuild_tau")
                if len(evnt.rebuilding_demand_house) > 0:
                    tmp.append(
                        evnt.rebuilding_demand_house
                        * (self.n_temporal_units_by_step / rebuild_tau)
                    )
        if not tmp:
            self._house_rebuild_demand = np.array([])
            self._house_rebuild_demand_tot = np.array([])
        else:
            house_reb_dem = np.stack(tmp, axis=-1)
            tot = house_reb_dem.sum(axis=1)
            self._house_rebuild_demand = house_reb_dem
            self._house_rebuild_demand_tot = tot

    @property
    def indus_rebuild_demand(self) -> npt.NDArray:
        r"""Returns industrial rebuilding demand matrix.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\ioz`, containing the sum of all currently
            rebuildable intermediate demand stock.
        """

        return self._indus_rebuild_demand

    @property
    def indus_rebuild_demand_tot(self) -> npt.NDArray:
        r"""Returns total industrial rebuilding demand vector.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the sum of all currently
            rebuildable intermediate demands.
        """

        return self._indus_rebuild_demand_tot

    @indus_rebuild_demand.setter
    def indus_rebuild_demand(self, source: list[EventKapitalRebuild]):
        r"""Computes rebuild demand from economic sectors (i.e. not households)

        Computes and sets rebuilding 'intermediate demand' for the given list of events
        by summing the `rebuildind_demand_indus` of each event. Only events
        tagged as rebuildable are accounted for.

        Parameters
        ----------
        events : 'list[Event]'
            A list of Event objects

        """

        tmp = []
        for evnt in source:
            if evnt.rebuildable:
                if evnt.rebuild_tau is None:
                    rebuild_tau = self.rebuild_tau
                else:
                    rebuild_tau = evnt.rebuild_tau
                    warn_once(logger, "Event has a custom rebuild_tau")
                tmp.append(
                    evnt.rebuilding_demand_indus
                    * (self.n_temporal_units_by_step / rebuild_tau)
                )

        if not tmp:
            self._indus_rebuild_demand = np.array([])
            self._indus_rebuild_demand_tot = np.array([])
        else:
            indus_reb_dem = np.stack(tmp, axis=-1)
            self._indus_rebuild_demand = indus_reb_dem
            self._indus_rebuild_demand_tot = indus_reb_dem.sum(axis=1)
            # logger.debug(f"Setting indus_rebuild_demand_tot to {indus_reb_dem}")

    @property
    def productive_capital_lost(self) -> npt.NDArray:
        r"""Returns current stock of destroyed capital

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the "stock"
        of capital currently destroyed for each industry.
        """

        return self._productive_capital_lost

    @productive_capital_lost.setter
    def productive_capital_lost(
        self, source: list[EventKapitalDestroyed] | npt.NDArray
    ) -> None:
        r"""Computes current capital lost and update production delta accordingly.

        Computes and sets the current stock of capital lost by each industry of
        the model due to the given list of events. Also update the production
        capacity lost accordingly by computing the ratio of capital lost of
        capital stock.

        Parameters
        ----------
        source : list[EventKapitalDestroyed] | npt.NDArray
            Either a list of events to consider for the destruction
        of capital or directly a vector of destroyed capital for each industry.

        """

        logger.debug("Updating productive_capital lost from list of events")
        if isinstance(source, list):
            if source:
                self._productive_capital_lost = np.add.reduce(
                    np.array(
                        [
                            e.regional_sectoral_productive_capital_destroyed
                            for e in source
                        ]
                    )
                )
            else:
                self._productive_capital_lost = np.zeros(self.VA_0.shape)
        elif isinstance(source, np.ndarray):
            self._productive_capital_lost = source

        productivity_loss_from_capital_destroyed = np.zeros(
            shape=self._productive_capital_lost.shape
        )
        np.divide(
            self._productive_capital_lost,
            self.k_stock,
            out=productivity_loss_from_capital_destroyed,
            where=self.k_stock != 0,
        )
        logger.debug("Updating production delta from productive_capital loss")
        self._prod_cap_delta_productive_capital = (
            productivity_loss_from_capital_destroyed
        )
        if (self._prod_cap_delta_productive_capital > 0.0).any():
            if self._prod_delta_type is None:
                self._prod_delta_type = "from_productive_capital"
            elif self._prod_delta_type == "from_arbitrary":
                self._prod_delta_type = "mixed_from_productive_capital_from_arbitrary"

    @property
    def prod_cap_delta_arbitrary(self) -> npt.NDArray:
        r"""Return the possible "arbitrary" production capacity lost vector if
        it was set.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, stating the amount of production
        capacity lost arbitrarily (ie exogenous).
        """
        return self._prod_cap_delta_arbitrary

    @prod_cap_delta_arbitrary.setter
    def prod_cap_delta_arbitrary(self, source: list[EventArbitraryProd] | npt.NDArray):
        """Computes and sets the loss of production capacity from "arbitrary" sources.

        Parameters
        ----------
        source : list[Event] | npt.NDArray
            Either a list of Event objects with arbitrary production losses
            set, or directly a vector of production capacity loss.

        """

        if isinstance(source, list):
            event_arb = np.array(
                [
                    ev.prod_cap_delta_arbitrary
                    for ev in source
                    if len(ev.prod_cap_delta_arbitrary) > 0
                ]
            )
            if event_arb.size == 0:
                self._prod_cap_delta_arbitrary = np.zeros(shape=self.X_0.shape)
            else:
                self._prod_cap_delta_arbitrary = np.maximum.reduce(event_arb)
        else:
            self._prod_capt_delta_arbitrary = source
        assert self._prod_cap_delta_arbitrary is not None
        if (self._prod_cap_delta_arbitrary > 0.0).any():
            if self._prod_delta_type is None:
                self._prod_delta_type = "from_arbitrary"
            elif self._prod_delta_type == "from_productive_capital":
                self._prod_delta_type = "mixed_from_productive_capital_from_arbitrary"

    @property
    def prod_cap_delta_productive_capital(self) -> npt.NDArray:
        r"""Return the possible production capacity lost due to capital destroyed vector if
        it was set.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, stating the amount of production
        capacity lost due to capital destroyed.
        """

        return self._prod_cap_delta_productive_capital

    @property
    def prod_cap_delta_tot(self) -> npt.NDArray:
        r"""Computes and return total current production delta.

        Returns
        -------
        npt.NDArray
            The total production delta (ie share of production capacity lost)
            for each industry.

        """

        logger.debug("Trying to retrieve current production delta")
        tmp = []
        if self._prod_delta_type is None:
            raise AttributeError("Production delta doesn't appear to be set yet.")
        elif self._prod_delta_type == "from_productive_capital":
            logger.debug(
                "Production delta is only set from productive_capital destruction"
            )
            tmp.append(self._prod_cap_delta_productive_capital)
        elif self._prod_delta_type == "from_arbitrary":
            logger.debug("Production delta is only set from arbitrary delta")
            tmp.append(self._prod_cap_delta_arbitrary)
        elif self._prod_delta_type == "mixed_from_productive_capital_from_arbitrary":
            logger.debug(
                "Production delta is a mixed form of productive_capital destruction and arbitrary delta"
            )
            tmp.append(self._prod_cap_delta_productive_capital)
            tmp.append(self._prod_cap_delta_arbitrary)
        else:
            raise NotImplementedError(
                f"Production delta type {self._prod_delta_type} not recognised"
            )
        # tmp.append(np.ones(shape=self.X_0.shape))
        # logger.debug("tmp: {}".format(tmp))
        self._prod_cap_delta_tot = np.amax(np.stack(tmp, axis=-1), axis=1)
        assert (
            self._prod_cap_delta_tot.shape == self.X_0.shape
        ), f"expected shape {self.X_0.shape}, received {self._prod_cap_delta_tot.shape}"
        return self._prod_cap_delta_tot

    @prod_cap_delta_tot.setter
    def prod_cap_delta_tot(self, source: list[Event]):
        r"""Computes and sets the loss of production capacity from both "arbitrary" sources and
        capital destroyed.

        Parameters
        ----------
        source : list[Event]
            A list of Event objects.

        """

        logger.debug("Updating total production delta from list of events")
        if not isinstance(source, list):
            ValueError(
                f"Setting prod_cap_delta_tot can only be done with a list of events, not a {type(source)}"
            )

        self.productive_capital_lost = [
            event for event in source if isinstance(event, EventKapitalDestroyed)
        ]
        self.prod_cap_delta_arbitrary = [
            event for event in source if isinstance(event, EventArbitraryProd)
        ]

    def update_system_from_events(self, events: list[Event]) -> None:
        """Update model variables according to given list of events

        Computes and sets both the total production delta from all events, and the total rebuilding demand.

        Parameters
        ----------
        events : 'list[Event]'
            List of events (as Event objects) to consider.

        """
        logger.debug("Updating system from list of events")
        self.prod_cap_delta_tot = events
        self.tot_rebuild_demand = [
            event for event in events if isinstance(event, EventKapitalRebuild)
        ]

    @property
    def production_cap(self) -> npt.NDArray:
        r"""Compute and update production capacity.

        Compute and update production capacity from current total production delta and overproduction.

        .. math::

            x^{Cap}_{f}(t) = \alpha_{f}(t) (1 - \Delta_{f}(t)) x_{f}(t)

        Raises
        ------
        ValueError
            Raised if any industry has negative production.
        """
        production_cap = self.X_0.copy()
        if self._prod_delta_type is not None:
            prod_delta_tot = self.prod_cap_delta_tot.copy()
            if (prod_delta_tot > 0.0).any():
                production_cap = production_cap * (1 - prod_delta_tot)
        if (self.overprod > 1.0).any():
            production_cap = production_cap * self.overprod
        if (production_cap < 0).any():
            raise ValueError(
                "Production capacity was found negative for at least on industry. It may be caused by an impact being to important for a sector."
            )
        return production_cap

    @property
    def total_demand(self) -> npt.NDArray:
        r"""Computes and returns total demand as the sum of intermediate demand (orders), final demand, and possible rebuilding demand."""

        if (self.matrix_orders < 0).any():
            raise RuntimeError("Some matrix orders are negative which shouldn't happen")

        tot_dem = self.matrix_orders.sum(axis=1) + self.final_demand.sum(axis=1)
        if (tot_dem < 0).any():
            raise RuntimeError("Some total demand are negative which shouldn't happen")
        if len(self.tot_rebuild_demand) > 0:
            tot_dem += self.tot_rebuild_demand
        return tot_dem

    @property
    def production_opt(self) -> npt.NDArray:
        r"""Computes and returns "optimal production" :math:`\iox^{textrm{Opt}}`, as the per industry minimum between
        total demand and production capacity.

        """

        return np.fmin(self.total_demand, self.production_cap)

    @property
    def inventory_constraints_opt(self) -> npt.NDArray:
        r"""Computes and returns inventory constraints for "optimal production" (see :meth:`calc_inventory_constraints`)"""

        return self.calc_inventory_constraints(self.production_opt)

    @property
    def inventory_constraints_act(self) -> npt.NDArray:
        r"""Computes and returns inventory constraints for "actual production" (see :meth:`calc_inventory_constraints`)"""
        return self.calc_inventory_constraints(self.production)

    def calc_production(self, current_temporal_unit: int) -> npt.NDArray:
        r"""Computes and updates actual production. See :ref:`boario-math-prod`.

        1. Computes ``production_opt`` and ``inventory_constraints`` as :

        .. math::
           :nowrap:

                \begin{alignat*}{4}
                      \iox^{\textrm{Opt}}(t) &= (x^{\textrm{Opt}}_{f}(t))_{f \in \firmsset} &&= \left ( \min \left ( d^{\textrm{Tot}}_{f}(t), x^{\textrm{Cap}}_{f}(t) \right ) \right )_{f \in \firmsset} && \text{Optimal production}\\
                      \mathbf{\ioinv}^{\textrm{Cons}}(t) &= (\omega^{\textrm{Cons},f}_p(t))_{\substack{p \in \sectorsset\\f \in \firmsset}} &&=
                    \begin{bmatrix}
                        \tau^{1}_1 & \hdots & \tau^{p}_1 \\
                        \vdots & \ddots & \vdots\\
                        \tau^1_n & \hdots & \tau^{p}_n
                    \end{bmatrix}
                    \odot \begin{bmatrix} \iox^{\textrm{Opt}}(t)\\ \vdots\\ \iox^{\textrm{Opt}}(t) \end{bmatrix} \odot \ioa^{\sectorsset} && \text{Inventory constraints} \\
                    &&&= \begin{bmatrix}
                        \tau^{1}_1 x^{\textrm{Opt}}_{1}(t) a_{11} & \hdots & \tau^{p}_1 x^{\textrm{Opt}}_{p}(t) a_{1p}\\
                        \vdots & \ddots & \vdots\\
                        \tau^1_n x^{\textrm{Opt}}_{1}(t) a_{n1} & \hdots & \tau^{p}_n x^{\textrm{Opt}}_{p}(t) a_{np}
                    \end{bmatrix} && \\
                \end{alignat*}

        2. If stocks do not meet ``inventory_constraints`` for any inputs, then decrease production accordingly :

        .. math::
           :nowrap:

                \begin{alignat*}{4}
                    \iox^{a}(t) &= (x^{a}_{f}(t))_{f \in \firmsset} &&= \left \{ \begin{aligned}
                                                           & x^{\textrm{Opt}}_{f}(t) & \text{if $\omega_{p}^f(t) \geq \omega^{\textrm{Cons},f}_p(t)$}\\
                                                           & x^{\textrm{Opt}}_{f}(t) \cdot \min_{p \in \sectorsset} \left ( \frac{\omega^s_{p}(t)}{\omega^{\textrm{Cons,f}}_p(t)} \right ) & \text{if $\omega_{p}^f(t) < \omega^{\textrm{Cons},f}_p(t)$}
                                                           \end{aligned} \right. \quad &&
                \end{alignat*}

        Also warns in logs if such shortages happen.


        Parameters
        ----------
        current_temporal_unit : int
            current step number

        """
        # 1.
        production_opt = self.production_opt.copy()
        inventory_constraints = self.inventory_constraints_opt.copy()
        # 2.
        if (
            stock_constraint := (self.matrix_stock < inventory_constraints)
            * self.matrix_share_thresh
        ).any():
            if not self.in_shortage:
                logger.info(
                    f"At least one industry entered shortage regime. (step:{current_temporal_unit})"
                )
            self.in_shortage = True
            self.had_shortage = True
            production_ratio_stock = np.ones(shape=self.matrix_stock.shape)
            np.divide(
                self.matrix_stock,
                inventory_constraints,
                out=production_ratio_stock,
                where=(self.matrix_share_thresh * (inventory_constraints != 0)),
            )
            production_ratio_stock[production_ratio_stock > 1] = 1
            if (production_ratio_stock < 1).any():
                production_max = (
                    np.tile(production_opt, (self.n_sectors, 1))
                    * production_ratio_stock
                )
                assert not (np.min(production_max, axis=0) < 0).any()
                self.production = np.min(production_max, axis=0)
            else:
                assert not (production_opt < 0).any()
                self.production = production_opt
        else:
            if self.in_shortage:
                self.in_shortage = False
                logger.info(
                    f"All industries exited shortage regime. (step:{current_temporal_unit})"
                )
            assert not (production_opt < 0).any()
            self.production = production_opt
        return stock_constraint

    def calc_inventory_constraints(self, production: npt.NDArray) -> npt.NDArray:
        r"""Compute inventory constraints (no psi parameter, for the psi version,
        the recommended one, see :meth:`~boario.extended_models.ARIOPsiModel.calc_inventory_constraints`)

        See :meth:`calc_production` for how inventory constraints are computed.

        Parameters
        ----------
        production : npt.NDArray
            The production vector to consider.

        Returns
        -------
        npt.NDArray
            For each input, for each industry, the size of the inventory required to produce at `production` level
        for the duration goal (`inv_duration`).

        """
        inventory_constraints = np.tile(production, (self.n_sectors, 1)) * self.tech_mat
        tmp = np.tile(
            np.nan_to_num(self.inv_duration, posinf=0.0)[:, np.newaxis],
            (1, self.n_regions * self.n_sectors),
        )
        return inventory_constraints * tmp

    def distribute_production(
        self,
        rebuildable_events: list[EventKapitalRebuild],
        scheme: str = "proportional",
    ) -> list[EventKapitalRebuild]:
        r"""Production distribution module

    #. Computes rebuilding demand for each rebuildable events (applying the `rebuild_tau` characteristic time)

    #. Creates/Computes total demand matrix (Intermediate + Final + Rebuild)

    #. Assesses if total demand is greater than realized production, hence requiring rationing

    #. Distributes production proportionally to demand such that :

        .. math::
           :nowrap:

               \begin{alignat*}{4}
                   &\ioorders^{\textrm{Received}}(t) &&= \left (\frac{o_{ff'}(t)}{d^{\textrm{Tot}}_f(t)} \cdot x^a_f(t) \right )_{f,f'\in \firmsset}\\
                   &\ioy^{\textrm{Received}}(t) &&= \left ( \frac{y_{f,c}}{d^{\textrm{Tot}}_f(t)}\cdot x^a_f(t) \right )_{f\in \firmsset, c \in \catfdset}\\
                   &\Damage^{\textrm{Repaired}}(t) &&= \left ( \frac{\gamma_{f,c}}{d^{\textrm{Tot}}_f(t)} \cdot x^a_f(t) \right )_{f\in \firmsset, c \in \catfdset}\\
               \end{alignat*}

        Where :

        - :math:`\ioorders^{\textrm{Received}}(t)` is the received orders matrix,

        - :math:`\ioy^{\textrm{Received}}(t)` is the final demand received matrix,

        - :math:`\Damage^{\textrm{Repared}}(t)` is the rebuilding/repair achieved matrix,

        - :math:`d^{\textrm{Tot}}_f(t)` is the total demand to industry :math:`f`,

        - :math:`x^a_f(t)` is :math:`f`'s realized production,

        - :math:`o_{ff'}(t)` is the quantity of product ordered by industry :math:`f'` to industry :math:`f`,

        - :math:`y_{fc}(t)` is the quantity of product ordered by household :math:`c` to industry :math:`f`,

        - :math:`\gamma_{fc}(t)` is the repaired/rebuilding demand ordered to :math:`f`.

    #. Updates stocks matrix. (Only if `np.allclose(stock_add, stock_use).all()` is false)

        .. math::
           :nowrap:

               \begin{alignat*}{4}
                   &\ioinv(t+1) &&= \ioinv(t) + \left ( \mathbf{I}_{\textrm{sum}} \cdot \ioorders^{\textrm{Received}}(t) \right ) - \left ( \colvec{\iox^{\textrm{a}}(t)}{\iox^{\textrm{a}}(t)} \odot \ioa^{\sectorsset} \right )\\
               \end{alignat*}

        Where :

        - :math:`\ioinv` is the inventory matrix,
        - :math:`\mathbf{I}_{\textrm{sum}}` is a row summation matrix,
        - :math:`\ioa^{\sectorsset}` is the (input not specific to region) technical coefficients matrix.

    #. Computes final demand not met due to rationing and write it.

    #. Updates rebuilding demand for each event (by substracting distributed production)

    Parameters
    ----------
    current_temporal_unit : int
        Current temporal unit (day|week|... depending on parameters) (required to write the final demand not met)
    events : 'list[Event]'
        List of rebuildable events
    scheme : str
        Placeholder for future distribution scheme
    separate_rebuilding : bool
        Currently unused.

    Returns
    -------
    list[Event]
        The list of events to remove from current events (as they are totally rebuilt)

    Raises
    ------
    RuntimeError
        If negative values are found in places there's should not be any
    ValueError
        If an attempt to run an unimplemented distribution scheme is tried

"""

        if scheme != "proportional":
            raise ValueError(f"Scheme {scheme} is currently not implemented")

        # list_of_demands = [self.matrix_orders, self.final_demand]
        # # 1. Calc demand from rebuilding requirements (with characteristic time rebuild_tau)
        house_reb_dem_per_event = house_reb_dem_tot_per_event = (
            indus_reb_dem_per_event
        ) = indus_reb_dem_tot_per_event = None

        if rebuildable_events:
            logger.debug("There are rebuildable events")
            n_events = len(rebuildable_events)
            tot_rebuilding_demand_summed = self.tot_rebuild_demand.copy()
            # debugging assert
            if not tot_rebuilding_demand_summed.shape == self.X_0.shape:
                raise RuntimeError(
                    f"received {tot_rebuilding_demand_summed.shape}, expected {self.X_0.shape}"
                )
            indus_reb_dem_tot_per_event = self.indus_rebuild_demand_tot.copy()
            indus_reb_dem_per_event = self.indus_rebuild_demand.copy()

            # expected shape assert (debug also)
            exp_shape_indus_per_event = (
                self.n_sectors * self.n_regions,
                self.n_sectors * self.n_regions,
                n_events,
            )
            exp_shape_indus_tot_per_event = (self.n_sectors * self.n_regions, n_events)
            assert (
                indus_reb_dem_per_event.shape == exp_shape_indus_per_event
            ), "expected shape is {}, given shape is {}".format(
                exp_shape_indus_per_event, indus_reb_dem_per_event.shape
            )
            assert (
                indus_reb_dem_tot_per_event.shape == exp_shape_indus_tot_per_event
            ), "expected shape is {}, given shape is {}".format(
                exp_shape_indus_tot_per_event, indus_reb_dem_tot_per_event.shape
            )

            house_reb_dem_tot_per_event = self.house_rebuild_demand_tot.copy()
            house_reb_dem_per_event = self.house_rebuild_demand.copy()

            # expected shape assert (debug also)
            exp_shape_house = (
                self.n_sectors * self.n_regions,
                self.n_fd_cat * self.n_regions,
                n_events,
            )
            exp_shape_house_tot = (self.n_sectors * self.n_regions, n_events)
            assert house_reb_dem_per_event.shape == exp_shape_house
            assert house_reb_dem_tot_per_event.shape == exp_shape_house_tot
            h_reb = (house_reb_dem_tot_per_event > 0).any()
            ind_reb = (indus_reb_dem_tot_per_event > 0).any()
        else:
            tot_rebuilding_demand_summed = np.zeros(shape=self.X_0.shape)
            h_reb = False
            ind_reb = False

        # # 2. Concat to have total demand matrix (Intermediate + Final + Rebuild)
        tot_demand = np.concatenate(
            [
                self.matrix_orders,
                self.final_demand,
                np.expand_dims(tot_rebuilding_demand_summed, 1),
            ],
            axis=1,
        )
        # # 3. Does production meet total demand
        logger.debug(f"tot demand shape : {tot_demand.shape}")
        rationing_required = (self.production - tot_demand.sum(axis=1)) < (
            -1 / self.monetary_factor
        )
        rationning_mask = np.tile(
            rationing_required[:, np.newaxis], (1, tot_demand.shape[1])
        )
        demand_share = np.full(tot_demand.shape, 0.0)
        tot_dem_summed = np.expand_dims(
            np.sum(tot_demand, axis=1, where=rationning_mask), 1
        )
        # Get demand share
        np.divide(
            tot_demand, tot_dem_summed, where=(tot_dem_summed != 0), out=demand_share
        )
        distributed_production = tot_demand.copy()
        # 4. distribute production proportionally to demand
        np.multiply(
            demand_share,
            np.expand_dims(self.production, 1),
            out=distributed_production,
            where=rationning_mask,
        )
        intmd_distribution = distributed_production[
            :, : self.n_sectors * self.n_regions
        ]
        # Stock use is simply production times technical coefs
        stock_use = np.tile(self.production, (self.n_sectors, 1)) * self.tech_mat
        if (stock_use < 0).any():
            raise RuntimeError(
                "Stock use contains negative values, this should not happen"
            )
        # 5. Restock is the production from each supplier, summed.
        stock_add = self._matrix_I_sum @ intmd_distribution
        if (stock_add < 0).any():
            raise RuntimeError(
                "stock_add (restocking) contains negative values, this should not happen"
            )
        if not np.allclose(stock_add, stock_use):
            self.matrix_stock = self.matrix_stock - stock_use + stock_add
            if (self.matrix_stock < 0).any():
                raise RuntimeError(
                    "matrix_stock contains negative values, this should not happen"
                )

        # 6. Compute final demand not met due to rationing
        final_demand_not_met = (
            self.final_demand
            - distributed_production[
                :,
                self.n_sectors
                * self.n_regions : (
                    self.n_sectors * self.n_regions + self.n_fd_cat * self.n_regions
                ),
            ]
        )
        final_demand_not_met = final_demand_not_met.sum(axis=1)
        # avoid -0.0 (just in case)
        final_demand_not_met[final_demand_not_met == 0.0] = 0.0
        self.final_demand_not_met = final_demand_not_met.copy()

        # 7. Compute production delivered to rebuilding
        logger.debug(f"distributed prod shape : {distributed_production.shape}")
        rebuild_prod = (
            distributed_production[
                :, (self.n_sectors * self.n_regions + self.n_fd_cat * self.n_regions) :
            ]
            .copy()
            .flatten()
        )
        self.rebuild_prod = rebuild_prod.copy()

        logger.debug(
            f"tot_rebuilding_demand_summed: {pd.Series(tot_rebuilding_demand_summed, index=self.industries)}"
        )

        if h_reb and ind_reb:
            logger.debug(
                "There are both household rebuilding demand and industry rebuilding demand"
            )
            logger.debug(
                f"indus_reb_dem_tot_per_event: {pd.DataFrame(indus_reb_dem_tot_per_event, index=self.industries)}"
            )
            logger.debug(
                f"house_reb_dem_tot_per_event: {pd.DataFrame(house_reb_dem_tot_per_event, index=self.industries)}"
            )
            logger.debug(
                f"dem_tot_per_event: {pd.DataFrame(house_reb_dem_tot_per_event+indus_reb_dem_tot_per_event, index=self.industries)}"  # type: ignore
            )

            tot_reb_dem_divider = np.tile(
                tot_rebuilding_demand_summed[:, np.newaxis],
                (1, indus_reb_dem_tot_per_event.shape[1]),
            )
            indus_shares = np.divide(
                indus_reb_dem_tot_per_event,
                tot_reb_dem_divider,
                out=np.zeros_like(indus_reb_dem_tot_per_event),
                where=tot_reb_dem_divider != 0,
            )
            house_shares = np.divide(
                house_reb_dem_tot_per_event,
                tot_reb_dem_divider,
                out=np.zeros_like(house_reb_dem_tot_per_event),
                where=tot_reb_dem_divider != 0,
            )
            # np.around(house_shares, 6, out=house_shares)
            # np.around(indus_shares, 6, out=indus_shares)
            logger.debug(
                f"indus_shares: {pd.DataFrame(indus_shares, index=self.industries)}"
            )
            logger.debug(
                f"house_shares: {pd.DataFrame(house_shares, index=self.industries)}"
            )

            #             tot_reb = pd.Series(tot_rebuilding_demand_summed, index=self.industries)
            #             indus_reb_dem_tot = pd.DataFrame(indus_reb_dem_tot_per_event, index=self.industries)[0]
            #             house_reb_dem_tot = pd.DataFrame(house_reb_dem_tot_per_event, index=self.industries)[0]
            #             np.around(indus_shares,10,indus_shares)
            #             np.around(house_shares,10,house_shares)
            #             np.around(tmp,10,tmp)
            #             indus_df = pd.DataFrame(indus_shares,index=self.industries)[0]
            #             house_df = pd.DataFrame(house_shares,index=self.industries)[0]
            #             tmp_df=pd.DataFrame(tmp, index=self.industries)[0]

            #             debug_df = pd.concat([tot_reb, indus_reb_dem_tot, house_reb_dem_tot, tot_reb-(house_reb_dem_tot+indus_reb_dem_tot), indus_df, house_df, tmp_df], axis=1)
            #             #logger.info(debug_df[debug_df.iloc[:,6]!=0].to_string())
            #             debug_str = f"""
            # {debug_df[debug_df.iloc[:,6]!=0].to_string()}
            # """
            assert np.allclose(
                (indus_shares + house_shares)[tot_rebuilding_demand_summed != 0], 1.0
            )

        elif h_reb:
            tot_reb_dem_divider = np.tile(
                tot_rebuilding_demand_summed[:, np.newaxis],
                (1, indus_reb_dem_tot_per_event.shape[1]),
            )
            house_shares = np.divide(
                house_reb_dem_tot_per_event,
                tot_reb_dem_divider,
                out=np.zeros_like(house_reb_dem_tot_per_event),
                where=tot_reb_dem_divider != 0,
            )
            indus_shares = np.zeros(indus_reb_dem_tot_per_event.shape)
        elif ind_reb:
            tot_reb_dem_divider = np.tile(
                tot_rebuilding_demand_summed[:, np.newaxis],
                (1, indus_reb_dem_tot_per_event.shape[1]),
            )
            indus_shares = np.divide(
                indus_reb_dem_tot_per_event,
                tot_reb_dem_divider,
                out=np.zeros_like(indus_reb_dem_tot_per_event),
                where=tot_reb_dem_divider != 0,
            )
            house_shares = np.zeros(house_reb_dem_tot_per_event.shape)

        else:
            return []

        logger.debug(f"rebuild_prod: {rebuild_prod}")
        logger.debug(f"indus_shares: {indus_shares}")

        indus_rebuild_prod = rebuild_prod[:, np.newaxis] * indus_shares  # type:ignore
        house_rebuild_prod = rebuild_prod[:, np.newaxis] * house_shares  # type:ignore

        if ind_reb:
            logger.debug(
                f"indus_rebuild_prod: {pd.DataFrame(indus_rebuild_prod, index=self.industries)}"
            )
        if h_reb:
            logger.debug(
                f"house_rebuild_prod: {pd.DataFrame(house_rebuild_prod, index=self.industries)}"
            )
            # logger.debug(f"tot_rebuilding_demand_summed: {pd.DataFrame(tot_rebuilding_demand_summed, index=self.industries)}")
        # logger.debug(f"house_rebuild_prod: {pd.Series(house_rebuild_prod.flatten(), index=self.industries)}")

        indus_rebuild_prod_distributed = np.zeros(shape=indus_reb_dem_per_event.shape)
        house_rebuild_prod_distributed = np.zeros(shape=house_reb_dem_per_event.shape)
        if ind_reb:
            # 1. We normalize rebuilding demand by total rebuilding demand (i.e. we get for each client asking, the share of the total demand)
            # This is done by broadcasting total demand and then dividing. (Perhaps this is not efficient ?)
            indus_rebuilding_demand_shares = np.zeros(
                shape=indus_reb_dem_per_event.shape
            )
            indus_rebuilding_demand_broad = np.broadcast_to(
                indus_reb_dem_tot_per_event[:, np.newaxis],
                indus_reb_dem_per_event.shape,
            )
            np.divide(
                indus_reb_dem_per_event,
                indus_rebuilding_demand_broad,
                where=(indus_rebuilding_demand_broad != 0),
                out=indus_rebuilding_demand_shares,
            )

            # 2. Then we multiply those shares by the total production (each client get production proportional to its demand relative to total demand)
            indus_rebuild_prod_broad = np.broadcast_to(
                indus_rebuild_prod[:, np.newaxis], indus_reb_dem_per_event.shape
            )
            np.multiply(
                indus_rebuilding_demand_shares,
                indus_rebuild_prod_broad,
                out=indus_rebuild_prod_distributed,
            )

        if h_reb:
            house_rebuilding_demand_shares = np.zeros(
                shape=house_reb_dem_per_event.shape
            )
            house_rebuilding_demand_broad = np.broadcast_to(
                house_reb_dem_tot_per_event[:, np.newaxis],
                house_reb_dem_per_event.shape,
            )
            np.divide(
                house_reb_dem_per_event,
                house_rebuilding_demand_broad,
                where=(house_rebuilding_demand_broad != 0),
                out=house_rebuilding_demand_shares,
            )
            house_rebuild_prod_broad = np.broadcast_to(
                house_rebuild_prod[:, np.newaxis], house_reb_dem_per_event.shape
            )
            np.multiply(
                house_rebuilding_demand_shares,
                house_rebuild_prod_broad,
                out=house_rebuild_prod_distributed,
            )

        # update rebuilding demand
        events_to_remove = []
        for e_id, evnt in enumerate(rebuildable_events):
            if len(evnt.rebuilding_demand_indus) > 0:
                evnt.rebuilding_demand_indus -= indus_rebuild_prod_distributed[
                    :, :, e_id
                ]
            if len(evnt.rebuilding_demand_house) > 0:
                evnt.rebuilding_demand_house -= house_rebuild_prod_distributed[
                    :, :, e_id
                ]
            if (evnt.rebuilding_demand_indus < (10 / self.monetary_factor)).all() and (
                evnt.rebuilding_demand_house < (10 / self.monetary_factor)
            ).all():
                events_to_remove.append(evnt)
        return events_to_remove

    def calc_matrix_stock_gap(self, matrix_stock_goal) -> npt.NDArray:
        """Computes and returns inputs stock gap matrix

        The gap is simply the difference between the goal (given as argument)
        and the current stock.

        Parameters
        ----------
        matrix_stock_goal : npt.NDArray of float
            The target inventories.

        Returns
        -------
        npt.NDArray
            The (only positive) gap between goal and current inventories.

        Raises
        ------
        RuntimeError
            If NaN are found in the result.

        """
        matrix_stock_gap = np.zeros(matrix_stock_goal.shape)
        # logger.debug("matrix_stock_goal: {}".format(matrix_stock_goal.shape))
        # logger.debug("matrix_stock: {}".format(self.matrix_stock.shape))
        # logger.debug("matrix_stock_goal_finite: {}".format(matrix_stock_goal[np.isfinite(matrix_stock_goal)].shape))
        # logger.debug("matrix_stock_finite: {}".format(self.matrix_stock[np.isfinite(self.matrix_stock)].shape))
        matrix_stock_gap[np.isfinite(matrix_stock_goal)] = (
            matrix_stock_goal[np.isfinite(matrix_stock_goal)]
            - self.matrix_stock[np.isfinite(self.matrix_stock)]
        )
        if np.isnan(matrix_stock_gap).any():
            raise RuntimeError("NaN in matrix stock gap")
        matrix_stock_gap[matrix_stock_gap < 0] = 0
        return matrix_stock_gap

    def calc_orders(self) -> None:
        r"""Computes and sets the orders (intermediate demand) for the next step.

        See :ref:`Order module <boario-math-orders>`

        Raises
        ------
        RuntimeError
            If negative orders are found, which shouldn't happen.
        """

        # total_demand = self.total_demand
        production_opt = self.production_opt.copy()
        matrix_stock_goal = np.tile(production_opt, (self.n_sectors, 1)) * self.tech_mat
        # Check this !
        with np.errstate(invalid="ignore"):
            matrix_stock_goal *= self.inv_duration[:, np.newaxis]
        if np.allclose(self.matrix_stock, matrix_stock_goal):
            matrix_stock_gap = np.zeros(matrix_stock_goal.shape)
        else:
            matrix_stock_gap = self.calc_matrix_stock_gap(matrix_stock_goal)
        matrix_stock_gap += (
            np.tile(self.production, (self.n_sectors, 1)) * self.tech_mat
        )
        if self.order_type == "alt":
            prod_ratio = np.ones(shape=self.X_0.shape)
            np.divide(
                self.production_cap, self.X_0, out=prod_ratio, where=self.X_0 != 0
            )
            Z_prod = self.Z_0 * prod_ratio[:, np.newaxis]
            Z_Cprod = np.tile(self._matrix_I_sum @ Z_prod, (self.n_regions, 1))
            out = np.zeros(shape=Z_prod.shape)
            np.divide(Z_prod, Z_Cprod, out=out, where=Z_Cprod != 0)
            tmp = np.tile(matrix_stock_gap, (self.n_regions, 1)) * out
        else:
            tmp = np.tile(matrix_stock_gap, (self.n_regions, 1)) * self.Z_distrib
        if (tmp < 0).any():
            raise RuntimeError("Negative orders computed")
        self.matrix_orders = tmp

    def calc_overproduction(self) -> None:
        r"""Computes and updates the overproduction vector.

        See :ref:`Overproduction module <boario-math-overprod>`

        """

        scarcity = np.full(self.production.shape, 0.0)
        total_demand = self.total_demand.copy()
        scarcity[total_demand != 0] = (
            total_demand[total_demand != 0] - self.production[total_demand != 0]
        ) / total_demand[total_demand != 0]
        scarcity[np.isnan(scarcity)] = 0
        overprod_chg = (
            ((self.overprod_max - self.overprod) * scarcity * self.overprod_tau)
            + (
                (self.overprod_base - self.overprod)
                * (scarcity == 0)
                * self.overprod_tau
            )
        ).flatten()
        self.overprod += overprod_chg
        self.overprod[self.overprod < 1.0] = 1.0

    def reset_module(
        self,
    ) -> None:
        """Resets the model to initial state [Deprecated]

        This method has not been checked extensively.
        """

        self.productive_capital_lost = np.zeros(self.production.shape)
        self.overprod = np.full(
            (self.n_regions * self.n_sectors), self.overprod_base, dtype=np.float64
        )
        self.matrix_stock = self.matrix_stock_0.copy()
        self.matrix_orders = self.Z_0.copy()
        self.production = self.X_0.copy()
        self.intmd_demand = self.Z_0.copy()
        self.final_demand = self.Y_0.copy()
        self.final_demand_not_met = np.zeros(self.Y_0.shape)
        self.rebuilding_demand = np.array([])
        self.in_shortage = False
        self.had_shortage = False
        self._prod_delta_type = None
        self._indus_rebuild_demand_tot = np.array([])
        self._house_rebuild_demand_tot = np.array([])
        self._indus_rebuild_demand = np.array([])
        self._house_rebuild_demand = np.array([])
        self._tot_rebuild_demand = np.array([])
        self._prod_cap_delta_productive_capital = np.array([])
        self._prod_cap_delta_arbitrary = np.array([])
        self._prod_cap_delta_tot = np.array([])

    def write_index(self, index_file: str | pathlib.Path) -> None:
        """Write the indexes of the different dataframes of the model in a json file.

        In order to easily rebuild the dataframes from the 'raw' data, this
        method create a JSON file with all columns and indexes names, namely :

        * regions names
        * sectors names
        * final demand categories
        * number of regions, sectors and industries (regions * sectors)

        Parameters
        ----------
        index_file : pathlib.Path
            Path to the file to save the indexes.
        """

        indexes = {
            "regions": list(self.regions),
            "sectors": list(self.sectors),
            "fd_cat": list(self.final_demand_cat),
            "n_sectors": self.n_sectors,
            "n_regions": self.n_regions,
            "n_industries": self.n_sectors * self.n_regions,
        }
        if isinstance(index_file, str):
            index_file = pathlib.Path(index_file)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        with index_file.open("w") as ffile:
            json.dump(indexes, ffile)

    def change_inv_duration(self, new_dur, old_dur=None) -> None:
        # replace this method by a property !
        if old_dur is None:
            old_dur = self.main_inv_dur
        old_dur = float(old_dur) / self.n_temporal_units_by_step
        new_dur = float(new_dur) / self.n_temporal_units_by_step
        logger.info(
            "Changing (main) inventories duration from {} steps to {} steps (there are {} temporal units by step so duration is {})".format(
                old_dur,
                new_dur,
                self.n_temporal_units_by_step,
                new_dur * self.n_temporal_units_by_step,
            )
        )
        self.inv_duration = np.where(
            self.inv_duration == old_dur, new_dur, self.inv_duration
        )
