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
import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from pymrio.core.mriosystem import IOSystem

from boario import logger
from boario.utils.misc import lexico_reindex, _fast_sum, _divide_arrays_ignore

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

    def __init__(
        self,
        pym_mriot: IOSystem,
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
        inventory_dict: Optional[dict[str, int]] = None,
        productive_capital_vector: Optional[
            pd.Series | npt.NDArray | pd.DataFrame
        ] = None,
        productive_capital_to_VA_dict: Optional[dict] = None,
    ) -> None:

        logger.debug("Initiating new ARIOBaseModel instance")
        # ############### Parameters variables #######################
        try:
            logger.info(
                "IO system metadata :\n{}\n{}\n{}\n{}".format(
                    str(pym_mriot.meta.description),
                    str(pym_mriot.meta.name),
                    str(pym_mriot.meta.system),
                    str(pym_mriot.meta.version),
                )
            )
        except AttributeError:
            warnings.warn(
                "It seems the MRIOT you loaded doesn't have metadata to print."
            )

        source_mriot = lexico_reindex(pym_mriot)
        self.mriot = source_mriot
        self._set_indexes(source_mriot)

        if hasattr(source_mriot, "monetary_factor"):
            warnings.warn(
                f"Custom monetary factor found in the mrio pickle file, continuing with this one ({getattr(source_mriot,'monetary_factor')})"
            )
            self.monetary_factor = getattr(source_mriot, "monetary_factor")
        else:
            self.monetary_factor = monetary_factor
        r"""int, default 10^6: Monetary unit factor (i.e. if the tables unit is 10^6 € instead of 1 €, it should be set to 10^6)."""

        self.n_temporal_units_by_step = temporal_units_by_step
        r"""int, default 1: The number of temporal_units between each step. (Current version of the model was not tested with values other than `1`)."""

        self.iotable_year_to_temporal_unit_factor = iotable_year_to_temporal_unit_factor
        r"""int, default 365: The (inverse of the) factor by which MRIO data should be multiplied in order to get "per temporal unit value", assuming IO data is yearly."""

        if self.iotable_year_to_temporal_unit_factor != 365:
            warnings.warn(
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

        self._init_input_stocks(main_inv_dur, inventory_dict, infinite_inventories_sect)
        #################################################################

        # ####### INITIAL MRIO STATE (in step temporality) ###############
        self._matrix_id = np.eye(self.n_sectors)
        self._matrix_I_sum = np.tile(self._matrix_id, self.n_regions)
        self.Z_0 = source_mriot.Z.to_numpy() # type: ignore
        r"""numpy.ndarray of float: 2-dim square matrix array :math:`\ioz` of size :math:`(n \times m, n \times m)` representing the daily intermediate (transaction) matrix (see :ref:`boario-math-init`)."""

        self.Z_C = self._matrix_I_sum @ self.Z_0
        r"""numpy.ndarray of float: 2-dim matrix array :math:`\ioz^{\sectorsset}` of size :math:`(n, n \times m)` representing the intermediate (transaction) matrix aggregated by inputs (see :ref:`here <boario-math-z-agg>`)."""

        self.Z_distrib = _divide_arrays_ignore(
            self.Z_0, (np.tile(self.Z_C, (self.n_regions, 1)))
        )
        r"""numpy.ndarray of float: math:`\ioz` normalised by :math:`\ioz^{\sectorsset}`, i.e. representing for each input the share of the total ordered transiting from an industry to another (see :ref:`here <boario-math-z-agg>`)."""

        self.Z_0 = source_mriot.Z.to_numpy() * self.steply_factor
        self.Y_0 = source_mriot.Y.to_numpy()
        self.Y_C = self._matrix_I_sum @ self.Y_0
        r"""numpy.ndarray of float: 2-dim matrix array :math:`\ioy^{\sectorsset}` of size :math:`(n, m \times \text{number of final demand categories})` representing the final demand matrix aggregated by inputs (see :ref:`here <boario-math-z-agg>`)."""

        self.Y_distrib = _divide_arrays_ignore(
            self.Y_0, (np.tile(self.Y_C, (self.n_regions, 1)))
        )
        r"""numpy.ndarray of float: math:`\ioy` normalised by :math:`\ioy^{\sectorsset}`, i.e. representing for each input the share of the total ordered transiting from an industry to final demand (see :ref:`here <boario-math-z-agg>`)."""

        self.Y_0 = source_mriot.Y.to_numpy() * self.steply_factor
        r"""numpy.ndarray of float: 2-dim array :math:`\ioy` of size :math:`(n \times m, m \times \text{number of final demand categories})` representing the daily final demand matrix."""

        self.X_0 = np.array(source_mriot.x.T).copy().flatten() * self.steply_factor
        r"""numpy.ndarray of float: Array :math:`\iox(0)` of size :math:`n \times m` representing the initial daily gross production."""

        value_added: pd.DataFrame = source_mriot.x.T - source_mriot.Z.sum(axis=0)  # type: ignore
        # value_added = value_added.reindex(sorted(value_added.index), axis=0)
        # value_added = value_added.reindex(sorted(value_added.columns), axis=1)
        if (value_added < 0).any().any():  # type: ignore
            tmp = (value_added[value_added < 0].dropna(axis=1)).columns  # type: ignore
            warnings.warn(
                f"""Found negative values in the value added, will set to 0. Note that sectors with null value added are not impacted by capital destruction.
                industries with negative VA: {tmp}
                """
            )

        value_added[value_added < 0] = 0.0
        self.gva_df = value_added.T.groupby("region").sum().T
        r"""pandas.DataFrame: Dataframe of the total GDP of each region of the model"""

        self.VA_0 = value_added.to_numpy().flatten()
        r"""numpy.ndarray of float: Array :math:`\iov` of size :math:`n \times m` representing the total value added for each sectors."""

        self.tech_mat = (self._matrix_I_sum @ source_mriot.A).to_numpy()  # type: ignore #to_numpy is not superfluous !
        r"""numpy.ndarray of float: 2-dim array :math:`\ioa` of size :math:`(n \times m, n \times m)` representing the technical coefficients matrix."""

        if productive_capital_vector is not None:
            self.productive_capital = productive_capital_vector
            """numpy.ndarray of float: Array of size :math:`n \times m` representing the estimated stock of capital of each industry."""

            if isinstance(self.productive_capital, pd.DataFrame):
                self.productive_capital = (
                    self.productive_capital.squeeze().sort_index().to_numpy()
                )
        elif productive_capital_to_VA_dict is None:
            warnings.warn("No capital to VA dictionary given, considering 4/1 ratio")
            self.capital_to_VA_ratio = 4
            self.productive_capital = self.VA_0 * self.capital_to_VA_ratio
        else:
            kratio = productive_capital_to_VA_dict
            kratio_ordered = [kratio[k] for k in sorted(kratio.keys())]
            self.capital_to_VA_ratio = np.tile(np.array(kratio_ordered), self.n_regions)
            self.productive_capital = self.VA_0 * self.capital_to_VA_ratio

        # Currently not used, and dubious definition
        # if value_added.ndim > 1:
        #     self.regional_production_share = (
        #         self.VA_0
        #         / value_added.sum(axis=0).groupby("region").transform("sum").to_numpy()
        #     )
        # else:
        #     self.regional_production_share = (
        #         self.VA_0 / value_added.groupby("region").transform("sum").to_numpy()
        #     )
        # self.regional_production_share = self.regional_production_share.flatten()
        # r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the estimated share of a sector in its regional economy."""

        self.threshold_not_input = (
            self.Z_C > np.tile(self.X_0, (self.n_sectors, 1)) * TECHNOLOGY_THRESHOLD
        )  # [n_sectors, n_regions*n_sectors]
        r"""numpy.ndarray of float: 2-dim square matrix array of size :math:`(n , n \times m)` representing the threshold under which an input is not considered being an input (0.00001)."""
        #################################################################

        # ###### SIMULATION VARIABLES ####################################
        self._entire_demand = np.zeros(
            shape=(
                self.n_regions * self.n_sectors,
                self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat,
            )
        )

        self.overprod = np.full(
            (self.n_regions * self.n_sectors), self.overprod_base, dtype=np.float64
        )
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the overproduction coefficients vector :math:`\mathbf{\alpha}(t)`."""

        with np.errstate(divide="ignore", invalid="ignore"):
            self.inputs_stock = (
                np.tile(self.X_0, (self.n_sectors, 1)) * self.tech_mat
            ) * self.inv_duration[:, np.newaxis]
        self.inputs_stock = np.nan_to_num(self.inputs_stock, nan=np.inf, posinf=np.inf)
        r"""numpy.ndarray of float: 2-dim square matrix array :math:`\ioinv` of size :math:`(n \times m, n \times m)` representing the stock of inputs (see :ref:`boario-math-init`)."""

        self.inputs_stock_0 = self.inputs_stock.copy()
        self.intermediate_demand = self.Z_0.copy()
        r"""numpy.ndarray of float: 2-dim square matrix array :math:`\ioorders` of size :math:`(n \times m, n \times m)` representing the matrix of orders."""

        self.production = self.X_0.copy()
        r"""numpy.ndarray of float: Array :math:`\iox(t)` of size :math:`n \times m` representing the current gross production."""

        self.final_demand = self.Y_0.copy()
        self.rebuilding_demand = None
        self._rebuild_demand_tot = np.zeros_like(self.X_0)
        self._prod_cap_delta_arbitrary = None # Required to init productive_capital_lost
        self.productive_capital_lost = None
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the estimated stock of capital currently destroyed for each industry."""
        self.prod_cap_delta_arbitrary = None
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing an arbitrary reduction of production capacity to each industry."""

        self.order_type = order_type
        #################################################################

        # ################# SIMULATION TRACKING VARIABLES ################
        self.in_shortage = False
        r"""Boolean stating if at least one industry is in shortage (i.e.) if at least one of its inputs inventory is low enough to reduce production."""

        self.had_shortage = False
        r"""Boolean stating if at least one industry was in shortage at some point."""

        self.rebuild_prod = None
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the remaining stock of rebuilding demand asked of each industry."""

        self.final_demand_not_met = np.zeros(self.Y_0.shape)
        r"""numpy.ndarray of float: Array of size :math:`n \times m` representing the final demand that could not be met at current step for each industry."""

        #################################################################

        # ## Internals
        self._prod_delta_type = None
        self._n_rebuilding_events = 0

    def _init_input_stocks(
        self,
        main_inv_dur: int,
        inventory_dict: dict[str, int] | None,
        infinite_inventories_sect: list[str] | None,
    ):
        self.main_inv_dur: int = main_inv_dur
        r"""int, default 90: The default numbers of days for inputs inventory to use if it is not defined for an input."""
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

        # Note that this creates inconsistencies between inventories and inv_duration
        if (self.inv_duration <= 1).any():
            warnings.warn(
                f"At least one product has inventory duration lower than the numbers of temporal units in one step ({self.n_temporal_units_by_step}), model will set it to 2 by default, but you should probably check this !"
            )
            self.inv_duration[self.inv_duration <= 1] = 2

    def _set_indexes(self, pym_mrio: IOSystem) -> None:
        reg = pym_mrio.get_regions()
        reg = typing.cast("pd.Index", reg)
        self.regions = reg.sort_values()
        r"""numpy.ndarray of str : An array of the regions of the model."""

        self.n_regions = len(reg)
        r"""int : The number :math:`m` of regions."""

        sec = pym_mrio.get_sectors()
        sec = typing.cast("pd.Index", sec)
        self.sectors = sec.sort_values()
        r"""numpy.ndarray of str: An array of the sectors of the model."""

        self.n_sectors = len(sec)
        r"""int : The number :math:`n` of sectors."""

        self.industries = pd.MultiIndex.from_product(
            [self.regions, self.sectors], names=["region", "sector"]
        )
        r"""pandas.MultiIndex : A pandas MultiIndex of the industries (region,sector) of the model."""

        self.n_industries = len(self.industries)
        r"""int : The number :math:`m * n` of industries."""

        #try:
        self.final_demand_cat = np.array(sorted(list(pym_mrio.get_Y_categories())))  # type: ignore
        r"""numpy.ndarray of str: An array of the final demand categories of the model (``["Final demand"]`` if there is only one)"""

        self.n_fd_cat = len(pym_mrio.get_Y_categories())  # type: ignore
        r"""int: The numbers of final demand categories."""

        # Not required anymore?
        # except (KeyError,IndexError):
        #     self.n_fd_cat = 1
        #     self.final_demand_cat = pd.Index(["Final demand"], name="category")

    ##### PRODUCTION CAPACITY CHANGES #####
    @property
    def prod_cap_delta_tot(self) -> npt.NDArray:
        r"""Computes and return total current production delta.

        Returns
        -------
        npt.NDArray
            The total production delta (ie share of production capacity lost)
            for each industry.

        """

        return self._prod_cap_delta_tot

    def update_prod_delta(self):
        tmp = []
        if self.prod_cap_delta_productive_capital is not None:
            tmp.append(self.prod_cap_delta_productive_capital)

        if self.prod_cap_delta_arbitrary is not None:
            tmp.append(self.prod_cap_delta_arbitrary)

        if tmp:
            self._prod_cap_delta_tot = np.amax(np.stack(tmp, axis=-1), axis=1)
        else:
            self._prod_cap_delta_tot = np.zeros_like(self.X_0)

    @property
    def productive_capital_lost(self) -> npt.NDArray | None:
        r"""Returns current stock of destroyed capital

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the "stock"
        of capital currently destroyed for each industry.
        """

        return self._productive_capital_lost

    @productive_capital_lost.setter
    def productive_capital_lost(self, value: npt.NDArray | None):
        r"""Returns current stock of destroyed capital

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the "stock"
        of capital currently destroyed for each industry.
        """

        self._productive_capital_lost = value
        if self._productive_capital_lost is not None:
            tmp = np.zeros_like(self.productive_capital)
            np.divide(
                self._productive_capital_lost,
                self.productive_capital,
                where=self.productive_capital != 0,
                out=tmp
            )
            self._prod_cap_delta_productive_capital = tmp
        else:
            self._prod_cap_delta_productive_capital = None

        self.update_prod_delta()

    @property
    def prod_cap_delta_productive_capital(self) -> npt.NDArray | None:
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
    def prod_cap_delta_arbitrary(self) -> npt.NDArray | None:
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
    def prod_cap_delta_arbitrary(self, value: npt.NDArray | None):
        if value is not None:
            if value.shape != self.X_0.shape:
                raise ValueError(f"Incorrect shape: {self.X_0.shape} expected, got {value.shape}")
        self._prod_cap_delta_arbitrary = value
        self.update_prod_delta()

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
        production_cap = production_cap * (1 - self.prod_cap_delta_tot)
        production_cap = production_cap * self.overprod
        if (production_cap < 0).any():
            raise ValueError(
                "Production capacity was found negative for at least on industry. It may be caused by an impact being to important for a sector."
            )
        return production_cap

    @property
    def entire_demand(self) -> npt.NDArray:
        r"""Returns the entire demand matrix, including intermediate demand (orders), final demand, and possible rebuilding demand."""
        return self._entire_demand

    def _chg_events_number(self):
        new_entire_demand = np.zeros(
            shape=(
                self.n_regions * self.n_sectors,
                self.n_regions * self.n_sectors
                + self.n_regions * self.n_fd_cat
                + (self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat)
                * self._n_rebuilding_events,
            )
        )
        # Only copying intermediate and final demand
        new_entire_demand[
            :, : self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
        ] = self.entire_demand[
            :, : self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
        ]
        self._entire_demand = new_entire_demand.copy()

    @property
    def entire_demand_tot(self) -> npt.NDArray:
        r"""Returns the entire demand matrix, including intermediate demand (orders), final demand, and possible rebuilding demand."""
        return self._entire_demand_tot

    @property
    def intermediate_demand(self) -> npt.NDArray:
        """Returns the entire intermediate demand matrix (orders)"""
        return self._entire_demand[:, : (self.n_regions * self.n_sectors)]

    @intermediate_demand.setter
    def intermediate_demand(self, value: npt.NDArray | pd.DataFrame | None):
        if value is None:
            value = np.zeros(
                shape=(self.n_regions * self.n_sectors, self.n_regions * self.n_sectors)
            )
        if isinstance(value, pd.DataFrame):
            value = value.values
        np.copyto(self._entire_demand[:, : (self.n_regions * self.n_sectors)], value)
        self._intermediate_demand_tot = _fast_sum(self.intermediate_demand, axis=1)
        self._entire_demand_tot = _fast_sum(self.entire_demand, axis=1)

    @property
    def intermediate_demand_tot(self) -> npt.NDArray:
        """Returns the total intermediate demand addressed to each industry"""
        return self._intermediate_demand_tot

    @property
    def final_demand(self) -> npt.NDArray:
        """Returns the entire intermediate demand matrix (orders)"""
        return self._entire_demand[
            :,
            (self.n_regions * self.n_sectors) : (
                self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
            ),
        ]

    @final_demand.setter
    def final_demand(self, value: npt.NDArray | pd.DataFrame | None):
        if value is None:
            value = np.zeros(
                shape=(self.n_regions * self.n_sectors, self.n_regions * self.n_fd_cat)
            )
        if isinstance(value, pd.DataFrame):
            value = value.values
        np.copyto(
            self._entire_demand[
                :,
                (self.n_regions * self.n_sectors) : (
                    self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
                ),
            ],
            value,
        )
        self._final_demand_tot = _fast_sum(self.final_demand, axis=1)
        self._entire_demand_tot = _fast_sum(self.entire_demand, axis=1)

    @property
    def final_demand_tot(self) -> npt.NDArray:
        """Returns the total final demand addressed to each industry"""
        return self._final_demand_tot

    @property
    def rebuild_demand(self) -> npt.NDArray | None:
        """Returns the entire intermediate demand matrix (orders)"""
        ret = self._entire_demand[
            :, (self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat) :
        ]
        if ret.size > 0:
            return ret
        else:
            return None

    @rebuild_demand.setter
    def rebuild_demand(self, value: npt.NDArray):
        if self._n_rebuilding_events < 1 and value is not None:
            raise RuntimeError("Cannot set a non-null rebuilding demand if the number of events is 0.")
        try:
            np.copyto(
                self._entire_demand[
                    :,
                    (
                        self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
                    ) :,
                ],
                value,
            )
        except ValueError as err:
            raise ValueError("Unable to assign rebuilding demand.") from err
        if self.rebuild_demand is None:
            raise RuntimeError("There was a problem assigning rebuilding demand")
        self._rebuild_demand_tot = _fast_sum(self.rebuild_demand, axis=1)
        indus = _fast_sum(
            self.rebuild_demand[
                :, : self.n_regions * self.n_sectors * self._n_rebuilding_events
            ],
            axis=1,
        )
        house = _fast_sum(
            self.rebuild_demand[
                :, self.n_regions * self.n_sectors * self._n_rebuilding_events :
            ],
            axis=1,
        )
        self._rebuild_demand_indus_tot = (
            indus if indus.size > 0 else np.zeros((1, self.n_regions * self.n_sectors))
        )
        self._rebuild_demand_house_tot = (
            house if house.size > 0 else np.zeros((1, self.n_regions * self.n_sectors))
        )
        self._entire_demand_tot = _fast_sum(self.entire_demand, axis=1)

    @property
    def rebuild_demand_tot(self) -> npt.NDArray:
        """Returns the total rebuild demand addressed to each industry"""
        return self._rebuild_demand_tot

    @property
    def rebuild_demand_house(self) -> npt.NDArray | None:
        r"""Returns household rebuilding demand matrix.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\ioy`, containing the sum of all currently
            rebuildable final demand stock.
        """

        return self._entire_demand[
            :,
            (
                self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
            )  # Intermediate and final demand
            + (
                self.n_regions * self.n_sectors * self._n_rebuilding_events
            ) :,  # indus demand * n_events
        ]

    @property
    def rebuild_demand_house_tot(self) -> npt.NDArray | None:
        r"""Returns total household rebuilding demand vector.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the sum of all currently
            rebuildable households demands.
        """

        return self._rebuild_demand_house_tot

    @property
    def rebuild_demand_indus(self) -> npt.NDArray | None:
        r"""Returns industrial rebuilding demand matrix.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\ioz`, containing the sum of all currently
            rebuildable intermediate demand stock.
        """
        return self._entire_demand[
            :,
            (
                self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat
            ) : self.n_regions * self.n_sectors + self.n_regions * self.n_fd_cat + # After intermediate and final demand
                # up to
            (
                self.n_regions * self.n_sectors  # indus demand
            )
            * self._n_rebuilding_events,  # times events
        ]

    @property
    def rebuild_demand_indus_tot(self) -> npt.NDArray | None:
        r"""Returns total industrial rebuilding demand vector.

        Returns
        -------
        npt.NDArray
            An array of same shape as math:`\iox`, containing the sum of all currently
            rebuildable intermediate demands.
        """

        return self._rebuild_demand_indus_tot

    @property
    def rebuild_prod(self) -> npt.NDArray | None:
        return self._rebuild_prod

    @property
    def rebuild_prod_indus(self) -> npt.NDArray | None:
        if self._rebuild_prod is not None:
            return self._rebuild_prod[
                :,
                : (self.n_regions * self.n_sectors * (self._n_rebuilding_events)),
            ]
        else:
            return None

    @property
    def rebuild_prod_house(self) -> npt.NDArray | None:
        if self._rebuild_prod is not None:
            return self._rebuild_prod[
                :, (self.n_regions * self.n_sectors * self._n_rebuilding_events) :
            ]
        else:
            return None

    def rebuild_prod_indus_event(self, ev_id) -> npt.NDArray | None:
        indus = self.rebuild_prod_indus
        if indus is not None:
            return indus[
                :,
                (self.n_regions * self.n_sectors)
                * ev_id : (self.n_regions * self.n_sectors)
                * (ev_id + 1),
            ]
        else:
            return None

    def rebuild_prod_house_event(self, ev_id) -> npt.NDArray | None:
        house = self.rebuild_prod_house
        if house is not None:
            return house[
                :,
                (self.n_regions * self.n_fd_cat)
                * ev_id : (self.n_regions * self.n_fd_cat)
                * (ev_id + 1),
            ]
        else:
            return None

    @property
    def rebuild_prod_tot(self) -> npt.NDArray | None:
        return self._rebuild_prod_tot

    @rebuild_prod.setter
    def rebuild_prod(self, value: npt.NDArray | None):
        self._rebuild_prod = value
        if value is not None:
            self._rebuild_prod_tot = _fast_sum(value, axis=1)
        else:
            self._rebuild_prod_tot = None

    @property
    def production_opt(self) -> npt.NDArray:
        r"""Computes and returns "optimal production" :math:`\iox^{textrm{Opt}}`, as the per industry minimum between
        total demand and production capacity.

        """

        return np.fmin(self.entire_demand_tot, self.production_cap)

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

        1. Computes ``production_opt`` and ``inventory_constraints``
        2. If stocks do not meet ``inventory_constraints`` for any inputs, then decrease production accordingly.

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
            stock_constraint := (self.inputs_stock < inventory_constraints)
            * self.threshold_not_input
        ).any():
            if not self.in_shortage:
                logger.info(
                    f"At least one industry entered shortage regime. (step:{current_temporal_unit})"
                )
            self.in_shortage = True
            self.had_shortage = True
            production_ratio_stock = np.ones(shape=self.inputs_stock.shape)
            np.divide(
                self.inputs_stock,
                inventory_constraints,
                out=production_ratio_stock,
                where=(self.threshold_not_input * (inventory_constraints != 0)),
            )
            production_ratio_stock[production_ratio_stock > 1] = 1
            production_max = (
                np.tile(production_opt, (self.n_sectors, 1))
                * production_ratio_stock
            )
            assert not (np.min(production_max, axis=0) < 0).any()
            self.production = np.min(production_max, axis=0)
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
        general_distribution_scheme: str = "proportional",
    ):
        r"""Production distribution module

        #. Computes rebuilding demand for each rebuildable events (applying the `rebuild_tau` characteristic time)

        #. Creates/Computes total demand matrix (Intermediate + Final + Rebuild)

        #. Assesses if total demand is greater than realized production, hence requiring rationing

        #. Distributes production proportionally to demand.

        #. Updates stocks matrix. (Only if `np.allclose(stock_add, stock_use).all()` is false)

        #. Computes final demand not met due to rationing and write it.

        #. Updates rebuilding demand for each event (by substracting distributed production)

        Parameters
        ----------
        rebuildable_events : 'list[Event]'
            List of rebuildable events
        scheme : str
            Placeholder for future distribution scheme

        Raises
        ------
        RuntimeError
            If negative values are found in places there's should not be any
        NotImplementedError
            If an attempt to run an unimplemented distribution scheme is tried

        """

        if general_distribution_scheme != "proportional":
            raise NotImplementedError(
                f"Scheme {general_distribution_scheme} is currently not implemented"
            )

        # list_of_demands = [self.matrix_orders, self.final_demand]
        # # 1. Calc demand from rebuilding requirements (with characteristic time rebuild_tau)

        # # 2. Concat to have total demand matrix (Intermediate + Final + Rebuild)
        # # 3. Does production meet total demand
        logger.debug(f"entire_demand_tot shape : {self.entire_demand.shape}")
        # rationing_required = (self.production - self.entire_demand_tot) < (
        #     -1 / self.monetary_factor
        # )
        # rationning_mask = np.tile(
        #     rationing_required[:, np.newaxis], (1, self.entire_demand.shape[1])
        # )
        demand_shares = np.full(self.entire_demand.shape, 0.0)
        tot_dem_summed = np.expand_dims(
            np.sum(
                self.entire_demand,
                axis=1,
                # where=rationning_mask
            ),
            1,
        )
        # Get demand share
        np.divide(
            self.entire_demand,
            tot_dem_summed,
            where=(tot_dem_summed != 0),
            out=demand_shares,
        )
        distributed_production = np.zeros_like(self.entire_demand)
        # 4. distribute production proportionally to demand
        np.multiply(
            demand_shares,
            np.expand_dims(self.production, 1),
            out=distributed_production,
            # where=rationning_mask,
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
            self.inputs_stock = self.inputs_stock - stock_use + stock_add
            if (self.inputs_stock < 0).any():
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
        final_demand_not_met = _fast_sum(final_demand_not_met, axis=1)
        # avoid -0.0 (just in case)
        final_demand_not_met[final_demand_not_met == 0.0] = 0.0
        self.final_demand_not_met = final_demand_not_met.copy()

        # 7. Compute production delivered to rebuilding
        logger.debug(f"distributed prod shape : {distributed_production.shape}")
        self.rebuild_prod = distributed_production[
            :, (self.n_sectors * self.n_regions + self.n_fd_cat * self.n_regions) :
        ].copy()
        if self.rebuild_demand is not None:
            self.rebuild_demand = self.rebuild_demand - self.rebuild_prod

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
            - self.inputs_stock[np.isfinite(self.inputs_stock)]
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
        if np.allclose(self.inputs_stock, matrix_stock_goal):
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
        self.intermediate_demand = tmp

    def calc_overproduction(self) -> None:
        r"""Computes and updates the overproduction vector.

        See :ref:`Overproduction module <boario-math-overprod>`

        """

        scarcity = np.full(self.production.shape, 0.0)
        total_demand = self.entire_demand_tot
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
        self.inputs_stock = self.inputs_stock_0.copy()
        self.production = self.X_0.copy()
        self.intermediate_demand = self.Z_0.copy()
        self.final_demand = self.Y_0.copy()
        self.final_demand_not_met = np.zeros(self.Y_0.shape)
        self.rebuilding_demand = np.array([])
        self.in_shortage = False
        self.had_shortage = False

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
