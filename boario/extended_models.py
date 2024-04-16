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

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from pymrio.core.mriosystem import IOSystem

from boario import logger

from boario.model_base import ARIOBaseModel, INV_THRESHOLD

__all__ = ["ARIOPsiModel"]


class ARIOPsiModel(ARIOBaseModel):
    def __init__(
        self,
        pym_mrio: IOSystem,
        *,
        order_type="alt",
        alpha_base=1.0,
        alpha_max=1.25,
        alpha_tau=365,
        rebuild_tau=60,
        main_inv_dur=90,
        monetary_factor=10**6,
        temporal_units_by_step: int = 1,
        iotable_year_to_temporal_unit_factor: int = 365,
        infinite_inventories_sect: Optional[list] = None,
        inventory_dict: Optional[dict] = None,
        productive_capital_vector: Optional[
            pd.Series | np.ndarray | pd.DataFrame
        ] = None,
        productive_capital_to_VA_dict: Optional[dict] = None,
        psi_param=0.80,
        inventory_restoration_tau: int | Dict[str, int] = 60,
    ) -> None:
        """An ARIO model with some additional features.

        Added feature are parameter psi of production adjustment inventories constraint threshold, as well as a characteristic time of inventories resupplying.
        See :ref:`model_type`.
        """

        super().__init__(
            pym_mrio,
            order_type=order_type,
            alpha_base=alpha_base,
            alpha_max=alpha_max,
            alpha_tau=alpha_tau,
            rebuild_tau=rebuild_tau,
            main_inv_dur=main_inv_dur,
            monetary_factor=monetary_factor,
            temporal_units_by_step=temporal_units_by_step,
            iotable_year_to_temporal_unit_factor=iotable_year_to_temporal_unit_factor,
            infinite_inventories_sect=infinite_inventories_sect,
            inventory_dict=inventory_dict,
            productive_capital_vector=productive_capital_vector,
            productive_capital_to_VA_dict=productive_capital_to_VA_dict,
        )

        logger.debug("Model is an ARIOPsiModel")

        if isinstance(psi_param, str):
            self.psi = float(psi_param.replace("_", "."))
            """float: Value of the psi parameter. (see :ref:`boario-math`)."""

        elif isinstance(psi_param, float):
            self.psi = psi_param
        elif isinstance(psi_param, int):
            self.psi = float(psi_param)
        else:
            raise ValueError(
                "'psi_param' parameter is neither a str rep of a float or a float"
            )
        if self.psi > 1.0:
            raise ValueError("'psi_param' parameter must be less or equal than 1.")

        if isinstance(inventory_restoration_tau, int):
            restoration_tau = [
                (
                    (self.n_temporal_units_by_step / inventory_restoration_tau)
                    if v >= INV_THRESHOLD
                    else v
                )
                for v in self.inventories
            ]  # for sector with no inventory TODO: reflect on that.
        elif isinstance(inventory_restoration_tau, dict):
            if not set(self.sectors).issubset(inventory_restoration_tau.keys()):
                sect_set = set(self.sectors)
                keys_sect = set(inventory_restoration_tau.keys())
                raise NotImplementedError(
                    f"""The given dict for inventory restoration tau does not contains all sectors as keys. Current implementation only allows dict with ALL sectors or just one integer value:
                    Sectors not in keys: {sect_set - keys_sect}
                    """
                )

            for _, value in inventory_restoration_tau.items():
                if isinstance(value, float):
                    if not value.is_integer():
                        raise ValueError(
                            f"Invalid value: {value} ({type(value)}) in inventory_restoration_tau, values should be integer: {inventory_restoration_tau}"
                        )
                elif not isinstance(value, int):
                    raise ValueError(
                        f"Invalid value: {value} ({type(value)}) in inventory_restoration_tau, values should be integer: {inventory_restoration_tau}"
                    )

            inventory_restoration_tau = {
                key: int(val) for key, val in inventory_restoration_tau.items()
            }
            inventory_restoration_tau = dict(sorted(inventory_restoration_tau.items()))
            restoration_tau = [
                (self.n_temporal_units_by_step / v) if v >= INV_THRESHOLD else v
                for _, v in inventory_restoration_tau.items()
            ]
        else:
            raise ValueError(
                f"Invalid inventory_restoration_tau: expected dict or int got {type(inventory_restoration_tau)}"
            )
        self.restoration_tau = np.array(restoration_tau)
        r"""numpy.ndarray of int: Array of size :math:`n` setting for each inputs its characteristic restoration time :math:`\tau_{\textrm{INV}}` in ``n_temporal_units_by_step``. (see :ref:`boario-math`)."""
        #################################################################

    @property
    def inventory_constraints_opt(self) -> np.ndarray:
        return self.calc_inventory_constraints(self.production_opt)

    @property
    def inventory_constraints_act(self) -> np.ndarray:
        return self.calc_inventory_constraints(self.production)

    # This is just so that theres a docstring for this one, it actually does the same as its parent method,
    # the difference is in the inventory constraints computation.
    def calc_production(self, current_temporal_unit: int) -> np.ndarray:
        r"""Computes and updates actual production. The difference with :class:`ARIOBaseModel` is in the way
        inventory constraints are computed. See :ref:`boario-math-prod`.

        Parameters
        ----------
        current_temporal_unit : int
            current step number

        """
        return super().calc_production(current_temporal_unit)

    def calc_inventory_constraints(self, production: np.ndarray) -> np.ndarray:
        r"""Compute inventory constraints (with psi parameter, for the non psi version,
        see :meth:`~boario.model_base.ARIOBaseModel.calc_inventory_constraints`)

        Parameters
        ----------
        production : np.ndarray
            The production vector to consider.

        Returns
        -------
        np.ndarray
            For each input, for each industry, the size of the inventory required to produce at `production` level
        for the duration goal (`inv_duration`) times the psi parameter.

        """

        inventory_constraints = (
            np.tile(production, (self.n_sectors, 1)) * self.tech_mat
        ) * self.psi
        np.multiply(
            inventory_constraints,
            np.tile(
                np.nan_to_num(self.inv_duration, posinf=0.0)[:, np.newaxis],
                (1, self.n_regions * self.n_sectors),
            ),
            out=inventory_constraints,
        )
        return inventory_constraints

    def calc_matrix_stock_gap(self, matrix_stock_goal) -> np.ndarray:
        matrix_stock_gap = super().calc_matrix_stock_gap(matrix_stock_goal)
        return np.expand_dims(self.restoration_tau, axis=1) * matrix_stock_gap
