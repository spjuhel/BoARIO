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
import pathlib
from typing import Dict
import numpy as np
from boario import logger
from boario.model_base import *
from boario.event import *
from pymrio.core.mriosystem import IOSystem

__all__ = ["ARIOPsiModel", "ARIOClimadaModel"]


class ARIOPsiModel(ARIOBaseModel):
    """An ARIO3 model with some additional features

    Added feature are parameter psi of production adjustment inventories constraint threshold, as well as a characteristic time of inventories resupplying and the alternative order module from Guan2020.

    Attributes
    ----------

    psi : float
          Value of the psi parameter. (see :ref:`boario-math`).
    restoration_tau : numpy.ndarray of int
                      Array of size `n_sector` setting for each inputs its characteristic restoration time in `n_temporal_units_by_step`. (see :ref:`boario-math`).
    Raises
    ------
    RuntimeError
        A RuntimeError can occur when data is inconsistent (negative stocks for
        instance)
    ValueError
    NotImplementedError
    """

    def __init__(
        self,
        pym_mrio: IOSystem,
        order_type="alt",
        alpha_base=1.0,
        alpha_max=1.25,
        alpha_tau=365,
        rebuild_tau=60,
        main_inv_dur=90,
        monetary_factor=10**6,
        psi_param=0.90,
        inventory_restoration_tau:int|Dict[str,int]=60,
        **kwargs,
    ) -> None:
        super().__init__(
            pym_mrio,
            order_type,
            alpha_base,
            alpha_max,
            alpha_tau,
            rebuild_tau,
            main_inv_dur,
            monetary_factor,
            **kwargs,
        )

        logger.debug("Model is an ARIOPsiModel")

        if isinstance(psi_param, str):
            self.psi = float(psi_param.replace("_", "."))
        elif isinstance(psi_param, float):
            self.psi = psi_param
        else:
            raise ValueError(
                "'psi_param' parameter is neither a str rep of a float or a float"
            )

        if isinstance(inventory_restoration_tau,int):
            restoration_tau = [
                (self.n_temporal_units_by_step / inventory_restoration_tau)
            if v >= INV_THRESHOLD
            else v
            for v in self.inventories
            ]  # for sector with no inventory TODO: reflect on that.
        elif isinstance(inventory_restoration_tau,dict):
            if not set(self.sectors).issubset(inventory_restoration_tau.keys()):
                raise NotImplementedError("The given dict for Inventory restoration tau does not contains all sectors as keys. Current implementation only allows dict with ALL sectors or just one integer value")

            for _ , value in inventory_restoration_tau.items():
                if not isinstance(value, int):
                    raise ValueError("Invalid value in inventory_restoration_tau, values should be integer.")

            inventory_restoration_tau = dict(sorted(inventory_restoration_tau.items()))
            restoration_tau = [
                (self.n_temporal_units_by_step / v)
            if v >= INV_THRESHOLD
            else v
                for _ , v in inventory_restoration_tau.items()
            ]
        else:
            raise ValueError(f"Invalid inventory_restoration_tau: expected dict or int got {type(inventory_restoration_tau)}")
        self.restoration_tau = np.array(restoration_tau)
        #################################################################

    @property
    def inventory_constraints_opt(self) -> np.ndarray:
        return self.calc_inventory_constraints(self.production_opt)

    @property
    def inventory_constraints_act(self) -> np.ndarray:
        return self.calc_inventory_constraints(self.production)

    def calc_inventory_constraints(self, production: np.ndarray) -> np.ndarray:
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


class ARIOClimadaModel(ARIOPsiModel):
    def __init__(
        self,
        pym_mrio: IOSystem,
        exp_stock,
        order_type="alt",
        alpha_base=1,
        alpha_max=1.25,
        alpha_tau=365,
        rebuild_tau=60,
        main_inv_dur=90,
        monetary_factor=10**6,
        psi_param=0.9,
        inventory_restoration_tau=60,
        **kwargs,
    ) -> None:
        super().__init__(
            pym_mrio,
            order_type,
            alpha_base,
            alpha_max,
            alpha_tau,
            rebuild_tau,
            main_inv_dur,
            monetary_factor,
            psi_param,
            inventory_restoration_tau,
            kapital_vector=exp_stock,
            **kwargs,
        )
