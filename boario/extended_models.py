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
import numpy as np
from boario import logger
from boario.model_base import *
from boario.event import *
from pymrio.core.mriosystem import IOSystem

__all__ = ['ARIOModelPsi']
class ARIOModelPsi(ARIOBaseModel):
    """ An ARIO3 model with some additional features

    Added feature are parameter psi of production adjustment inventories constraint threshold, as well as a characteristic time of inventories resupplying and the alternative order module from Guan2020.

    Attributes
    ----------

    psi : float
          Value of the psi parameter. (see :ref:`boario-math`).
    restoration_tau : numpy.ndarray of int
                      Array of size `n_sector` setting for each inputs its characteristic restoration time in `temporal_units_by_step`. (see :ref:`boario-math`).
    Raises
    ------
    RuntimeError
        A RuntimeError can occur when data is inconsistent (negative stocks for
        instance)
    ValueError
    NotImplementedError
    """

    def __init__(self,
                 pym_mrio: IOSystem,
                 mrio_params: dict,
                 simulation_params: dict,
                 results_storage: pathlib.Path
                 ) -> None:

        super().__init__(pym_mrio, mrio_params, simulation_params, results_storage)
        logger.debug("Model is an ARIOModelPsi")
        self.psi = float(simulation_params['psi_param'].replace("_","."))
        inv = mrio_params['inventories_dict']
        inventories = [ np.inf if inv[k]=='inf' else inv[k] for k in sorted(inv.keys())]
        restoration_tau = [(self.n_temporal_units_by_step / simulation_params['inventory_restoration_tau']) if v >= INV_THRESHOLD else v for v in inventories] # for sector with no inventory TODO: reflect on that.
        self.restoration_tau = np.array(restoration_tau)
        #################################################################

    @property
    def inventory_constraints_opt(self) -> np.ndarray:
        return self.calc_inventory_constraints(self.production_opt)

    @property
    def inventory_constraints_act(self) -> np.ndarray:
        return self.calc_inventory_constraints(self.production)

    def calc_inventory_constraints(self, production: np.ndarray) -> np.ndarray:
        inventory_constraints = (np.tile(production, (self.n_sectors, 1)) * self.tech_mat) * self.psi
        np.multiply(inventory_constraints, np.tile(np.nan_to_num(self.inv_duration, posinf=0.)[:,np.newaxis],(1,self.n_regions*self.n_sectors)), out=inventory_constraints)
        return inventory_constraints

    def calc_matrix_stock_gap(self, matrix_stock_goal) -> np.ndarray:
        matrix_stock_gap = super().calc_matrix_stock_gap(matrix_stock_goal)
        return np.expand_dims(self.restoration_tau, axis=1) * matrix_stock_gap
