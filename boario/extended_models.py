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

import json
import pathlib
from signal import pthread_sigmask
from typing import Union
import pymrio as pym
import numpy as np
from nptyping import NDArray
from boario import logger
from boario.model_base import *
from boario.event import *
from pymrio.core.mriosystem import IOSystem

__all__ = ['ARIOModelPsi']
class ARIOModelPsi(BaseARIOModel):
    """ An ARIO3 model with some additional features

    Added feature are parameter psi of production adjustment inventories constraint threshold, as well as a characteristic time of inventories resupplying and the alternative order module from Guan2020.

    Attributes
    ----------

    psi : float
          Value of the psi parameter. (see :doc:`math`).
    restoration_tau : numpy.ndarray of int
                      Array of size `n_sector` setting for each inputs its characteristic restoration time with `model_timestep` days as unit. (see :doc:`math`).
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

        self.psi = simulation_params['psi_param']
        inv = mrio_params['inventories_dict']
        inventories = [ np.inf if inv[k]=='inf' else inv[k] for k in sorted(inv.keys())]
        restoration_tau = [(self.n_days_by_step / simulation_params['inventory_restoration_time']) if v >= INV_THRESHOLD else v for v in inventories] # for sector with no inventory TODO: reflect on that.
        self.restoration_tau = np.array(restoration_tau)
        #################################################################

    def calc_production(self, current_step:int):
        r"""Compute and update actual production

        1. Compute ``production_opt`` and ``inventory_constraints`` as :

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
                    \end{bmatrix} \cdot \psi && \\
                \end{alignat*}

        2. If stocks do not meet inventory_constraints for any inputs -> Decrease production accordingly :

        .. math::
           :nowrap:

                \begin{alignat*}{4}
                    \iox^{a}(t) &= (x^{a}_{f}(t))_{f \in \firmsset} &&= \left \{ \begin{aligned}
                                                           & x^{\textrm{Opt}}_{f}(t) & \text{if $\omega_{p}^f(t) \geq \omega^{\textrm{Cons},f}_p(t)$}\\
                                                           & x^{\textrm{Opt}}_{f}(t) \cdot \min_{p \in \sectorsset} \left ( \frac{\omega^s_{p}(t)}{\omega^{\textrm{Cons,f}}_p(t)} \right ) & \text{if $\omega_{p}^f(t) < \omega^{\textrm{Cons},f}_p(t)$}
                                                           \end{aligned} \right. \quad &&
                \end{alignat*}

        Also warns in log if such shortage happens.


        Parameters
        ----------
        current_step : int
            current step number

        Returns
        -------

        A boolean NDArray `stock_constraint` of the same shape as `matrix_sock` (ie `(n_sectors,n_regions*n_sectors)`), with True for any input not meeting the inventory constraints.

        """
        #1.
        production_opt = np.fmin(self.total_demand, self.production_cap)
        inventory_constraints = (np.tile(production_opt, (self.n_sectors, 1)) * self.tech_mat) * self.psi
        np.multiply(inventory_constraints, np.tile(np.nan_to_num(self.inv_duration, posinf=0.)[:,np.newaxis],(1,self.n_regions*self.n_sectors)), out=inventory_constraints)
        #2.
        if (stock_constraint := (self.matrix_stock < inventory_constraints) * self.matrix_share_thresh).any():
            if not self.in_shortage:
                logger.info('At least one industry entered shortage regime. (step:{})'.format(current_step))
            self.in_shortage = True
            self.had_shortage = True
            production_ratio_stock = np.ones(shape=self.matrix_stock.shape)
            np.divide(self.matrix_stock, inventory_constraints, out=production_ratio_stock, where=(self.matrix_share_thresh * (inventory_constraints!=0)))
            production_ratio_stock[production_ratio_stock > 1] = 1
            if (production_ratio_stock < 1).any():
                production_max = np.tile(production_opt, (self.n_sectors, 1)) * production_ratio_stock
                assert not (np.min(production_max,axis=0) < 0).any()
                self.production = np.min(production_max, axis=0)
            else:
                assert not (production_opt < 0).any()
                self.production = production_opt
        else:
            if self.in_shortage:
                self.in_shortage = False
                logger.info('All industries exited shortage regime. (step:{})'.format(current_step))
            assert not (production_opt < 0).any()
            self.production = production_opt
        return stock_constraint

    def calc_orders(self, events:'list[Event]'):
        """TODO describe function

        :param stocks_constraints:
        :type stocks_constraints:
        :returns:

        """
        self.calc_prod_reqby_demand(events)
        production_opt = np.fmin(self.total_demand, self.production_cap)
        matrix_stock_goal = np.tile(production_opt, (self.n_sectors, 1)) * self.tech_mat
        # Check this !
        matrix_stock_gap = matrix_stock_goal * 0
        with np.errstate(invalid='ignore'):
            matrix_stock_goal *= self.inv_duration[:,np.newaxis]
        if np.allclose(self.matrix_stock, matrix_stock_goal):
            #debug_logger.info("Stock replenished ?")
            pass
        else:
            matrix_stock_gap[np.isfinite(matrix_stock_goal)] = (matrix_stock_goal[np.isfinite(matrix_stock_goal)] - self.matrix_stock[np.isfinite(self.matrix_stock)])
        assert (not np.isnan(matrix_stock_gap).any()), "NaN in matrix stock gap"
        matrix_stock_gap[matrix_stock_gap < 0] = 0
        matrix_stock_gap = np.expand_dims(self.restoration_tau, axis=1) * matrix_stock_gap
        matrix_stock_gap += (np.tile(self.production, (self.n_sectors, 1)) * self.tech_mat)
        if self.order_type == "alt":
            prod_ratio = np.divide(self.production,self.X_0, where=self.X_0!=0)
            Z_prod = self.Z_0 * prod_ratio[:, np.newaxis]
            Z_Cprod = np.tile(self._matrix_I_sum @ Z_prod,(self.n_regions,1))
            out=np.zeros(shape=Z_prod.shape)
            np.divide(Z_prod,Z_Cprod,out=out, where=Z_Cprod!=0)
            tmp = (np.tile(matrix_stock_gap, (self.n_regions, 1)) * out)
        else:
            tmp = (np.tile(matrix_stock_gap, (self.n_regions, 1)) * self.Z_distrib)
        assert not (tmp < 0).any()
        self.matrix_orders = tmp

    def update_params(self, new_params):
        super().update_params(new_params)
        self.psi = new_params['psi_param']
        inv = new_params['inventories_dict']
        inventories = [ np.inf if inv[k]=='inf' else inv[k] for k in sorted(inv.keys())]
        restoration_tau = [(self.n_days_by_step / new_params['inventory_restoration_time']) if v >= INV_THRESHOLD else v for v in inventories]
        self.restoration_tau = np.array(restoration_tau)
