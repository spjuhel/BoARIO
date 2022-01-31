import json
import pathlib
import pymrio as pym
import numpy as np
from nptyping import NDArray
from typing import Any, Union
from ario3.utils.logger import init_logger
from pymrio.core.mriosystem import IOSystem

__all__ = ['MrioSystem']

logger = init_logger(__name__,pathlib.Path.cwd()/"run.log")

VALUE_ADDED_NAMES = ['VA', 'Value Added', 'value added',
                        'factor inputs', 'factor_inputs', 'Factors Inputs',
                        'Satellite Accounts', 'satellite accounts', 'satellite_accounts',
                     'satellite']

VA_idx = np.array(['Taxes less subsidies on products purchased: Total',
       'Other net taxes on production',
       "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
       "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
       "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
       'Operating surplus: Consumption of fixed capital',
       'Operating surplus: Rents on land',
       'Operating surplus: Royalties on resources',
       'Operating surplus: Remaining net operating surplus'], dtype=object)

def lexico_reindex(mrio: pym.IOSystem) -> pym.IOSystem:

    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.index), axis=0)
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.columns), axis=1)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0) #type: ignore
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)

    return mrio

class MrioSystem(object):
    def __init__(self,
                 pym_mrio: IOSystem,
                 mrio_params: dict,
                 simulation_params: dict,
                 results_storage: pathlib.Path
                 ) -> None:

        """TODO describe function

        :param pym_mrio:
        :type pym_mrio: IOSystem
        :param mrio_params:
        :type mrio_params: dict
        :param simulation_params:
        :type simulation_params: dict
        :param results_storage:
        :type results_storage: pathlib.Path
        :returns:

        """
        logger.debug("Initiating new MrioSystem instance")
        super().__init__()

        self.results_storage = results_storage
        self.regions = np.array(sorted(list(pym_mrio.get_regions()))) #type: ignore
        self.n_regions = len(pym_mrio.get_regions()) #type: ignore
        self.sectors = np.array(sorted(list(pym_mrio.get_sectors()))) #type: ignore
        self.n_sectors = len(pym_mrio.get_sectors()) #type: ignore
        try:
            self.fd_cat = np.array(sorted(list(pym_mrio.get_Y_categories()))) #type: ignore
            self.n_fd_cat = len(pym_mrio.get_Y_categories()) #type: ignore
        except KeyError:
            self.n_fd_cat = 1
            self.fd_cat = np.array(["Final demand"])
        except IndexError:
            self.n_fd_cat= 1
            self.fd_cat = np.array(["Final demand"])
        self.monetary_unit = mrio_params['monetary_unit']
        logger.info("Monetary unit is: %s", self.monetary_unit)
        self.psi = simulation_params['psi_param']
        self.model_timestep = simulation_params['model_time_step']
        self.timestep_dividing_factor = simulation_params['timestep_dividing_factor']
        self.rebuild_tau = simulation_params['rebuild_tau']
        self.overprod_max = simulation_params['alpha_max']
        self.overprod_tau = simulation_params['alpha_tau']
        self.overprod_base = simulation_params['alpha_base']
        self.detailled = False

        pym_mrio = lexico_reindex(pym_mrio)
        self.matrix_id = np.eye(self.n_sectors)
        self.matrix_I_sum = np.tile(self.matrix_id, self.n_regions)
        inv = mrio_params['inventories_dict']
        inventories = [ np.inf if inv[k]=='inf' else inv[k] for k in sorted(inv.keys())]
        self.inv_duration = np.array(inventories)
        self.restoration_tau = np.full(self.n_sectors, simulation_params['inventory_restoration_time'])

        self.Z_0 = pym_mrio.Z.to_numpy()
        self.Z_C = (self.matrix_I_sum @ self.Z_0)
        with np.errstate(divide='ignore',invalid='ignore'):
            self.Z_distrib = (np.divide(self.Z_0,(np.tile(self.Z_C, (self.n_regions, 1)))))
        self.Z_distrib = np.nan_to_num(self.Z_distrib)

        self.Z_0 = (pym_mrio.Z.to_numpy() / self.timestep_dividing_factor)
        self.Y_0 = (pym_mrio.Y.to_numpy() / self.timestep_dividing_factor)
        self.X_0 = (pym_mrio.x.T.to_numpy().flatten() / self.timestep_dividing_factor) #type: ignore
        #self.classic_demand_evolution = (pym_mrio.x.T.to_numpy().flatten() / self.timestep_dividing_factor) #type: ignore

        exts_names, exts = pym_mrio.get_extensions(), pym_mrio.get_extensions(True)
        tmp_chk = False
        for name in exts_names:
            ext = next(exts)
            if name in VALUE_ADDED_NAMES:
                value_added = ext.F #type: ignore
                tmp_chk = True
        if not tmp_chk:
            raise NotImplementedError('Value added table not found in given MRIO, contact the dev !')
        value_added = value_added.loc[VA_idx]
        value_added = value_added.reindex(sorted(value_added.index), axis=0) #type: ignore
        value_added = value_added.reindex(sorted(value_added.columns), axis=1)
        if value_added.ndim > 1:
            self.gdp_df = value_added.sum(axis=0).groupby('region').sum()
            self.VA_0 = (value_added.sum(axis=0).to_numpy())
        else:
            self.gdp_df = value_added.groupby('region').sum()
            self.VA_0 = (value_added.to_numpy())
        self.tech_mat = ((self.matrix_I_sum @ pym_mrio.A).to_numpy())
        self.overprod = np.full((self.n_regions * self.n_sectors), self.overprod_base, dtype=np.float64)
        with np.errstate(divide='ignore',invalid='ignore'):
            self.matrix_stock = ((np.tile(self.X_0, (self.n_sectors, 1)) * self.tech_mat) * self.inv_duration[:,np.newaxis])
        self.matrix_stock = np.nan_to_num(self.matrix_stock,nan=np.inf, posinf=np.inf)
        self.matrix_stock_0 = self.matrix_stock.copy()
        self.matrix_orders = self.Z_0.copy()
        self.production = self.X_0.copy()
        self.production_cap = self.X_0.copy()
        self.intmd_demand = self.Z_0.copy()
        self.final_demand = self.Y_0.copy()
        self.rebuilding_demand = None
        self.prod_max_toward_rebuilding = None
        self.kapital_lost = np.zeros(self.production.shape)
        if value_added.ndim > 1:
            self.gdp_share_sector = (self.VA_0 / value_added.sum(axis=0).groupby('region').transform('sum').to_numpy())
        else:
            self.gdp_share_sector = (self.VA_0 / value_added.groupby('region').transform('sum').to_numpy())
        self.gdp_share_sector = self.gdp_share_sector.flatten()
        kratio = mrio_params['capital_ratio_dict']
        kratio_ordered = [kratio[k] for k in sorted(kratio.keys())]
        self.kstock_ratio_to_VA = np.tile(np.array(kratio_ordered),self.n_regions)

        self.matrix_share_thresh = self.Z_C > np.tile(self.X_0, (self.n_sectors, 1)) * 0.00001 # [n_sectors, n_regions*n_sectors]
        results_storage = results_storage.absolute()
        self.results_storage = results_storage

        self.production_evolution = np.memmap(results_storage/"iotable_XVA_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.production_cap_evolution = np.memmap(results_storage/"iotable_X_max_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.classic_demand_evolution = np.memmap(results_storage/"classic_demand_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.rebuild_demand_evolution = np.memmap(results_storage/"rebuild_demand_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.overproduction_evolution = np.memmap(results_storage/"overprodvector_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.final_demand_unmet_evolution = np.memmap(results_storage/"final_demand_unmet_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.rebuild_production_evolution = np.memmap(results_storage/"rebuild_prod_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        if simulation_params['register_stocks']:
            self.stocks_evolution = np.memmap(results_storage/"stocks_record", dtype='float64', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors, self.n_sectors*self.n_regions))
        self.limiting_stocks_evolution = np.memmap(results_storage/"limiting_stocks_record", dtype='bool', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors, self.n_sectors*self.n_regions))

    def calc_production_cap(self):
        """TODO describe function

        :returns:

        """
        self.production_cap = self.X_0.copy()
        productivity_loss = np.zeros(shape=self.kapital_lost.shape)
        k_stock = (self.VA_0 * self.kstock_ratio_to_VA)
        np.divide(self.kapital_lost, k_stock, out=productivity_loss, where=k_stock!=0)
        if (productivity_loss > 0.).any():
            self.production_cap = self.production_cap * (1 - productivity_loss)
        if (self.overprod > 1.0).any():
            self.production_cap *= self.overprod
        assert not (self.production_cap < 0).any()

    def calc_prod_reqby_demand(self):
        """TODO describe function

        :returns:

        """
        dmg_demand_restorable = self.aggregate_rebuild_demand() #* self.rebuild_tau
        prod_reqby_demand = self.matrix_orders.sum(axis=1) + self.final_demand.sum(axis=1)
        if dmg_demand_restorable is not None:
            prod_reqby_demand += dmg_demand_restorable.sum(axis=1)
        assert not (prod_reqby_demand < 0).any()
        return prod_reqby_demand

    def calc_production(self):
        """TODO describe function

        :returns:

        """
        prod_reqby_demand = self.calc_prod_reqby_demand()
        production_opt = np.fmin(prod_reqby_demand, self.production_cap)
        supply_constraint = (np.tile(production_opt, (self.n_sectors, 1)) * self.tech_mat) * self.psi
        np.multiply(supply_constraint, np.tile(np.nan_to_num(self.inv_duration, posinf=0.)[:,np.newaxis],(1,self.n_regions*self.n_sectors)), out=supply_constraint)
        if (stock_constraint := (self.matrix_stock < supply_constraint) * self.matrix_share_thresh).any():
            production_ratio_stock = np.ones(shape=self.matrix_stock.shape)
            np.divide(self.matrix_stock, supply_constraint, out=production_ratio_stock, where=(self.matrix_share_thresh * (supply_constraint!=0)))
            production_ratio_stock[production_ratio_stock > 1] = 1
            if (production_ratio_stock < 1).any():
                production_max = np.tile(production_opt, (self.n_sectors, 1)) * production_ratio_stock
                assert not (np.min(production_max,axis=0) < 0).any()
                self.production = np.min(production_max, axis=0)
            else:
                assert not (production_opt < 0).any()
                self.production = production_opt
        else:
            assert not (production_opt < 0).any()
            self.production = production_opt
        return stock_constraint

    def calc_orders(self, stocks_constraints):
        """TODO describe function

        :param stocks_constraints:
        :type stocks_constraints:
        :returns:

        """
        prod_reqby_demand = self.calc_prod_reqby_demand()
        production_opt = np.fmin(prod_reqby_demand, self.production_cap)
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
        matrix_stock_gap = np.expand_dims(self.model_timestep/self.restoration_tau, axis=1) * matrix_stock_gap
        # Speed up restocking ?
        matrix_stock_gap[stocks_constraints] *=2
        matrix_stock_gap += (np.tile(self.production, (self.n_sectors, 1)) * self.tech_mat)
        assert not ((np.tile(matrix_stock_gap, (self.n_regions, 1)) * self.Z_distrib) < 0).any()
        self.matrix_orders = (np.tile(matrix_stock_gap, (self.n_regions, 1)) * self.Z_distrib)

    def aggregate_rebuild_demand(self):
        """TODO describe function

        :returns:

        """
        if self.rebuilding_demand is None:
            return None
        else:
            assert self.rebuilding_demand.ndim == 3
            return self.rebuilding_demand.sum(axis=0)

    def calc_rebuilding_production(self):
        if self.rebuilding_demand is None:
            return np.full(self.production.shape, 0.0), self.production
        elif self.prod_max_toward_rebuilding is not None:
            #non_rebuild_demand = np.concatenate([self.matrix_orders, self.final_demand], axis=1)
            rebuild_demand = self.rebuilding_demand.sum(axis=2)
            rebuild_production = self.production[np.newaxis,:] * self.prod_max_toward_rebuilding
            rebuild_production = np.minimum(rebuild_production, rebuild_demand)
            #surplus_from_rebuild = (rebuild_production - rebuild_demand)
            #surplus_from_rebuild[surplus_from_rebuild<0.0] = 0.0
            #surplus_from_rebuild = surplus_from_rebuild.sum(axis=0)
            scarcity = np.full(rebuild_production.shape,0.0)
            scarcity[rebuild_demand > 0.] = (rebuild_demand[rebuild_demand > 0.] - rebuild_production[rebuild_demand > 0.]) / rebuild_demand[rebuild_demand > 0.]
            scarcity[scarcity < 0] = 0.0

            #scarcity[np.isinf(scarcity)] = 0
            prod_max_toward_rebuild_chg = ((1. - self.prod_max_toward_rebuilding) * scarcity * (self.model_timestep / self.rebuild_tau) + (0. - self.prod_max_toward_rebuilding) * (scarcity == 0) * (self.model_timestep / self.rebuild_tau))
            assert not prod_max_toward_rebuild_chg[(prod_max_toward_rebuild_chg < -1) | (prod_max_toward_rebuild_chg > 1)].any()
            self.prod_max_toward_rebuilding += prod_max_toward_rebuild_chg
            self.prod_max_toward_rebuilding[self.prod_max_toward_rebuilding < 0] = 0
            self.prod_max_toward_rebuilding = self.prod_max_toward_rebuilding.round(10)
            assert not self.prod_max_toward_rebuilding[(self.prod_max_toward_rebuilding < 0) | (self.prod_max_toward_rebuilding > 1)].any()
            non_rebuild_production = self.production - rebuild_production.sum(axis=0)

            assert np.allclose(rebuild_production.sum(axis=0) + non_rebuild_production, self.production)
            return rebuild_production, non_rebuild_production
        else:
            raise ValueError("Attempt to compute prod_max_toward_rebuilding_chg while prod_max_toward_rebuilding is None")

    def distribute_production(self,
                              t: int,
                              scheme='proportional'):
        if scheme != 'proportional':
            raise ValueError("Scheme %s not implemented"% scheme)

        rebuild_production, non_rebuild_production = self.calc_rebuilding_production()
        self.write_rebuild_prod(t,rebuild_production.sum(axis=0)) #type: ignore
        # 'Usual' demand (intermediate and final)
        non_rebuild_demand = np.concatenate([self.matrix_orders, self.final_demand], axis=1)
        rationning_required = (non_rebuild_production - non_rebuild_demand.sum(axis=1))<(-1/self.monetary_unit)
        rationning_mask = np.tile(rationning_required[:,np.newaxis],(1,(self.n_regions*self.n_sectors)+(self.n_regions*self.n_fd_cat)))
        demand_share = np.full(non_rebuild_demand.shape,0.0)
        tot_dem = np.expand_dims(np.sum(non_rebuild_demand, axis=1, where=rationning_mask),1)
        np.divide(non_rebuild_demand, tot_dem, where=(tot_dem!=0), out=demand_share)
        #with np.errstate(divide='ignore',invalid='ignore'):
        #    demand_share = np.divide(non_rebuild_demand, np.expand_dims(np.sum(non_rebuild_demand, axis=1),1))
        #shape = demand_share.shape
        #demand_share = demand_share.ravel()
        #demand_share[np.isnan(demand_share)]=0
        #assert (not demand_share.isnull().any()), "NaNs in demand share (distribution module)"
        #demand_share = demand_share.reshape(shape)
        distributed_non_rebuild_production = non_rebuild_demand
        np.multiply(demand_share, np.expand_dims(non_rebuild_production,1), out=distributed_non_rebuild_production, where=rationning_mask)

        # Rebuilding

        if self.rebuilding_demand is not None:
            rebuild_demand_share = np.full(self.rebuilding_demand.shape,0.0)
            tot_rebuilding_demand = np.broadcast_to(self.rebuilding_demand.sum(axis=2)[:,:,np.newaxis],self.rebuilding_demand.shape)
            rebuild_demand_share[tot_rebuilding_demand!=0] = np.divide(self.rebuilding_demand[tot_rebuilding_demand!=0], tot_rebuilding_demand[tot_rebuilding_demand!=0])
            distributed_rebuild_production = np.multiply(rebuild_demand_share, np.expand_dims(rebuild_production,2))
            assert not ((self.rebuilding_demand - distributed_rebuild_production).round(10) < 0).any()
            self.rebuilding_demand -= distributed_rebuild_production#.reshape(self.rebuilding_demand.shape)
            self.__update_kapital_lost()

        intmd_distribution = distributed_non_rebuild_production[:,:self.n_sectors * self.n_regions]
        stock_use = np.tile(self.production, (self.n_sectors,1)) * self.tech_mat
        assert not (stock_use < 0).any()
        stock_add = self.matrix_I_sum @ intmd_distribution
        assert not (stock_add < 0).any()
        if not np.allclose(stock_add, stock_use):
            assert not (self.matrix_stock < 0).any()
            self.matrix_stock = self.matrix_stock - stock_use + stock_add
            if (self.matrix_stock < 0).any():
                self.matrix_stock.dump(self.results_storage/"matrix_stock_dump.pkl")
                assert False, "Errors in the stocks, matrix has been dumped in the results dir"

        final_demand_not_met = self.final_demand - distributed_non_rebuild_production[:,self.n_sectors*self.n_regions:]#(self.n_sectors*self.n_regions + self.n_fd_cat*self.n_regions)]
        final_demand_not_met = final_demand_not_met.sum(axis=1)
        # avoid -0.0 (just in case)
        final_demand_not_met[final_demand_not_met==0.] = 0.

        self.write_final_demand_unmet(t, final_demand_not_met)

    def update_kapital_lost(self,
                        ):
        self.__update_kapital_lost()

    def __update_kapital_lost(self,
                        ):
        self.kapital_lost = self.aggregate_rebuild_demand().sum(axis=0)


    def calc_overproduction(self):
        prod_reqby_demand = self.calc_prod_reqby_demand()
        scarcity = np.full(self.production.shape, 0.0)
        scarcity[prod_reqby_demand!=0] = (prod_reqby_demand[prod_reqby_demand!=0] - self.production[prod_reqby_demand!=0]) / prod_reqby_demand[prod_reqby_demand!=0]
        scarcity[np.isnan(scarcity)] = 0
        overprod_chg = (((self.overprod_max - self.overprod) * scarcity * (self.model_timestep / self.overprod_tau)) + ((self.overprod_base - self.overprod) * (scarcity == 0) * self.model_timestep/self.overprod_tau)).flatten()
        self.overprod += overprod_chg
        self.overprod[self.overprod < 1.] = 1.

    def check_stock_increasing(self, t:int):
        tmp = np.full(self.matrix_stock.shape,0.0)
        mask = np.isfinite(self.matrix_stock_0)
        np.subtract(self.matrix_stock,self.matrix_stock_0, out=tmp, where=mask)
        check_1 = tmp > 0.0
        tmp = np.full(self.matrix_stock.shape,0.0)
        np.subtract(self.stocks_evolution[t], self.stocks_evolution[t-1], out=tmp, where=mask)
        check_2 = (tmp >= 0.0)
        return (check_1 & check_2).all()

    def check_production_eq_strict(self):
        return ((np.isclose(self.production, self.X_0)) | np.greater(self.production, self.X_0)).all()

    def check_production_eq_soft(self, t:int, period:int = 10):
        return self.check_monotony(self.production_evolution, t, period)

    def check_stocks_monotony(self, t:int, period:int = 10):
        return self.check_monotony(self.stocks_evolution, t, period)

    def check_initial_equilibrium(self):
        return (np.allclose(self.production, self.X_0) and np.allclose(self.matrix_stock, self.matrix_stock_0))

    def check_equilibrium_soft(self, t:int):
        return (self.check_stock_increasing(t) and self.check_production_eq_strict)

    def check_equilibrium_monotony(self, t:int, period:int=10):
        return self.check_production_eq_soft(t, period) and self.check_stocks_monotony(t, period)

    def check_monotony(self, x, t:int, period:int = 10):
        return np.allclose(x[t], x[t-period], atol=0.0001)

    def check_crash(self, prod_threshold : float=0.80):
        tmp = np.full(self.production.shape, 0.0)
        checker = np.full(self.production.shape, 0.0)
        mask = self.X_0 != 0
        np.subtract(self.X_0, self.production, out=tmp, where=mask)
        np.divide(checker, self.X_0, out=checker, where=mask)
        return np.where(checker >= prod_threshold)[0].size

    def reset_module(self,
                 simulation_params: dict,
                 ) -> None:
        # Reset OUTPUTS
        self.reset_record_files(simulation_params['n_timesteps'], simulation_params['register_stocks'])
        # Reset variable attributes
        self.kapital_lost = np.zeros(self.production.shape)
        self.overprod = np.full((self.n_regions * self.n_sectors), self.overprod_base, dtype=np.float64)
        with np.errstate(divide='ignore',invalid='ignore'):
            self.matrix_stock = ((np.tile(self.X_0, (self.n_sectors, 1)) * self.tech_mat) * self.inv_duration[:,np.newaxis])
        self.matrix_stock = np.nan_to_num(self.matrix_stock,nan=np.inf, posinf=np.inf)
        self.matrix_stock_0 = self.matrix_stock.copy()
        self.matrix_orders = self.Z_0.copy()
        self.production = self.X_0.copy()
        self.production_cap = self.X_0.copy()
        self.intmd_demand = self.Z_0.copy()
        self.final_demand = self.Y_0.copy()
        self.rebuilding_demand = None
        self.prod_max_toward_rebuilding = None

    def update_params(self, new_params):
        self.psi = new_params['psi_param']
        self.model_timestep = new_params['model_time_step']
        self.timestep_dividing_factor = new_params['timestep_dividing_factor']
        self.rebuild_tau = new_params['rebuild_tau']
        self.overprod_max = new_params['alpha_max']
        self.overprod_tau = new_params['alpha_tau']
        self.overprod_base = new_params['alpha_base']
        self.restoration_tau = np.full(self.n_sectors, new_params['inventory_restoration_time'])
        if self.results_storage != pathlib.Path(new_params['storage_dir']+"/"+new_params['results_storage']):
            self.results_storage = pathlib.Path(new_params['storage_dir']+"/"+new_params['results_storage'])
            self.reset_record_files(new_params['n_timesteps'], new_params['register_stocks'])

    def reset_record_files(self, n_steps:int, reg_stocks: bool):
        self.production_evolution = np.memmap(self.results_storage/"iotable_XVA_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        self.production_cap_evolution = np.memmap(self.results_storage/"iotable_X_max_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        self.classic_demand_evolution = np.memmap(self.results_storage/"classic_demand_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        self.rebuild_demand_evolution = np.memmap(self.results_storage/"rebuild_demand_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        self.overproduction_evolution = np.memmap(self.results_storage/"overprodvector_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        self.final_demand_unmet_evolution = np.memmap(self.results_storage/"final_demand_unmet_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        self.rebuild_production_evolution = np.memmap(self.results_storage/"rebuild_prod_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors*self.n_regions))
        if reg_stocks:
            self.stocks_evolution = np.memmap(self.results_storage/"stocks_record", dtype='float64', mode="w+", shape=(n_steps, self.n_sectors, self.n_sectors*self.n_regions))
            self.limiting_stocks_evolution = np.memmap(self.results_storage/"limiting_stocks_record", dtype='bool', mode="w+", shape=(n_steps, self.n_sectors, self.n_sectors*self.n_regions))


    def write_production(self, t:int):
        self.production_evolution[t] = self.production

    def write_production_max(self, t:int):
        self.production_cap_evolution[t] = self.production_cap

    def write_classic_demand(self, t:int):
         self.classic_demand_evolution[t] = self.matrix_orders.sum(axis=1) + self.final_demand.sum(axis=1)

    def write_rebuild_demand(self, t:int):
        to_write = np.full(self.n_regions*self.n_sectors,0.0)
        if (r_dem := self.aggregate_rebuild_demand()) is not None:
            self.rebuild_demand_evolution[t] = r_dem.sum(axis=1)
        else:
            self.rebuild_demand_evolution[t] = to_write

    def write_rebuild_prod(self, t:int, rebuild_prod_agg:np.ndarray):
        self.rebuild_production_evolution[t] = rebuild_prod_agg

    def write_overproduction(self, t:int):
        self.overproduction_evolution[t] = self.overprod

    def write_final_demand_unmet(self, t:int, final_demand_unmet:np.ndarray):
        self.final_demand_unmet_evolution[t] = final_demand_unmet

    def write_stocks(self, t:int):
        self.stocks_evolution[t] = self.matrix_stock

    def write_limiting_stocks(self, t:int,
                              limiting_stock:NDArray):
        self.limiting_stocks_evolution[t] = limiting_stock

    def write_index(self, index_file):
        indexes= {
            "regions":list(self.regions),
            "sectors":list(self.sectors),
            "fd_cat":list(self.fd_cat),
            "n_sectors":self.n_sectors,
            "n_regions":self.n_regions,
            "n_industries":self.n_sectors*self.n_regions
        }
        with index_file.open('w') as f:
            json.dump(indexes,f)
