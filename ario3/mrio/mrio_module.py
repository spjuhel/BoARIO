from collections import OrderedDict
import json
from pathlib import Path
from ario3.utils.logger import debug_logger, run_logger
import pymrio as pym
import numpy as np
from nptyping import NDArray
import ario3.mrio.fast as fast
from pymrio.core.mriosystem import IOSystem

VALUE_ADDED_NAMES = ['VA', 'Value Added', 'value added',
                        'factor inputs', 'factor_inputs', 'Factors Inputs',
                        'Satellite Accounts', 'satellite accounts', 'satellite_accounts']

def lexico_reindex(mrio: pym.IOSystem) -> pym.IOSystem:

    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.index), axis=0)
    mrio.Z = mrio.Z.reindex(sorted(mrio.Z.columns), axis=1)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.index), axis=0)
    mrio.Y = mrio.Y.reindex(sorted(mrio.Y.columns), axis=1)
    mrio.x = mrio.x.reindex(sorted(mrio.x.index), axis=0) #type: ignore
    mrio.A = mrio.A.reindex(sorted(mrio.A.index), axis=0)
    mrio.A = mrio.A.reindex(sorted(mrio.A.columns), axis=1)

    return mrio

class Mrio_System(object):
    def __init__(self,
                 pym_mrio: IOSystem,
                 mrio_params: dict,
                 simulation_params: dict,
                 result_storage: Path
                 ) -> None:
        super().__init__()
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
        self.inv_duration = np.array(inventories, dtype="float32")
        self.restoration_tau = np.full(self.n_sectors, simulation_params['inventory_restoration_time'], dtype="float32")

        self.Z_0 = pym_mrio.Z.to_numpy(dtype="float32")
        self.Z_C = (self.matrix_I_sum @ self.Z_0)
        with np.errstate(divide='ignore',invalid='ignore'):
            self.Z_distrib = (np.divide(self.Z_0,(np.tile(self.Z_C, (self.n_regions, 1)))))
        self.Z_distrib = np.nan_to_num(self.Z_distrib)

        self.Z_0 = (pym_mrio.Z.to_numpy(dtype="float32") / self.timestep_dividing_factor)
        self.Y_0 = (pym_mrio.Y.to_numpy(dtype="float32") / self.timestep_dividing_factor)
        self.X_0 = (pym_mrio.x.T.to_numpy(dtype="float32").flatten() / self.timestep_dividing_factor) #type: ignore
        self.classic_demand_evolution = (pym_mrio.x.T.to_numpy(dtype="float32").flatten() / self.timestep_dividing_factor) #type: ignore

        exts_names, exts = pym_mrio.get_extensions(), pym_mrio.get_extensions(True)
        tmp_chk = False
        for name in exts_names:
            ext = next(exts)
            if name in VALUE_ADDED_NAMES:
                value_added = ext.F #type: ignore
                tmp_chk = True
        if not tmp_chk:
            raise NotImplementedError('Value added table not found in given MRIO, contact the dev !')

        value_added = value_added.reindex(sorted(value_added.index), axis=0) #type: ignore
        value_added = value_added.reindex(sorted(value_added.columns), axis=1)
        if value_added.ndim > 1:
            self.VA_0 = (value_added.sum(axis=0).to_numpy(dtype="float32"))
        else:
            self.VA_0 = (value_added.to_numpy(dtype="float32"))
        self.tech_mat = ((self.matrix_I_sum @ pym_mrio.A).to_numpy(dtype="float32"))
        self.overprod = np.full((self.n_regions * self.n_sectors), self.overprod_base, dtype="float32")
        with np.errstate(divide='ignore',invalid='ignore'):
            self.matrix_stock = ((np.tile(self.X_0, (self.n_sectors, 1)) * self.tech_mat) * self.inv_duration[:,np.newaxis])
        self.matrix_stock = np.nan_to_num(self.matrix_stock,nan=np.inf, posinf=np.inf)
        self.matrix_stock_0 = self.matrix_stock
        self.matrix_orders = self.Z_0
        self.production = self.X_0
        self.production_cap = self.X_0
        self.intmd_demand = self.Z_0
        self.final_demand = self.Y_0
        self.rebuilding_demand = None
        self.prod_max_toward_rebuilding = None
        self.kapital_lost = np.zeros(self.production.shape, dtype="float32")
        self.impacts_to_rebuild = [] # deprecated ?
        if value_added.ndim > 1:
            self.gdp_share_sector = (self.VA_0 / value_added.sum(axis=0).groupby('region').transform('sum').to_numpy())
        else:
            self.gdp_share_sector = (self.VA_0 / value_added.groupby('region').transform('sum').to_numpy())
        self.gdp_share_sector = self.gdp_share_sector.flatten()
        kratio = mrio_params['capital_ratio_dict']
        kratio_ordered = [kratio[k] for k in sorted(kratio.keys())]
        self.kstock_ratio_to_VA = np.tile(np.array(kratio_ordered, dtype="float32"),self.n_regions)

        self.matrix_share_thresh = self.Z_C > np.tile(self.classic_demand_evolution, (self.n_sectors, 1)) * 0.00001 # [n_sectors, n_regions*n_sectors]
        result_storage = result_storage.absolute()

        self.production_evolution = np.memmap(result_storage/"iotable_XVA_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.production_cap_evolution = np.memmap(result_storage/"iotable_X_max_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.classic_demand_evolution = np.memmap(result_storage/"classic_demand_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.rebuild_demand_evolution = np.memmap(result_storage/"rebuild_demand_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.overproduction_evolution = np.memmap(result_storage/"overprodvector_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.final_demand_unmet_evolution = np.memmap(result_storage/"final_demand_unmet_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.rebuild_production_evolution = np.memmap(result_storage/"rebuild_prod_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors*self.n_regions))
        self.stocks_evolution = np.memmap(result_storage/"stocks_record", dtype='float32', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors, self.n_sectors*self.n_regions))
        self.limiting_stocks_evolution = np.memmap(result_storage/"limiting_stocks_record", dtype='bool', mode="w+", shape=(simulation_params['n_timesteps'], self.n_sectors, self.n_sectors*self.n_regions))

    def calc_production_cap(self):
        self.production_cap = self.X_0
        productivity_loss = np.zeros(shape=self.kapital_lost.shape)
        k_stock = (self.VA_0 * self.kstock_ratio_to_VA)
        np.divide(self.kapital_lost, k_stock, out=productivity_loss, where=k_stock!=0)
        if (productivity_loss > 0.).any():
            self.production_cap = self.production_cap * (1 - productivity_loss)
        if (self.overprod > 1.0).any():
            self.production_cap *= self.overprod
        assert not (self.production_cap < 0).any()

    def calc_prod_reqby_demand(self):
        dmg_demand_restorable = self.aggregate_rebuild_demand() #* self.rebuild_tau
        prod_reqby_demand = self.matrix_orders.sum(axis=1) + self.final_demand.sum(axis=1)
        if dmg_demand_restorable is not None:
            prod_reqby_demand += dmg_demand_restorable.sum(axis=1)
        assert not (prod_reqby_demand < 0).any()
        return prod_reqby_demand

    def calc_production(self):
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
        matrix_stock_gap[stocks_constraints] *=2
        matrix_stock_gap += (np.tile(self.production, (self.n_sectors, 1)) * self.tech_mat)
        assert not ((np.tile(matrix_stock_gap, (self.n_regions, 1)) * self.Z_distrib) < 0).any()
        self.matrix_orders = (np.tile(matrix_stock_gap, (self.n_regions, 1)) * self.Z_distrib)

    def aggregate_rebuild_demand(self):
        if self.rebuilding_demand is None:
            return None
        else:
            assert self.rebuilding_demand.ndim == 3
            return self.rebuilding_demand.sum(axis=0)

    def calc_rebuilding_production(self):
        if self.rebuilding_demand is None:
            return np.full(self.production.shape, 0.0), self.production
        elif self.prod_max_toward_rebuilding is not None:
            rebuild_demand = self.rebuilding_demand.sum(axis=2)
            rebuild_production = self.production[np.newaxis,:] * self.prod_max_toward_rebuilding
            rebuild_production = np.minimum(rebuild_production, rebuild_demand)
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
        with np.errstate(divide='ignore',invalid='ignore'):
            demand_share = np.divide(non_rebuild_demand, np.expand_dims(np.sum(non_rebuild_demand, axis=1),1))
        shape = demand_share.shape
        demand_share = demand_share.ravel()
        demand_share[np.isnan(demand_share)]=0
        #assert (not demand_share.isnull().any()), "NaNs in demand share (distribution module)"
        demand_share = demand_share.reshape(shape)
        distributed_non_rebuild_production = np.multiply(demand_share, np.expand_dims(non_rebuild_production,1))

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
        stock_add = self.matrix_I_sum @ intmd_distribution
        if not np.allclose(stock_add, stock_use):
            self.matrix_stock = self.matrix_stock - stock_use + stock_add
            assert not (self.matrix_stock < 0).any()

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

    def check_equilibrium(self):
        return (np.allclose(self.production, self.classic_demand_evolution) and np.allclose(self.matrix_stock, self.matrix_stock_0))

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

    def calc_production_cap_fast(self):
        self.production_cap = fast.calc_production_cap(self.kapital_lost.astype(np.float32), self.VA_0, self.kstock_ratio_to_VA, self.production_cap, self.overprod, self.production)

    def distribute_production_fast(self, t:int, scheme="proportional"):
        if scheme != 'proportional':
            raise ValueError("Scheme %s not implemented"% scheme)
        rebuild_production, non_rebuild_production, self.prod_max_toward_rebuilding = self.calc_rebuilding_production_fast()
        if self.prod_max_toward_rebuilding is not None:
            self.prod_max_toward_rebuilding = self.prod_max_toward_rebuilding.round(10)

        # Non rebuild
        self.write_rebuild_prod(t,rebuild_production.sum(axis=0)) #type: ignore
        self.matrix_stock, final_demand_not_met = fast.distribute_nr_production(self.production, np.ascontiguousarray(non_rebuild_production), np.ascontiguousarray(self.matrix_orders), np.ascontiguousarray(self.final_demand), self.n_sectors, self.n_regions, np.ascontiguousarray(self.tech_mat), np.ascontiguousarray(self.matrix_stock), self.n_fd_cat)
        self.write_final_demand_unmet(t, final_demand_not_met)

        # Rebuild
        if self.rebuilding_demand is not None:
            tot_rebuilding_demand = np.broadcast_to(np.sum(self.rebuilding_demand, axis=2)[:,:,np.newaxis],self.rebuilding_demand.shape)
            self.rebuilding_demand = fast.distribute_r_production(np.ascontiguousarray(self.rebuilding_demand), np.ascontiguousarray(rebuild_production), np.ascontiguousarray(tot_rebuilding_demand))
            self.__update_kapital_lost()

    def calc_rebuilding_production_fast(self):
        if self.rebuilding_demand is None:
            return np.full(self.production.shape, 0.0), self.production_cap, self.prod_max_toward_rebuilding
        elif self.prod_max_toward_rebuilding is not None:
            return fast.calc_rebuilding_production(np.ascontiguousarray(self.rebuilding_demand), self.production, np.ascontiguousarray(self.prod_max_toward_rebuilding), self.model_timestep, self.rebuild_tau)
        else:
            raise ValueError("Attempt to compute prod_max_toward_rebuilding_chg while prod_max_toward_rebuilding is None")

    def calc_overproduction_fast(self):
        dmg_demand = fast.aggregate_rebuild_demand(self.rebuilding_demand)
        if dmg_demand is None:
            dmg_demand = np.full(self.matrix_orders.shape,0.0, dtype="float32")
        self.overprod = fast.calc_overproduction(np.ascontiguousarray(dmg_demand).astype(np.float32), np.ascontiguousarray(self.matrix_orders), np.ascontiguousarray(self.final_demand), self.production, self.overprod_max, self.overprod, self.model_timestep, np.float32(self.overprod_tau), np.float32(self.overprod_base))

    def calc_prod_reqby_demand_fast(self):
        return fast.calc_prod_reqby_demand(np.ascontiguousarray(self.rebuilding_demand), np.ascontiguousarray(self.matrix_orders), np.ascontiguousarray(self.final_demand))

    def calc_orders_fast(self):
        inv_duration = np.nan_to_num(self.inv_duration, posinf=0.)[:,np.newaxis]
        dmg_demand = fast.aggregate_rebuild_demand(self.rebuilding_demand)
        if dmg_demand is None:
            dmg_demand = np.full(self.matrix_orders.shape,0.0, dtype="float32")
        self.matrix_orders = fast.calc_orders(np.ascontiguousarray(dmg_demand).astype(np.float32), np.ascontiguousarray(self.matrix_orders), np.ascontiguousarray(self.final_demand), self.production_cap, self.n_sectors, np.ascontiguousarray(self.tech_mat), inv_duration.astype(np.float32), np.ascontiguousarray(self.matrix_stock).astype(np.float32), self.model_timestep, self.restoration_tau, self.production, self.n_regions, np.ascontiguousarray(self.Z_distrib).astype(np.float32))

    def calc_production_fast(self):
        inv_duration = np.nan_to_num(self.inv_duration, posinf=0.)
        dmg_demand = fast.aggregate_rebuild_demand(self.rebuilding_demand)
        if dmg_demand is None:
            dmg_demand = np.full(self.matrix_orders.shape,0.0, dtype="float32")
        self.production, stock_constraint = fast.calc_production(np.ascontiguousarray(dmg_demand).astype(np.float32), np.ascontiguousarray(self.matrix_orders), np.ascontiguousarray(self.final_demand), self.production_cap, self.n_sectors, np.ascontiguousarray(self.tech_mat), np.float32(self.psi), inv_duration.astype(np.float32), self.n_regions, np.ascontiguousarray(self.matrix_stock).astype(np.float32), np.ascontiguousarray(self.matrix_share_thresh))
        return stock_constraint
