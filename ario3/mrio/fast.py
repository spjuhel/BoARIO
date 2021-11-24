import numpy as np
import numba
from numba import jit, vectorize, njit

@vectorize
def clip(x,up,down):
    if x>up:
        return up
    elif x<down:
        return down
    else:
        return x


@jit("f4[::1,:](f4[::1],int64)", #type: ignore
    nopython=True)
def fast_repeat_A(prod, n):
    a = prod.repeat(n)
    b = a.reshape((-1,n)).T
    return b

@jit(["f4[:,::1](f4[::1],int64)", "int64[:,::1](int64[::1],int64)"], #type: ignore
    nopython=True)
def fast_repeat_B(arr, n):
    a =arr.repeat(n)
    b = a.reshape((-1, n))
    return b

@njit
def fast_repeat_C(arr, n):
    N = len(arr[:,0])
    M = len(arr[0])
    res = np.zeros((N*n, M))
    for i in range(n):
        res[i*N:(i+1)*N,:] = arr
    return res

def aggregate_rebuild_demand(rebuilding_demand):
    if rebuilding_demand is None:
        return None
    else:
        assert rebuilding_demand.ndim == 3
        return np.sum(rebuilding_demand, axis=0)

@jit("f4[::1](f4[::1], f4[::1], f4[::1], f4[::1], f4[::1], f4[::1])",
     nopython=True, cache=True, debug=True) #type: ignore
def calc_production_cap(kapital_lost, VA_0, kstock_ratio_to_VA, production_cap, overprod, production):
    productivity_loss = np.zeros(shape=kapital_lost.shape, dtype="float32")
    k_stock = (VA_0 * kstock_ratio_to_VA)
    productivity_loss[k_stock!=0] = kapital_lost[k_stock!=0] / k_stock[k_stock!=0]
    if (productivity_loss > 0.).any():
        production_cap = np.multiply(production,(1 - productivity_loss))
    if (overprod > 1.0).any():
        production_cap *= overprod
    return production_cap

@jit("f4[::1](f4[:,::1], f4[:,::1], f4[:,::1])",
     nopython=True, cache=True) #type: ignore
def calc_prod_reqby_demand(dmg_demand, matrix_orders, final_demand):
    prod_reqby_demand = np.sum(matrix_orders,axis=1) + np.sum(final_demand, axis=1)
    if dmg_demand is not None:
        prod_reqby_demand += np.sum(dmg_demand, axis=1)
    assert not (prod_reqby_demand < 0).any()
    return prod_reqby_demand

@jit("Tuple((f4[::1], bool_[:,::1]))(f4[:,::1], f4[:,::1], f4[:,::1], f4[::1], int64, f4[:,::1], f4, f4[::1], int64, f4[:,::1], bool_[:,::1])",
     nopython=True, cache=True) #type: ignore
def calc_production(dmg_demand, matrix_orders, final_demand, production_cap, n_sectors, tech_mat, psi, inv_duration, n_regions, matrix_stock, matrix_share_thresh):
    prod_reqby_demand = calc_prod_reqby_demand(dmg_demand, matrix_orders, final_demand)
    production_opt = np.fmin(prod_reqby_demand, production_cap)
    tmp = fast_repeat_A(production_opt, n_sectors)
    supply_constraint = (tmp * tech_mat) * psi
    tmp = fast_repeat_B(inv_duration, n_regions*n_sectors)
    supply_constraint = supply_constraint * tmp
    stock_constraint = (matrix_stock < supply_constraint) * matrix_share_thresh
    if (stock_constraint).any():
        production_ratio_stock = np.ones(shape=matrix_stock.shape)
        for i in range(len(matrix_stock[:,0])):
            for j in range(len(matrix_stock[0])):
                if matrix_share_thresh[i,j] * supply_constraint[i,j] !=0:
                    production_ratio_stock[i,j] = matrix_stock[i,j] / supply_constraint[i,j]
        production_ratio_stock = clip(production_ratio_stock, 1,0) #type: ignore
        if (production_ratio_stock < 1).any():
            production_max = fast_repeat_A(production_opt, n_sectors) * production_ratio_stock
            #assert not (production_max.min(axis=0) < 0).any()
            production = np.empty(production_max.shape[1], dtype="float32")
            for i in range (len(production)):
                production[i] = np.min(production_max[:,i])
        else:
            #assert not (production_opt < 0).any()
            production = production_opt
    else:
        assert not (production_opt < 0).any()
        production = production_opt
    return production, stock_constraint

@jit("f4[:,::1](f4[:,::1], f4[:,::1], f4[:,::1], f4[::1], int64, f4[:,::1], f4[:,::1], f4[:,::1], int64, f4[::1], f4[::1], int64, f4[:,::1])",
     nopython=True, cache=True) #type: ignore
def calc_orders(dmg_demand, matrix_orders, final_demand, production_cap, n_sectors, tech_mat, inv_duration, matrix_stock, model_timestep, restoration_tau, production, n_regions, Z_distrib):
    prod_reqby_demand = calc_prod_reqby_demand(dmg_demand, matrix_orders, final_demand)
    production_opt = np.fmin(prod_reqby_demand, production_cap)
    matrix_stock_goal = fast_repeat_A(production_opt, n_sectors) * tech_mat
    # Check this !
    matrix_stock_gap = np.full(matrix_stock_goal.shape, 0.0, dtype="float32")
    matrix_stock_goal = matrix_stock_goal * inv_duration
    for i in range(len(matrix_stock_gap[:,0])):
        for j in range(len(matrix_stock_gap[0])):
            if np.isfinite(matrix_stock_goal[i,j]):
                matrix_stock_gap[i,j] = matrix_stock_goal[i,j] - matrix_stock[i,j]
    assert (not np.isnan(matrix_stock_gap).any()), "NaN in matrix stock gap"
    matrix_stock_gap = clip(matrix_stock_gap, np.inf,0) #type: ignore
    tmp = model_timestep / restoration_tau.astype(np.float32)
    tmp = fast_repeat_B(tmp,n_regions*n_sectors)
    matrix_stock_gap = tmp * matrix_stock_gap + fast_repeat_A(production, n_sectors) * tech_mat
    res = fast_repeat_C(matrix_stock_gap.astype(np.float32), n_regions) * Z_distrib
    assert not (res < 0).any()
    return res.astype(np.float32)


@jit("Tuple((f4[:,::1], f4[::1], f4[:,::1]))(f4[:,:,::1], f4[::1], f4[:,::1], int64, int64)",
     nopython=True, cache=True) #type: ignore
def calc_rebuilding_production(rebuilding_demand, production, prod_max_toward_rebuilding, model_timestep, rebuild_tau):
    n_demand = rebuilding_demand.shape[0]
    m = rebuilding_demand.shape[1]
    assert n_demand == prod_max_toward_rebuilding.shape[0]
    rebuild_production = fast_repeat_A(production, n_demand)
    rebuild_demand = np.sum(rebuilding_demand, axis=2)
    rebuild_production = rebuild_production * prod_max_toward_rebuilding
    rebuild_production = np.minimum(rebuild_production, rebuild_demand)
    scarcity = np.full(rebuild_production.shape,0.0)
    for i in range(n_demand):
        for j in range(m):
                if rebuild_demand[i,j] > 0:
                    scarcity[i,j] = (rebuild_demand[i,j] - rebuild_production[i,j]) / rebuild_demand[i,j]

    scarcity = clip(scarcity, np.inf, 0.0) #type: ignore

    prod_max_toward_rebuild_chg = ((1. - prod_max_toward_rebuilding) * scarcity * (model_timestep / rebuild_tau) + (0. - prod_max_toward_rebuilding) * (scarcity == 0) * (model_timestep / rebuild_tau))
    #assert not prod_max_toward_rebuild_chg[(prod_max_toward_rebuild_chg < -1) | (prod_max_toward_rebuild_chg > 1)].any()
    prod_max_toward_rebuilding += prod_max_toward_rebuild_chg
    prod_max_toward_rebuilding = clip(prod_max_toward_rebuilding, np.inf, 0.0)  #type: ignore
    prod_max_toward_rebuilding = prod_max_toward_rebuilding.astype(np.float32)
    #assert not prod_max_toward_rebuilding[(prod_max_toward_rebuilding < 0) | (prod_max_toward_rebuilding > 1)].any()
    non_rebuild_production = production - np.sum(rebuild_production, axis=0)
    #assert np.allclose(np.sum(rebuild_production, axis=0) + non_rebuild_production, production)
    return rebuild_production, non_rebuild_production, prod_max_toward_rebuilding

@jit("Tuple((f4[:,::1], f4[::1]))(f4[::1], f4[::1], f4[:,::1], f4[:,::1], int64, int64, f4[:,::1], f4[:,::1], int64)",
     nopython=True, cache=True) #type: ignore
def distribute_nr_production(production, non_rebuild_production, matrix_orders, final_demand, n_sectors, n_regions, tech_mat, matrix_stock, n_fd_cat):
    matrix_I_sum = np.eye(n_sectors)
    matrix_I_sum = fast_repeat_C(matrix_I_sum, 2)
    non_rebuild_demand = np.concatenate((matrix_orders, final_demand), np.int64(1))
    tmp = np.sum(non_rebuild_demand, axis=1)
    tmp = fast_repeat_B(tmp, n_regions*n_sectors)
    demand_share = np.divide(non_rebuild_demand, tmp)
    shape = demand_share.shape
    demand_share = demand_share.ravel()
    for i in range(demand_share.shape[0]):
        if np.isnan(demand_share[i]):
            demand_share[i] = 0
    demand_share = demand_share.reshape(shape)
    tmp = fast_repeat_B(non_rebuild_production, n_regions*n_sectors)
    distributed_non_rebuild_production = np.multiply(demand_share, tmp)

    intmd_distribution = distributed_non_rebuild_production[:,:n_sectors * n_regions]
    stock_use = fast_repeat_A(production, n_sectors) * tech_mat
    stock_add = matrix_I_sum.astype(np.float32) @ np.ascontiguousarray(intmd_distribution)
    matrix_stock = matrix_stock - stock_use + stock_add
    assert not (matrix_stock < 0).any()

    final_demand_not_met = final_demand - distributed_non_rebuild_production[:,n_sectors*n_regions:(n_sectors*n_regions + n_fd_cat*n_regions)]
    final_demand_not_met = np.sum(final_demand_not_met,axis=1)
    return matrix_stock, final_demand_not_met

@jit("f4[:,:,::1](f4[:,:,::1], f4[:,::1], f4[:,:,::1])",
     nopython=True, cache=True) #type: ignore
def distribute_r_production(rebuilding_demand, rebuild_production, tot_rebuilding_demand):
    distributed_rebuild_production = np.full(rebuilding_demand.shape,0.0)
    n = rebuilding_demand.shape[0]
    m = rebuilding_demand.shape[1]
    p = rebuilding_demand.shape[2]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if tot_rebuilding_demand[i,j,k] != 0:
                    distributed_rebuild_production[i,j,k] = (rebuilding_demand[i,j,k] / tot_rebuilding_demand[i,j,k]) * rebuild_production[i,j]
    rebuilding_demand -= distributed_rebuild_production#.reshape(rebuilding_demand.shape)
    return rebuilding_demand

@jit("f4[::1](f4[:,::1], f4[:,::1], f4[:,::1], f4[::1], f4, f4[::1], int64, f4, f4)",
     nopython=True, cache=True) #type: ignore
def calc_overproduction(dmg_demand, matrix_orders, final_demand, production, overprod_max, overprod, model_timestep, overprod_tau, overprod_base):
    prod_reqby_demand = calc_prod_reqby_demand(dmg_demand, matrix_orders, final_demand)
    scarcity = np.full(production.shape, 0.0)
    scarcity[prod_reqby_demand!=0] = (prod_reqby_demand[prod_reqby_demand!=0] - production[prod_reqby_demand!=0]) / prod_reqby_demand[prod_reqby_demand!=0]
    scarcity[np.isnan(scarcity)] = 0
    overprod_chg = (((overprod_max - overprod) * scarcity * (model_timestep / overprod_tau)) + ((overprod_base - overprod) * (scarcity == 0) * model_timestep/overprod_tau)).flatten()
    overprod += overprod_chg
    return overprod

@jit(nopython=True, cache=True) #type: ignore
def check_equilibrium(production, classic_demand_evolution, matrix_stock, matrix_stock_0):
    return (np.allclose(production, classic_demand_evolution) and np.allclose(matrix_stock, matrix_stock_0))
