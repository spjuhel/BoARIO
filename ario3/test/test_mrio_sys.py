import numpy as np
from ario3.mrio import MrioSystem
import pytest
import pickle
import pymrio as pym
import json
import copy
import pathlib
from ario3.utils.logger import init_logger

logger = init_logger(__name__, pathlib.Path.cwd()/"test_run.log")

with pathlib.Path("ario3/test/mock_data/mrio_params.json").absolute().open('r') as fp:
    exio_params = json.load(fp)

with pathlib.Path("ario3/test/mock_data/params.json").absolute().open('r') as fp:
    params = json.load(fp)

result_storage = pathlib.Path("ario3/test/mock_data/results/")

with pathlib.Path("ario3/test/mock_data/mrio.pkl").absolute().open('rb') as fp:
    exio = pickle.load(fp)

test_mrio = MrioSystem(exio, exio_params, params, result_storage)

print('done')

def test_init():
    assert test_mrio.n_regions == test_mrio.regions.size
    assert test_mrio.n_sectors == test_mrio.sectors.size


def test_production_cap():
    mrio = copy.copy(test_mrio)
    prod_cap = mrio.production_cap * 1.0
    mrio.calc_production_cap()
    logger.debug("Maximum value in difference: %f (should be 0.0)",(prod_cap - mrio.production_cap).max())
    assert np.allclose(prod_cap, mrio.production_cap)

    mrio.kapital_lost = (mrio.VA_0 * mrio.kstock_ratio_to_VA) / 2
    mrio.calc_production_cap()
    assert np.allclose(prod_cap/2, mrio.production_cap)

def test_production():
    mrio = copy.copy(test_mrio)
    prod = mrio.production
    check = mrio.calc_production()
    logger.debug((prod - mrio.production).max())
    assert np.allclose(prod, mrio.production)
    assert np.logical_not(check).all()

    mrio.production_cap = mrio.production_cap / 2
    check = mrio.calc_production()
    assert np.allclose(mrio.production, mrio.production_cap)
    assert np.logical_not(check).all()

    mrio.production_cap *= 2
    check = mrio.calc_production()
    assert np.allclose(prod, mrio.production)
    assert np.logical_not(check).all()

    mrio.matrix_stock = mrio.matrix_stock * (mrio.psi + 0.000001)
    check = mrio.calc_production()
    assert np.allclose(prod, mrio.production)
    assert np.logical_not(check).all()

    mrio.production = test_mrio.production
    mrio.matrix_stock = mrio.matrix_stock * (mrio.psi - 0.0001)
    check = mrio.calc_production()
    inv_dur = np.logical_not(np.isfinite(mrio.inv_duration))
    thresh = mrio.matrix_share_thresh
    never_limiting_stocks = np.logical_not(thresh)
    check2 = np.logical_or(check,never_limiting_stocks)
    # logger.debug("n sectors: %f", mrio.n_sectors)
    # logger.debug("n regions: %f", mrio.n_regions)
    # logger.debug("Infinite stocks: "+np.array2string(inv_dur[:, np.newaxis]*np.full(check.shape, True)))
    # logger.debug("never limiting or currently limiting stocks: "+np.array2string(check2))
    final_check = (np.logical_or(check2, inv_dur[:, np.newaxis]))
    #problematic_idx = np.flatnonzero(np.logical_not(final_check))
    problematic_idx2 = np.nonzero(np.logical_not(final_check))
    # logger.debug("infinite or never limiting or currently limiting stocks: "+np.array2string(final_check))
    # logger.debug("Thresh at idx: "+np.array2string(mrio.matrix_share_thresh[problematic_idx2]))
    # logger.debug("Stock contraigning at idx: "+np.array2string(check[problematic_idx2]))
    # logger.debug("Infinite ? at idx: "+np.array2string((inv_dur[:, np.newaxis]*np.full(check.shape, True))[problematic_idx2]))
    assert (final_check).all()
    mrio.production = test_mrio.production

#def test_orders():
#    mrio = copy.copy(test_mrio)
#    mrio.calc_orders()
#    assert np.allclose(mrio.matrix_orders, test_mrio.matrix_orders)

#    mrio.matrix_orders = test_mrio.matrix_orders
#    mrio.matrix_stock = mrio.matrix_stock * 0.5
#    mrio.calc_orders()
#    check_orders = np.full(mrio.matrix_stock.shape,0.0)
#    check_orders[np.isfinite(mrio.matrix_stock)] = (test_mrio.matrix_stock * 0.5 * (test_mrio.model_timestep/test_mrio.restoration_tau)[:,np.newaxis])[np.isfinite(test_mrio.matrix_stock)]
#    check_orders += (np.tile(test_mrio.production, (test_mrio.n_sectors, 1)) * test_mrio.tech_mat)
#    check_orders = (np.tile(check_orders, (mrio.n_regions, 1)) * mrio.Z_distrib)
#    assert np.allclose(mrio.matrix_orders, check_orders)
