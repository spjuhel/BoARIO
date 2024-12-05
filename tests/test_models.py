import re
from boario.extended_models import ARIOPsiModel

from boario.model_base import INV_THRESHOLD, ARIOBaseModel
from boario.utils.misc import lexico_reindex
import pytest


import pymrio

import numpy as np
import pandas as pd

from boario.utils.recovery_functions import *

import boario

##### UNIT TESTS


#### INTEGRATION TESTS


@pytest.fixture
def test_mriot():
    mrio = pymrio.load_test()  # .calc_all()
    mrio.aggregate(
        region_agg=["reg1", "reg1", "reg2", "reg2", "reg3", "reg3"],
        sector_agg=[
            "food",
            "mining",
            "manufactoring",
            "other",
            "construction",
            "other",
            "other",
            "other",
        ],
    )
    mrio.calc_all()
    return mrio


@pytest.fixture
def test_model(test_mriot):
    model = ARIOPsiModel(test_mriot)
    return model


class TestARIOPsiModel:
    # Tests creating an instance of the class with valid parameters.
    def test_init_valid_parameters(self, test_mriot):
        model = ARIOPsiModel(test_mriot)
        assert isinstance(model, ARIOPsiModel)
        assert isinstance(model.monetary_factor, (int, np.integer))
        assert model.n_temporal_units_by_step == 1
        assert model.iotable_year_to_temporal_unit_factor == 365
        assert model.overprod_max == 1.25
        assert model.overprod_tau == 1 / 365
        assert model.overprod_base == 1.0
        assert model.order_type == "alt"
        assert model.in_shortage == False
        assert model.had_shortage == False

        assert isinstance(model.rebuild_tau, (int, np.integer))
        test_mriot = lexico_reindex(test_mriot)
        np.testing.assert_allclose(model.Z_0, test_mriot.Z.to_numpy() * 1 / 365)
        assert np.allclose(model.Y_0, test_mriot.Y.to_numpy() * 1 / 365)
        assert np.allclose(model.X_0, test_mriot.x.T.to_numpy().flatten() * 1 / 365)
        assert np.allclose(model.intermediate_demand, test_mriot.Z.to_numpy() * 1 / 365)
        assert np.allclose(
            model.production, test_mriot.x.T.to_numpy().flatten() * 1 / 365
        )
        assert np.allclose(model.final_demand, test_mriot.Y.to_numpy() * 1 / 365)
        assert model.productive_capital_lost is None

    # Tests creating an instance of the class with the `psi_param` parameter as a string with a valid float value.
    def test_init_psi_param_string(self, test_mriot):
        model = ARIOPsiModel(test_mriot, psi_param="0_80")
        assert isinstance(model, ARIOPsiModel)
        assert model.psi == 0.80

    # Tests creating an instance of the class with the `psi_param` parameter as a float value.
    def test_init_psi_param_float(self, test_mriot):
        model = ARIOPsiModel(test_mriot, psi_param=0.80)
        assert isinstance(model, ARIOPsiModel)
        assert model.psi == 0.80

    # Tests creating an instance of the class with the `psi_param` parameter as an integer value.
    def test_init_psi_param_int_invalid(self, test_mriot):
        with pytest.raises(ValueError):
            model = ARIOPsiModel(test_mriot, psi_param=80)

    def test_init_psi_param_int_valid(self, test_mriot):
        model = ARIOPsiModel(test_mriot, psi_param=1)
        assert isinstance(model, ARIOPsiModel)
        assert model.psi == 1.0

    def test_monetary_factor(self, test_mriot):
        model = ARIOPsiModel(test_mriot, monetary_factor=10**7)
        assert model.monetary_factor == 10**7

    def test_warnings(self, test_mriot):
        del test_mriot.meta
        test_mriot.monetary_factor = 1
        with pytest.warns() as record:
            ARIOPsiModel(
                test_mriot, psi_param=1, iotable_year_to_temporal_unit_factor=7
            )

        assert len(record) == 4
        assert (
            str(record[0].message)
            == "It seems the MRIOT you loaded doesn't have metadata to print."
        )
        assert (
            str(record[1].message)
            == "Custom monetary factor found in the IOSystem, continuing with this one (1)"
        )
        assert (
            str(record[2].message)
            == "iotable_to_daily_step_factor is not set to 365 (days). This should probably not be the case if the IO tables you use are on a yearly basis."
        )
        assert (
            str(record[3].message)
            == "No capital to VA dictionary given, considering 4/1 ratio"
        )  #    "It seems the MRIOT you loaded doesn't have metadata to print."

    def test_incomplete_mriot(self, test_mriot):
        Z, Y = test_mriot.Z.copy(), test_mriot.Y.copy()
        del test_mriot.Z
        with pytest.raises(ValueError):
            ARIOPsiModel(test_mriot)
        test_mriot.Z = Z
        del test_mriot.Y
        with pytest.raises(ValueError):
            ARIOPsiModel(test_mriot)
        test_mriot.Y = Y
        del test_mriot.x
        with pytest.raises(ValueError):
            ARIOPsiModel(test_mriot)

    def test_mriot_neg_VA(self, test_mriot):
        test_mriot.x.iloc[0, 0] = 1
        test_mriot.A = pymrio.calc_A(test_mriot.Z, test_mriot.x)
        with pytest.warns(
            UserWarning,
            match=re.compile(
                r"Found negative values in the value added, will set to 0.",
                re.MULTILINE,
            ),
        ):
            ARIOPsiModel(test_mriot)

    def test_productive_capital_vec(self, test_mriot):
        vec = (test_mriot.x.T - test_mriot.Z.sum(axis=0)) * 2
        model = ARIOPsiModel(test_mriot, productive_capital_vector=vec)
        vec_exp = (
            ((test_mriot.x.T - test_mriot.Z.sum(axis=0)) * 2).squeeze().sort_index()
        )
        np.testing.assert_array_equal(model.productive_capital, vec_exp)

    def test_productive_capital_dict(self, test_mriot):
        kratio = {
            "food": 10,
            "mining": 10,
            "manufactoring": 10,
            "other": 10,
            "construction": 10,
        }
        model = ARIOPsiModel(test_mriot, productive_capital_to_VA_dict=kratio)
        kratio_ordered = [kratio[k] for k in sorted(kratio.keys())]
        tiled = np.tile(np.array(kratio_ordered), len(test_mriot.get_regions()))
        VA = (
            ((test_mriot.x.T - test_mriot.Z.sum(axis=0)))
            .squeeze()
            .sort_index()
            .to_numpy()
        )
        np.testing.assert_array_almost_equal(model.VA_0, VA)
        vec_exp = VA * tiled
        np.testing.assert_array_almost_equal(model.productive_capital, vec_exp)

    def test_init_input_stock(self, test_mriot):
        main_inv_dur = 10
        inventory_dict = {
            "food": 1,
            "mining": 10,
            "manufactoring": 5,
            "other": 10,
            "construction": 10,
        }
        infinite_inventories_sect = ["other"]
        with pytest.warns(
            UserWarning,
            match=re.compile(
                r"At least one product has inventory duration lower than the numbers of temporal units in one step",
                re.MULTILINE,
            ),
        ):
            model = ARIOPsiModel(
                test_mriot,
                main_inv_dur=main_inv_dur,
                inventory_dict=inventory_dict,
                infinite_inventories_sect=infinite_inventories_sect,
                temporal_units_by_step=2,
            )

        # Probably to change when issue #122 is solved.
        assert model.inventories == [10, 1, 5, 10, np.inf]
        expected = np.array([10, 2, 5, 10, np.inf]) / 2
        expected[expected < 2] = 2
        np.testing.assert_array_equal(model.inv_duration, expected)

    def test_init_inventory_tau_dict(self, test_mriot):
        inventory_tau_dict = [10.0, 15.0, 13.0, 20.0]
        with pytest.raises(ValueError):
            model = ARIOPsiModel(
                test_mriot,
                inventory_restoration_tau=inventory_tau_dict,
            )

        inventory_tau_dict = {
            "mining": 10,
            "manufactoring": 15,
            "other": 50,
            "construction": 3,
        }
        with pytest.raises(NotImplementedError):
            model = ARIOPsiModel(
                test_mriot,
                inventory_restoration_tau=inventory_tau_dict,
            )

        inventory_tau_dict = {
            "food": 5.5,
            "mining": 10,
            "manufactoring": 15,
            "other": 50,
            "construction": 3,
        }
        with pytest.raises(ValueError):
            model = ARIOPsiModel(
                test_mriot,
                inventory_restoration_tau=inventory_tau_dict,
            )

        inventory_tau_dict = {
            "food": "q",
            "mining": 10,
            "manufactoring": 15,
            "other": 50,
            "construction": 3,
        }
        with pytest.raises(ValueError):
            model = ARIOPsiModel(
                test_mriot,
                inventory_restoration_tau=inventory_tau_dict,
            )

        inventory_tau_dict = {
            "food": 5,
            "mining": 10,
            "manufactoring": 15,
            "other": 50,
            "construction": 3,
        }
        model = ARIOPsiModel(
            test_mriot,
            inventory_restoration_tau=inventory_tau_dict,
        )

        # Expected is 1 / tau, sorted by key
        expected = np.maximum(
            np.array([1 / 3.0, 1 / 5.0, 1 / 15.0, 1 / 10.0, 1 / 50.0]), INV_THRESHOLD
        )
        np.testing.assert_array_almost_equal(model.restoration_tau, expected)

    def test_production_capacity_init_state(self, test_mriot):
        model = ARIOPsiModel(test_mriot)

        assert model.productive_capital_lost is None
        assert model._prod_cap_delta_productive_capital is None
        assert model.prod_cap_delta_productive_capital is None

        assert model.prod_cap_delta_arbitrary is None
        assert model._prod_cap_delta_arbitrary is None

        np.testing.assert_array_equal(
            model._prod_cap_delta_tot, np.zeros_like(test_mriot.x.squeeze())
        )
        np.testing.assert_array_equal(
            model.prod_cap_delta_tot, np.zeros_like(test_mriot.x.squeeze())
        )

    def test_productive_capital_lost(self, test_mriot):
        expected_init_k = np.array(
            [
                1000.0,
                1000.0,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        )
        model = ARIOPsiModel(test_mriot, productive_capital_vector=expected_init_k)

        # Initial state
        np.testing.assert_array_equal(model.productive_capital, expected_init_k)

        # Applying loss
        expected_value = expected_init_k / 10
        expected_delta = np.divide(
            expected_value, expected_init_k, where=expected_init_k != 0
        )
        model.productive_capital_lost = expected_value
        np.testing.assert_array_equal(model.productive_capital_lost, expected_value)
        np.testing.assert_array_equal(
            model.prod_cap_delta_productive_capital, expected_delta
        )
        np.testing.assert_array_equal(model._prod_cap_delta_tot, expected_delta)
        np.testing.assert_array_equal(model.prod_cap_delta_tot, expected_delta)

        # Applying no loss
        model.productive_capital_lost = None
        np.testing.assert_array_equal(model.productive_capital, expected_init_k)
        assert model.productive_capital_lost is None
        assert model._prod_cap_delta_productive_capital is None
        assert model.prod_cap_delta_productive_capital is None
        np.testing.assert_array_equal(
            model._prod_cap_delta_tot, np.zeros_like(test_mriot.x.squeeze())
        )
        np.testing.assert_array_equal(
            model.prod_cap_delta_tot, np.zeros_like(test_mriot.x.squeeze())
        )

        # Applying incorrect loss vector
        expected_value = np.array([100, 100])
        with pytest.raises(ValueError):
            model.productive_capital_lost = expected_value

    def test_arbitrary_capacity_loss(self, test_mriot):
        model = ARIOPsiModel(test_mriot)
        expected_value = np.array(
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        # Applying loss
        model.prod_cap_delta_arbitrary = expected_value
        np.testing.assert_array_equal(model.prod_cap_delta_arbitrary, expected_value)
        np.testing.assert_array_equal(model._prod_cap_delta_arbitrary, expected_value)
        np.testing.assert_array_equal(model._prod_cap_delta_tot, expected_value)
        np.testing.assert_array_equal(model.prod_cap_delta_tot, expected_value)

        # Applying no loss
        model.prod_cap_delta_arbitrary = None
        assert model.prod_cap_delta_arbitrary is None
        assert model._prod_cap_delta_arbitrary is None
        np.testing.assert_array_equal(
            model._prod_cap_delta_tot, np.zeros_like(test_mriot.x.squeeze())
        )
        np.testing.assert_array_equal(
            model.prod_cap_delta_tot, np.zeros_like(test_mriot.x.squeeze())
        )

        # Applying incorrect loss vector
        expected_value = np.array([100, 100])
        with pytest.raises(ValueError):
            model.prod_cap_delta_arbitrary = expected_value

    def test_arbitrary_capital_capacity_loss(self, test_mriot):
        expected_init_k = np.array(
            [
                1000.0,
                1000.0,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        )
        model = ARIOPsiModel(test_mriot, productive_capital_vector=expected_init_k)

        # Initial state
        np.testing.assert_array_equal(model.productive_capital, expected_init_k)

        # Applying loss
        capital_loss = expected_init_k / 10
        arbitratry_loss = np.array(
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        expected_delta_from_k = np.array(
            [0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        expected_delta_tot = np.array(
            [0.5, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        # Applying loss
        model.prod_cap_delta_arbitrary = arbitratry_loss
        model.productive_capital_lost = capital_loss
        np.testing.assert_array_equal(model.prod_cap_delta_arbitrary, arbitratry_loss)
        np.testing.assert_array_equal(model.productive_capital_lost, capital_loss)
        np.testing.assert_array_equal(
            model.prod_cap_delta_productive_capital, expected_delta_from_k
        )
        np.testing.assert_array_equal(model.prod_cap_delta_tot, expected_delta_tot)

        # Removing one loss
        model.prod_cap_delta_arbitrary = None
        np.testing.assert_array_equal(model.prod_cap_delta_tot, expected_delta_from_k)

        # Removing one loss
        model.prod_cap_delta_arbitrary = arbitratry_loss
        model.productive_capital_lost = None
        np.testing.assert_array_equal(model.prod_cap_delta_tot, arbitratry_loss)

    def test_production_capacity(self, test_mriot):
        expected_init_k = np.array(
            [
                1000.0,
                1000.0,
                0.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
                1000.0,
            ]
        )
        model = ARIOPsiModel(test_mriot, productive_capital_vector=expected_init_k)

        expected_prod_cap = (
            test_mriot.x.squeeze().sort_index().copy().to_numpy() * 1 / 365
        )

        np.testing.assert_array_equal(model.production_cap, expected_prod_cap)

        capital_loss = expected_init_k / 10
        arbitratry_loss = np.array(
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        expected_delta_from_k = np.array(
            [0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        expected_delta_tot = np.array(
            [0.5, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        )
        # Applying loss
        model.prod_cap_delta_arbitrary = arbitratry_loss
        np.testing.assert_array_equal(
            model.production_cap, expected_prod_cap * (1 - arbitratry_loss)
        )

        model.productive_capital_lost = capital_loss
        np.testing.assert_array_equal(
            model.production_cap, expected_prod_cap * (1 - expected_delta_tot)
        )

        model.prod_cap_delta_arbitrary = None
        np.testing.assert_array_equal(
            model.production_cap, expected_prod_cap * (1 - expected_delta_from_k)
        )

        with pytest.raises(ValueError):
            model.prod_cap_delta_arbitrary = np.array(
                [
                    2.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            )
            model.production_cap

    def test_normal_demand(self, test_mriot):
        model = ARIOPsiModel(test_mriot)

        test_mriot = lexico_reindex(test_mriot)

        expected_intmd = test_mriot.Z.to_numpy() * 1 / 365
        expected_final = test_mriot.Y.to_numpy() * 1 / 365
        expected_entire = np.concatenate([expected_intmd, expected_final], axis=1)

        # Initial state
        np.testing.assert_array_almost_equal(model.entire_demand, expected_entire)
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.intermediate_demand, expected_intmd)
        np.testing.assert_array_almost_equal(
            model.intermediate_demand_tot, expected_intmd.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.final_demand, expected_final)
        np.testing.assert_array_almost_equal(
            model.final_demand_tot, expected_final.sum(axis=1)
        )

        # modify intmd
        expected_intmd = test_mriot.Z
        expected_entire = np.concatenate([expected_intmd, expected_final], axis=1)
        model.intermediate_demand = expected_intmd
        np.testing.assert_array_almost_equal(model.entire_demand, expected_entire)
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.intermediate_demand, expected_intmd)
        np.testing.assert_array_almost_equal(
            model.intermediate_demand_tot, expected_intmd.sum(axis=1)
        )

        # modify final
        expected_final = test_mriot.Y
        expected_entire = np.concatenate([expected_intmd, expected_final], axis=1)
        model.final_demand = expected_final
        np.testing.assert_array_almost_equal(model.entire_demand, expected_entire)
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.final_demand, expected_final)
        np.testing.assert_array_almost_equal(
            model.final_demand_tot, expected_final.sum(axis=1)
        )

        # modify intmd
        expected_intmd = np.zeros_like(test_mriot.Z)
        expected_entire = np.concatenate([expected_intmd, expected_final], axis=1)
        model.intermediate_demand = None
        np.testing.assert_array_almost_equal(model.entire_demand, expected_entire)
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.intermediate_demand, expected_intmd)
        np.testing.assert_array_almost_equal(
            model.intermediate_demand_tot, expected_intmd.sum(axis=1)
        )

        # modify final
        expected_final = np.zeros_like(test_mriot.Y)
        expected_entire = np.concatenate([expected_intmd, expected_final], axis=1)
        model.final_demand = None
        np.testing.assert_array_almost_equal(model.entire_demand, expected_entire)
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.final_demand, expected_final)
        np.testing.assert_array_almost_equal(
            model.final_demand_tot, expected_final.sum(axis=1)
        )

    def test_rebuild_demand(self, test_mriot):
        model = ARIOPsiModel(test_mriot)

        test_mriot = lexico_reindex(test_mriot)
        expected_intmd = test_mriot.Z.to_numpy() * 1 / 365
        expected_final = test_mriot.Y.to_numpy() * 1 / 365
        expected_entire_noreb = np.concatenate([expected_intmd, expected_final], axis=1)
        expected_reb0 = np.zeros_like(expected_entire_noreb)
        expected_indus = np.ones_like(expected_intmd) * 2
        expected_house = np.ones_like(expected_final) * 3
        expected_reb1 = np.concatenate([expected_indus, expected_house], axis=1)
        expected_entire = np.concatenate([expected_entire_noreb, expected_reb0], axis=1)
        expected_entire_withreb = np.concatenate(
            [expected_entire_noreb, expected_reb1], axis=1
        )
        # Init
        assert model.rebuild_demand is None

        # Check cannot add without changing number of events
        with pytest.raises(RuntimeError):
            model.rebuild_demand = np.ones_like(test_mriot.Z)

        model._n_rebuilding_events = 1
        model._chg_events_number()

        # check no change on normal demand
        np.testing.assert_array_almost_equal(model.entire_demand, expected_entire)
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.intermediate_demand, expected_intmd)
        np.testing.assert_array_almost_equal(
            model.intermediate_demand_tot, expected_intmd.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.final_demand, expected_final)
        np.testing.assert_array_almost_equal(
            model.final_demand_tot, expected_final.sum(axis=1)
        )

        # check rebuild demand is 0 in this case
        np.testing.assert_array_almost_equal(model.rebuild_demand, expected_reb0)
        np.testing.assert_array_almost_equal(
            model.rebuild_demand_tot, expected_reb0.sum(axis=1)
        )

        # check cannot put a wrongly shaped demand:
        with pytest.raises(ValueError):
            model.rebuild_demand = np.ones_like(test_mriot.Z)

        model.rebuild_demand = expected_reb1
        # check no change on intmd and final demand
        np.testing.assert_array_almost_equal(model.intermediate_demand, expected_intmd)
        np.testing.assert_array_almost_equal(
            model.intermediate_demand_tot, expected_intmd.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.final_demand, expected_final)
        np.testing.assert_array_almost_equal(
            model.final_demand_tot, expected_final.sum(axis=1)
        )

        # check rebuild demand
        np.testing.assert_array_almost_equal(model.rebuild_demand, expected_reb1)
        np.testing.assert_array_almost_equal(
            model.rebuild_demand_tot, expected_reb1.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.rebuild_demand_indus, expected_indus)
        np.testing.assert_array_almost_equal(
            model.rebuild_demand_indus_tot, expected_indus.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.rebuild_demand_house, expected_house)
        np.testing.assert_array_almost_equal(
            model.rebuild_demand_house_tot, expected_house.sum(axis=1)
        )

        np.testing.assert_array_almost_equal(
            model.entire_demand, expected_entire_withreb
        )
        np.testing.assert_array_almost_equal(
            model.entire_demand_tot, expected_entire_withreb.sum(axis=1)
        )

    def test_rebuild_prod(self, test_mriot):
        model = ARIOPsiModel(test_mriot)
        assert model.rebuild_prod is None
        assert model.rebuild_prod_indus is None
        assert model.rebuild_prod_house is None
        assert model.rebuild_prod_indus_event(0) is None
        assert model.rebuild_prod_house_event(0) is None
        assert model.rebuild_prod_tot is None

        expected_intmd = test_mriot.Z.to_numpy() * 1 / 365
        expected_final = test_mriot.Y.to_numpy() * 1 / 365
        expected_indus = np.ones_like(expected_intmd) * 2
        expected_house = np.ones_like(expected_final) * 3
        expected_reb1 = np.concatenate([expected_indus, expected_house], axis=1)

        model._n_rebuilding_events = 1
        model.rebuild_prod = expected_reb1

        np.testing.assert_array_almost_equal(model.rebuild_prod, expected_reb1)
        np.testing.assert_array_almost_equal(
            model.rebuild_prod_tot, expected_reb1.sum(axis=1)
        )
        np.testing.assert_array_almost_equal(model.rebuild_prod_indus, expected_indus)
        np.testing.assert_array_almost_equal(model.rebuild_prod_house, expected_house)
        np.testing.assert_array_almost_equal(
            model.rebuild_prod_indus_event(0), expected_indus
        )
        np.testing.assert_array_almost_equal(
            model.rebuild_prod_house_event(0), expected_house
        )

    def test_production_opt(self, test_mriot):
        model = ARIOPsiModel(test_mriot)
        np.testing.assert_array_almost_equal(model.production_cap, model.production_opt)

        model.intermediate_demand = np.zeros_like(model.Z_0)
        np.testing.assert_array_almost_equal(
            model.final_demand_tot, model.production_opt
        )

    def test_calc_production(self, test_mriot):
        model = ARIOPsiModel(test_mriot)

        assert not model.in_shortage
        assert not model.had_shortage

        model.in_shortage = True
        model.calc_production(0)
        assert not model.in_shortage
        np.testing.assert_array_almost_equal(model.production, model.production_opt)

        model.inputs_stock = model.inventory_constraints_opt / 2
        model.calc_production(0)
        np.testing.assert_array_almost_equal(model.production, model.production_opt / 2)

    def test_calc_inventory_constraints(self, test_mriot):
        model = ARIOBaseModel(test_mriot)
        tmp = np.tile(
            np.nan_to_num(model.inv_duration, posinf=0.0)[:, np.newaxis],
            (1, model.n_regions * model.n_sectors),
        )
        expected = model.Z_C * tmp / 365
        expected_2 = expected * 2
        np.testing.assert_array_almost_equal(model.inventory_constraints_opt, expected)
        np.testing.assert_array_almost_equal(model.inventory_constraints_act, expected)
        np.testing.assert_array_almost_equal(
            model.calc_inventory_constraints(model.production * 2), expected_2
        )

        model = ARIOPsiModel(test_mriot)
        tmp = np.tile(
            np.nan_to_num(model.inv_duration, posinf=0.0)[:, np.newaxis],
            (1, model.n_regions * model.n_sectors),
        )
        expected = model.Z_C * model.psi * tmp / 365
        expected_2 = expected * 2
        np.testing.assert_array_almost_equal(model.inventory_constraints_opt, expected)
        np.testing.assert_array_almost_equal(model.inventory_constraints_act, expected)
        np.testing.assert_array_almost_equal(
            model.calc_inventory_constraints(model.production * 2), expected_2
        )

    def test_distribute_production(self, test_mriot):
        model = ARIOBaseModel(test_mriot)
        with pytest.raises(NotImplementedError):
            model.distribute_production("scheme_error")

        model.distribute_production()

        with pytest.raises(RuntimeError):
            model.production *= -1
            model.distribute_production()

        ## add test with rebuild demand

    def test_calc_matrix_stock_gap(self, test_mriot):
        model = ARIOBaseModel(test_mriot)
        goal = model.inputs_stock
        zeros = np.zeros_like(goal)
        np.testing.assert_array_equal(zeros, model.calc_matrix_stock_gap(goal))
        np.testing.assert_array_equal(goal, model.calc_matrix_stock_gap(goal * 2))

        model = ARIOPsiModel(test_mriot)
        tau = np.expand_dims(model.restoration_tau, axis=1)
        goal = model.inputs_stock
        zeros = np.zeros_like(goal)
        np.testing.assert_array_equal(zeros, model.calc_matrix_stock_gap(goal))
        np.testing.assert_array_equal(goal * tau, model.calc_matrix_stock_gap(goal * 2))

    def test_calc_orders_base_alt(self, test_mriot):
        model = ARIOBaseModel(test_mriot)
        model.calc_orders()

    def test_calc_orders_base_noalt(self, test_mriot):
        model = ARIOBaseModel(test_mriot, order_type="noalt")
        model.calc_orders()

    def test_calc_orders_psi_alt(self, test_mriot):
        model = ARIOPsiModel(test_mriot)
        model.calc_orders()

    def test_calc_orders_psi_noalt(self, test_mriot):
        model = ARIOPsiModel(test_mriot, order_type="noalt")
        model.calc_orders()
