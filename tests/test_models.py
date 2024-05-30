from boario.extended_models import ARIOPsiModel

import pytest

# import pymrio for the test MRIO
import pymrio

# import pandas for the plot
import pandas as pd
import numpy as np
import numpy.testing as nptest

# import the different classes
import boario
from boario.utils.recovery_functions import *

boario.disable_console_logging()

@pytest.fixture
def test_mrio():
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
def test_model(test_mrio):
    model = ARIOPsiModel(test_mrio)
    return model


class TestARIOPsiModel:
    # Tests creating an instance of the class with valid parameters.
    def test_init_valid_parameters(self, test_mrio):
        model = ARIOPsiModel(test_mrio)
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

        assert np.allclose(model.Z_0, test_mrio.Z.to_numpy() * 1 / 365)
        assert np.allclose(model.Y_0, test_mrio.Y.to_numpy() * 1 / 365)
        assert np.allclose(model.X_0, test_mrio.x.T.to_numpy().flatten() * 1 / 365)
        assert np.allclose(model.matrix_orders, test_mrio.Z.to_numpy() * 1 / 365)
        assert np.allclose(
            model.production, test_mrio.x.T.to_numpy().flatten() * 1 / 365
        )
        assert np.allclose(model.intmd_demand, test_mrio.Z.to_numpy() * 1 / 365)
        assert np.allclose(model.final_demand, test_mrio.Y.to_numpy() * 1 / 365)
        assert np.allclose(
            model.productive_capital_lost, np.zeros(shape=len(test_mrio.x.squeeze()))
        )

    # Tests creating an instance of the class with the `psi_param` parameter as a string with a valid float value.
    def test_init_psi_param_string(self, test_mrio):
        model = ARIOPsiModel(test_mrio, psi_param="0_80")
        assert isinstance(model, ARIOPsiModel)
        assert model.psi == 0.80

    # Tests creating an instance of the class with the `psi_param` parameter as a float value.
    def test_init_psi_param_float(self, test_mrio):
        model = ARIOPsiModel(test_mrio, psi_param=0.80)
        assert isinstance(model, ARIOPsiModel)
        assert model.psi == 0.80

    # Tests creating an instance of the class with the `psi_param` parameter as an integer value.
    def test_init_psi_param_int_invalid(self, test_mrio):
        with pytest.raises(ValueError):
            model = ARIOPsiModel(test_mrio, psi_param=80)

    def test_init_psi_param_int_valid(self, test_mrio):
        model = ARIOPsiModel(test_mrio, psi_param=1)
        assert isinstance(model, ARIOPsiModel)
        assert model.psi == 1.0
