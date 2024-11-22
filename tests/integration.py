import pandas as pd
import pymrio
import pytest

from boario import event
from boario.extended_models import ARIOPsiModel
from boario.simulation import Simulation


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
    model = ARIOPsiModel(
        pym_mrio=test_mrio,
        order_type="alt",
        alpha_base=1.0,
        alpha_max=1.25,
        alpha_tau=365,
        rebuild_tau=60,
        main_inv_dur=90,
        monetary_factor=10**6,
        temporal_units_by_step=1,
        iotable_year_to_temporal_unit_factor=365,
        infinite_inventories_sect=None,
        inventory_dict=None,
        productive_capital_vector=None,
        productive_capital_to_VA_dict=None,
        psi_param=0.80,
        inventory_restoration_tau=60,
    )
    return model


@pytest.fixture
def test_sim(test_model):
    sim = Simulation(test_model)
    return sim


def test_minor_rec_event(test_sim):
    impact = 50.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    ev = event.from_scalar_regions_sectors(
        impact=impact,
        event_type="recovery",
        duration=duration,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        impact_regional_distrib="equal",
        impact_sectoral_distrib="equal",
        recovery_tau=180,
        recovery_function="linear",
    )
    test_sim.add_event(ev)
    test_sim.loop()

    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[0] == 0.0
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[1] == pytest.approx(
        impact
    )
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[
        1 + duration
    ] == pytest.approx(impact)
    assert (
        test_sim.productive_capital_to_recover.sum(axis=1).iloc[3 + duration] < impact
    )

    ## Impact should be fully recovered after 210 days.
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[210] == 0.0

    # production capacity of unaffected sector should be unchanged
    production_cap_norm = (
        test_sim.production_capacity / test_sim.production_capacity.loc[0]
    )
    production_cap_norm_no_impact = production_cap_norm[
        production_cap_norm.columns.difference(
            pd.MultiIndex.from_product([affected_regions, affected_sectors])
        )
    ]
    pd.testing.assert_frame_equal(
        production_cap_norm_no_impact,
        pd.DataFrame(
            1.0,
            index=production_cap_norm_no_impact.index,
            columns=production_cap_norm_no_impact.columns,
        ),
    )

    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()

    # All sectors within the region should have some, negligible, effects.
    assert (min_values < 1.0).all()

    # But all sectors within the region, different from the impacted ones, should have some, *negligible*, effects.
    assert (
        min_values.drop(["manufactoring", "mining"])
        > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_medium_rec_event(test_sim):
    impact = 500000000.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)
    ev = event.from_scalar_regions_sectors(
        impact=impact,
        event_type="recovery",
        duration=duration,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
        recovery_tau=180,
        recovery_function="convexe",
    )
    test_sim.add_event(ev)
    test_sim.loop()

    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[0] == 0.0
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[1] == pytest.approx(
        impact
    )
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[
        1 + duration
    ] == pytest.approx(impact)
    assert (
        test_sim.productive_capital_to_recover.sum(axis=1).iloc[3 + duration] < impact
    )

    # production capacity of unaffected sector should never be below 1.0
    production_cap_norm = (
        test_sim.production_capacity / test_sim.production_capacity.loc[0]
    ).min()
    production_cap_norm_no_impact = production_cap_norm[
        production_cap_norm.index.difference(
            pd.MultiIndex.from_product([affected_regions, affected_sectors])
        )
    ]
    pd.testing.assert_series_equal(
        production_cap_norm_no_impact,
        pd.Series(1.0, index=production_cap_norm_no_impact.index),
    )

    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 0.9999).all()


def test_shortage_rec_event(test_sim):
    impact = 5000000000.0
    duration = 120
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)
    ev = event.from_scalar_regions_sectors(
        impact=impact,
        event_type="recovery",
        duration=duration,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
        recovery_tau=180,
        recovery_function="convexe",
    )
    test_sim.add_event(ev)
    test_sim.loop()

    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[0] == 0.0
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[1] == pytest.approx(
        impact
    )
    assert test_sim.productive_capital_to_recover.sum(axis=1).iloc[
        1 + duration
    ] == pytest.approx(impact)
    assert (
        test_sim.productive_capital_to_recover.sum(axis=1).iloc[3 + duration] < impact
    )

    # production capacity of unaffected sector should never be below 1.0
    production_cap_norm = (
        test_sim.production_capacity / test_sim.production_capacity.loc[0]
    ).min()
    production_cap_norm_no_impact = production_cap_norm[
        production_cap_norm.index.difference(
            pd.MultiIndex.from_product([affected_regions, affected_sectors])
        )
    ]
    pd.testing.assert_series_equal(
        production_cap_norm_no_impact,
        pd.Series(1.0, index=production_cap_norm_no_impact.index),
    )

    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 0.9999).all()
    assert test_sim.model.had_shortage


def test_crashing_rec_event(test_sim):
    impact = 5000000000.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)
    ev = event.from_scalar_regions_sectors(
        impact=impact,
        event_type="recovery",
        duration=duration,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
        recovery_tau=180,
        recovery_function="convexe",
    )
    test_sim.add_event(ev)
    with pytest.raises(RuntimeError):
        test_sim.loop()


def test_minor_reb_event(test_sim):
    impact = 50.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)
    ev = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=impact,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        rebuilding_sectors={"construction": 1.0},
        duration=duration,
        rebuild_tau=180,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop(["mining", "manufactoring"])
        > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_medium_reb_event(test_sim):
    impact = 500000.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)
    ev = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=impact,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        rebuilding_sectors={"construction": 1.0},
        duration=duration,
        rebuild_tau=180,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop(["mining", "manufactoring"])
        > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_shortage_reb_event(test_sim):
    impact = 5000000.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)
    ev = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=impact,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        rebuilding_sectors={"construction": 1.0},
        duration=duration,
        rebuild_tau=180,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop(["mining", "manufactoring"])
        > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()
    # production capacity of unaffected sector should never be below 1.0
    production_cap_norm = (
        test_sim.production_capacity / test_sim.production_capacity.loc[0]
    ).min()
    production_cap_norm_no_impact = production_cap_norm[
        production_cap_norm.index.difference(
            pd.MultiIndex.from_product([affected_regions, affected_sectors])
        )
    ]
    pd.testing.assert_series_equal(
        production_cap_norm_no_impact,
        pd.Series(1.0, index=production_cap_norm_no_impact.index),
    )

    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 0.9999).all()
    assert test_sim.model.had_shortage


def test_multiple_rec_event(test_sim):
    ev1 = event.from_scalar_regions_sectors(
        event_type="recovery",
        impact=1000000,
        affected_regions=["reg1"],
        affected_sectors=["manufactoring", "food"],
        duration=20,
        recovery_tau=180,
        impact_regional_distrib="equal",
        impact_sectoral_distrib="equal",
    )
    ev2 = event.from_scalar_regions_sectors(
        event_type="recovery",
        occurrence=7,
        impact=1000000,
        affected_regions=["reg2"],
        affected_sectors=["manufactoring", "food"],
        duration=20,
        recovery_tau=180,
        impact_regional_distrib="equal",
        impact_sectoral_distrib="equal",
    )
    test_sim.add_events([ev1, ev2])
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    # assert (min_values < (1.0 - 1 / test_sim.model.monetary_factor)).all()
    # assert test_sim.model.had_shortage


def test_multiple_reb_event(test_sim):
    ev1 = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=1000000,
        affected_regions=["reg1"],
        affected_sectors=["manufactoring", "food"],
        rebuilding_sectors={"construction": 1.0},
        duration=30,
        rebuild_tau=100,
        impact_regional_distrib="equal",
        impact_sectoral_distrib="equal",
    )
    ev2 = event.from_scalar_regions_sectors(
        event_type="rebuild",
        occurrence=7,
        impact=1000000,
        affected_regions=["reg2"],
        affected_sectors=["manufactoring", "food"],
        rebuilding_sectors={"construction": 1.0},
        duration=30,
        rebuild_tau=100,
        impact_regional_distrib="equal",
        impact_sectoral_distrib="equal",
    )
    test_sim.add_events([ev1, ev2])
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    # assert (min_values < (1.0 - 1 / test_sim.model.monetary_factor)).all()
    # assert test_sim.model.had_shortage


def test_household_reb_event(test_sim):
    impact = 500000.0
    duration = 20
    affected_regions = ["reg1"]
    affected_sectors = ["manufactoring", "mining"]
    va = test_sim.model.mriot.x.T - test_sim.model.mriot.Z.sum(axis=0)

    household_impact = pd.Series(
        [500000.0],
        index=pd.MultiIndex.from_tuples(
            [("reg1", "Final consumption expenditure by households")]
        ),
    )

    ev = event.from_scalar_regions_sectors(
        event_type="rebuild",
        impact=impact,
        households_impact=household_impact,
        affected_regions=affected_regions,
        affected_sectors=affected_sectors,
        rebuilding_sectors={"construction": 1.0},
        duration=duration,
        rebuild_tau=180,
        impact_regional_distrib="equal",
        impact_sectoral_distrib=va.loc["indout", "reg1"],
    )
    test_sim.add_event(ev)
    test_sim.loop()
    min_values = (
        test_sim.production_realised.loc[:, "reg1"]
        / test_sim.production_realised.loc[0, "reg1"]
    ).min()
    assert (min_values < 1.0).all()
    assert (
        min_values.drop(["mining", "manufactoring"])
        > (1.0 - 1 / test_sim.model.monetary_factor)
    ).all()


def test_arbitrary_events(test_sim):
    impact = pd.Series(
        [0.2, 0.3],
        index=pd.MultiIndex.from_tuples(
            [("reg1", "mining"), ("reg1", "manufactoring")]
        ),
    )
    duration = 20
    ev = event.from_series(
        event_type="arbitrary",
        impact=impact,
        duration=duration,
        recovery_function="concave",
        recovery_tau=180,
    )
    test_sim.add_event(ev)
    test_sim.loop()
