from boario.utils.recovery_functions import (
    concave_recovery,
    convexe_recovery,
    convexe_recovery_scaled,
    linear_recovery,
)
import pytest

from contextlib import nullcontext as does_not_raise

# import pandas for the plot
import pandas as pd
# import the different classes
import boario.event as event

import pandas as pd
import pytest


@pytest.fixture
def sample_series():
    return pd.Series(
        {
            ("RegionA", "Sector1"): 30000.0,
            ("RegionA", "Sector2"): 10000.0,
            ("RegionB", "Sector3"): 40000.0,
        }
    )


class TestEventInitSeries:

    @pytest.mark.parametrize("ev_t", ["recovery", "rebuild"])
    def test_event_from_series_invalid(self, ev_t):
        with pytest.raises(ValueError):
            event.from_series(pd.Series(), event_type=ev_t)

        with pytest.raises(ValueError):
            event.from_series(pd.Series([0]), event_type=ev_t)

        with pytest.raises(ValueError):
            event.from_series(pd.Series([-1]), event_type=ev_t)

    def test_event_from_series_recovery(self, sample_series):
        event_instance = event.from_series(
            sample_series, event_type="recovery", recovery_tau=1
        )
        assert event_instance.occurrence == 1
        assert event_instance.duration == 1
        assert event_instance.total_impact == 80000.0
        assert event_instance.name is None
        assert event_instance.recovery_function == event.linear_recovery
        pd.testing.assert_index_equal(
            event_instance.aff_industries, sample_series.index, check_names=False
        )
        pd.testing.assert_index_equal(
            event_instance.aff_regions,
            pd.Index(["RegionA", "RegionB"]),
            check_names=False,
        )
        pd.testing.assert_index_equal(
            event_instance.aff_sectors,
            pd.Index(["Sector1", "Sector2", "Sector3"]),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            event_instance.impact, sample_series, check_names=False
        )
        pd.testing.assert_series_equal(
            event_instance.impact_industries_distrib,
            pd.Series(
                {
                    ("RegionA", "Sector1"): 3 / 8,
                    ("RegionA", "Sector2"): 1 / 8,
                    ("RegionB", "Sector3"): 4 / 8,
                }
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            event_instance.impact_regional_distrib,
            pd.Series({"RegionA": 1 / 2, "RegionB": 1 / 2}),
            check_names=False,
        )

    def test_event_from_series_rebuilding(self, sample_series):
        rebuild_tau = 60
        rebuild_sectors = pd.Series({"SectorRebuild1": 1.0})
        event_instance = event.from_series(
            sample_series,
            event_type="rebuild",
            rebuild_tau=rebuild_tau,
            rebuilding_sectors=rebuild_sectors,
        )
        assert event_instance.occurrence == 1
        assert event_instance.duration == 1
        assert event_instance.total_impact == 80000.0
        assert event_instance.name is None
        assert event_instance.impact_households is None
        assert event_instance.rebuild_tau == 60
        pd.testing.assert_series_equal(
            event_instance.rebuilding_sectors, rebuild_sectors, check_names=False
        )
        pd.testing.assert_index_equal(
            event_instance.aff_industries, sample_series.index, check_names=False
        )
        pd.testing.assert_index_equal(
            event_instance.aff_regions,
            pd.Index(["RegionA", "RegionB"]),
            check_names=False,
        )
        pd.testing.assert_index_equal(
            event_instance.aff_sectors,
            pd.Index(["Sector1", "Sector2", "Sector3"]),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            event_instance.impact, sample_series, check_names=False
        )
        pd.testing.assert_series_equal(
            event_instance.impact_industries_distrib,
            pd.Series(
                {
                    ("RegionA", "Sector1"): 3 / 8,
                    ("RegionA", "Sector2"): 1 / 8,
                    ("RegionB", "Sector3"): 4 / 8,
                }
            ),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            event_instance.impact_regional_distrib,
            pd.Series({"RegionA": 1 / 2, "RegionB": 1 / 2}),
            check_names=False,
        )

    def test_event_from_series_arbitrary(self, sample_series):
        with pytest.raises(NotImplementedError):
            event.from_series(sample_series, event_type="arbitrary")

    def test_event_from_series_wrongtype(self, sample_series):
        with pytest.raises(ValueError):
            event.from_series(sample_series, event_type="wrongtype")  # type: ignore

    def test_event_from_series_notype(self, sample_series):
        with pytest.raises(TypeError):
            event.from_series(sample_series)  # type: ignore


class TestEventInitScalar:

    def test_event_from_scalar_arbitrary(self):
        with pytest.raises(NotImplementedError):
            event.from_scalar_industries(impact=1000, affected_industries=pd.Index([("RegionA", "Sector1")]), event_type="arbitrary")  # type: ignore

    def test_event_from_scalar_wrongtype(self):
        with pytest.raises(ValueError):
            event.from_scalar_industries(impact=1000, affected_industries=pd.Index([("RegionA", "Sector1")]), event_type="wrongtype")  # type: ignore

    def test_event_from_scalar_notype(self):
        with pytest.raises(TypeError):
            event.from_scalar_industries(sample_series, affected_industries=pd.Index([("RegionA", "Sector1")]))  # type: ignore

    def test_event_from_scalar_reg_sec_arbitrary(self):
        with pytest.raises(NotImplementedError):
            event.from_scalar_regions_sectors(
                impact=1000,
                affected_regions=pd.Index(["RegionA"]),
                affected_sectors=pd.Index(["SectorA"]),
                event_type="arbitrary",
            )

    def test_event_from_scalar_reg_sec_wrongtype(self):
        with pytest.raises(ValueError):
            event.from_scalar_regions_sectors(
                impact=1000,
                affected_regions=pd.Index(["RegionA"]),
                affected_sectors=pd.Index(["SectorA"]),
                event_type="wrongtype",
            )

    def test_event_from_scalar_reg_sec_notype(self):
        with pytest.raises(TypeError):
            event.from_scalar_regions_sectors(sample_series, affected_regions=pd.Index(["RegionA"]), affected_sectors=pd.Index(["SectorA"]))  # type: ignore

    @pytest.mark.parametrize(
        "impact, fails",
        [
            (0, True),
            (-1, True),
            (1000, False),
        ],
    )
    @pytest.mark.parametrize(
        "affected_indus, fails2",
        [
            ([], True),
            (pd.Index([]), True),
            ([("RegionA", "Sector1"), ("RegionA", "Sector2")], False),
            (pd.Index([("RegionA", "Sector1"), ("RegionA", "Sector2")]), False),
        ],
    )
    @pytest.mark.parametrize(
        "distrib, fails3, expected",
        [
            (
                None,
                False,
                pd.Series(
                    {("RegionA", "Sector1"): 500.0, ("RegionA", "Sector2"): 500.0}
                ),
            ),
            (
                "equal",
                False,
                pd.Series(
                    {("RegionA", "Sector1"): 500.0, ("RegionA", "Sector2"): 500.0}
                ),
            ),
            ("wrong", True, None),
            (0.5, True, None),
            (pd.Series({("RegionA", "Sector2"): 500.0}), True, None),
            (
                pd.Series(
                    {
                        ("RegionA", "Sector1"): 500.0,
                        ("RegionA", "Sector3"): 500.0,
                    }
                ),
                True,
                None,
            ),
            (
                pd.Series({("RegionA", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                False,
                pd.Series(
                    {("RegionA", "Sector1"): 400.0, ("RegionA", "Sector2"): 600.0}
                ),
            ),
        ],
    )
    def test_distribute_impact_industries(
        self, impact, fails, affected_indus, fails2, distrib, expected, fails3
    ):
        if fails or fails2 or fails3:
            to_raise = pytest.raises(ValueError)
        else:
            to_raise = does_not_raise()
        with to_raise:
            if distrib is not None:
                res = event.Event.distribute_impact_industries(
                    impact, affected_industries=affected_indus, distrib=distrib
                )
            else:
                res = event.Event.distribute_impact_industries(
                    impact, affected_industries=affected_indus
                )
            pd.testing.assert_series_equal(res, expected, check_names=False)

    @pytest.mark.parametrize(
        "impact, imp_fail",
        [(0, True), ([2], True), (1000, False)],
    )
    @pytest.mark.parametrize(
        "affected_industries, aff_indus_fail",
        [
            ([], True),
            (1, True),
            (
                pd.Series({("RegionA", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                True,
            ),
            (pd.Index([("RegionA", "Sector1"), ("RegionA", "Sector2")]), False),
        ],
    )
    @pytest.mark.parametrize(
        "impact_distrib, imp_dist_fail",
        [
            (0, True),
            ([], True),
            (
                pd.Series({("RegionB", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                True,
            ),
            (
                pd.Series({("RegionA", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                False,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "occurrence, occ_fail",
        [(0, True), (5, False)],
    )
    @pytest.mark.parametrize(
        "duration, dur_fail",
        [(0, True), (5, False)],
    )
    @pytest.mark.parametrize(
        "recovery_tau, r_tau_fail",
        [(None, True), (0, True), (60, False)],
    )
    @pytest.mark.parametrize(
        "rebuild_tau, r2_tau_fail",
        [
            (None, False),
            (1, False),
        ],
    )
    def test_from_scalar_industries_recover(
        self,
        impact,
        imp_fail,
        affected_industries,
        aff_indus_fail,
        impact_distrib,
        imp_dist_fail,
        occurrence,
        occ_fail,
        duration,
        dur_fail,
        recovery_tau,
        r_tau_fail,
        rebuild_tau,
        r2_tau_fail,
    ):
        if (
            imp_fail
            or aff_indus_fail
            or imp_dist_fail
            or occ_fail
            or dur_fail
            or r_tau_fail
            or r2_tau_fail
        ):
            to_raise = pytest.raises((ValueError, TypeError, AttributeError))
        else:
            to_raise = does_not_raise()

        with to_raise:
            all_kwargs = {
                "affected_industries": affected_industries,
                "impact_distrib": impact_distrib,
                "occurrence": occurrence,
                "duration": duration,
                "recovery_tau": recovery_tau,
                "rebuild_tau": rebuild_tau,
            }
            kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
            event.from_scalar_industries(impact, event_type="recovery", **kwargs)

    @pytest.mark.parametrize(
        "impact, imp_fail",
        [(0, True), ([2], True), (1000, False)],
    )
    @pytest.mark.parametrize(
        "affected_industries, aff_indus_fail",
        [
            ([], True),
            (1, True),
            (
                pd.Series({("RegionA", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                True,
            ),
            (pd.Index([("RegionA", "Sector1"), ("RegionA", "Sector2")]), False),
        ],
    )
    @pytest.mark.parametrize(
        "impact_distrib, imp_dist_fail",
        [
            (0, True),
            ([], True),
            (
                pd.Series({("RegionB", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                True,
            ),
            (
                pd.Series({("RegionA", "Sector1"): 0.40, ("RegionA", "Sector2"): 0.60}),
                False,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "rebuild_tau, r2_tau_fail",
        [
            (None, True),
            (1, False),
        ],
    )
    @pytest.mark.parametrize(
        "rebuilding_sectors, reb_sec_fail",
        [
            (None, True),
            (pd.Series([]), True),
            (pd.Series({"Sector1": "12"}), True),
            (pd.Series({"Sector1": 14}), True),
            (pd.Series({"Sector1": 0.5, "Sector2": 0.5}), False),
            ({"Sector1": 0.5, "Sector2": 0.5}, False),
        ],
    )
    def test_from_scalar_industries_rebuilding(
        self,
        impact,
        imp_fail,
        affected_industries,
        aff_indus_fail,
        impact_distrib,
        imp_dist_fail,
        rebuild_tau,
        r2_tau_fail,
        rebuilding_sectors,
        reb_sec_fail,
    ):
        if imp_fail or aff_indus_fail or imp_dist_fail or r2_tau_fail or reb_sec_fail:
            to_raise = pytest.raises((ValueError, TypeError, AttributeError))
        else:
            to_raise = does_not_raise()

        with to_raise:
            all_kwargs = {
                "affected_industries": affected_industries,
                "impact_distrib": impact_distrib,
                "rebuild_tau": rebuild_tau,
                "rebuilding_sectors": rebuilding_sectors,
            }
            kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
            event.from_scalar_industries(impact, event_type="rebuild", **kwargs)

    @pytest.mark.parametrize(
        "event_type",
        ["rebuild", "recovery"],
    )
    @pytest.mark.parametrize(
        "affected_regions, aff_reg_fail",
        [([], True), (pd.Index([]), True), ("RegionA", False)],
    )
    @pytest.mark.parametrize(
        "affected_sectors, aff_sec_fail",
        [([], True), (pd.Index([]), True), ("Sector1", False)],
    )
    @pytest.mark.parametrize(
        "impact_regional_distrib, reg_dist_fail",
        [
            ([], True),
            (1, True),
            (pd.Series({"NoRegionA": 1.0}), True),
            (pd.Series({"RegionA": 60}), False),
            (None, False),
            ("equal", False),
        ],
    )
    @pytest.mark.parametrize(
        "impact_sectoral_distrib, sec_dist_fail",
        [
            ([], True),
            (1, True),
            (pd.Series({"NoSector1": 1.0}), True),
            (pd.Series({"Sector1": 60}), False),
            (None, False),
            ("equal", False),
        ],
    )
    def test_from_scalar_region_sectors(
        self,
        event_type,
        affected_regions,
        aff_reg_fail,
        affected_sectors,
        aff_sec_fail,
        impact_regional_distrib,
        reg_dist_fail,
        impact_sectoral_distrib,
        sec_dist_fail,
    ):

        if aff_reg_fail or aff_sec_fail or reg_dist_fail or sec_dist_fail:
            to_raise = pytest.raises((ValueError, TypeError, AttributeError))
        else:
            to_raise = does_not_raise()

        with to_raise:
            all_kwargs = {
                "affected_regions": affected_regions,
                "affected_sectors": affected_sectors,
                "impact_regional_distrib": impact_regional_distrib,
                "impact_sectoral_distrib": impact_sectoral_distrib,
                "rebuild_tau": 1,
                "recovery_tau": 1,
                "rebuilding_sectors": pd.Series({"Sector1": 1.0}),
            }
            kwargs = {k: v for k, v in all_kwargs.items() if v is not None}
            event.from_scalar_regions_sectors(
                impact=1000, event_type=event_type, **kwargs
            )

    @pytest.mark.parametrize(
        "impact_vec, fail",
        [
            ([], True),
            (1, True),
            (None, True),
            (pd.Series([]), True),
            (pd.Series({"Sector1": 100.0, "Sector2": 100.0}), False),
        ],
    )
    @pytest.mark.parametrize(
        "distrib, fail2",
        [
            ([], True),
            (1, True),
            (None, True),
            (pd.Series({"Sector1": 2.0}), True),
            (pd.Series({"Sector1": 1.0}), True),
            (pd.Series({"Sector1": 0.5, "Sector2": 0.5}), False),
        ],
    )
    def test__distribute_impact(self, impact_vec, fail, distrib, fail2):
        if fail or fail2:
            to_raise = pytest.raises((ValueError, TypeError, AttributeError))
        else:
            to_raise = does_not_raise()

        with to_raise:
            ret = event.Event._distribute_impact(impact_vec, distrib)
            pd.testing.assert_series_equal(impact_vec * distrib, ret)

    # Monetary factor
    def test_monetary_factor(self, sample_series):
        event_instance = event.from_series(
            sample_series,
            event_type="recovery",
            recovery_tau=1,
            event_monetary_factor=100,
        )
        assert event_instance.event_monetary_factor == 100

    @pytest.mark.parametrize(
        "h_impact, expected",
        [
            ([], "error"),
            (1, "error"),
            (None, "None"),
            (
                pd.Series({("RegionA", "Sector1"): 1000.0}),
                pd.Series({("RegionA", "Sector1"): 1000.0}),
            ),
            (pd.Series({("RegionA", "Sector1"): 0.5}), "error"),
        ],
    )
    def test_households_impacts(self, sample_series, h_impact, expected):
        if isinstance(expected, str) and expected == "error":
            with pytest.raises(ValueError):
                event_instance = event.from_series(
                    sample_series,
                    event_type="recovery",
                    recovery_tau=1,
                    households_impact=h_impact,
                )
        else:
            event_instance = event.from_series(
                sample_series,
                event_type="recovery",
                recovery_tau=1,
                households_impact=h_impact,
            )
            if not isinstance(expected, str):
                pd.testing.assert_series_equal(event_instance.impact_households, expected)  # type: ignore
            elif expected == "None":
                assert event_instance.impact_households is None

    # recovery function
    def test_event_from_series_recovery(self, sample_series):

        def r_fun1(init_impact_stock, elapsed_temporal_unit, recovery_time):
            return init_impact_stock * (1 - (elapsed_temporal_unit / recovery_time))

        def r_fun2(init_impact_stock, recovery_time):
            return init_impact_stock * (1 - (1 / recovery_time))

        event_instance = event.from_series(
            sample_series, event_type="recovery", recovery_tau=1
        )

        assert event_instance.recovery_tau == 1

        event_instance.recovery_function = None
        assert event_instance.recovery_function == linear_recovery

        event_instance.recovery_function = "convexe"
        assert event_instance.recovery_function == convexe_recovery_scaled

        event_instance.recovery_function = "convexe noscale"
        assert event_instance.recovery_function == convexe_recovery

        event_instance.recovery_function = "concave"
        assert event_instance.recovery_function == concave_recovery

        with pytest.raises(NotImplementedError):
            event_instance.recovery_function = "not existing"

        with pytest.raises(ValueError):
            event_instance.recovery_function = r_fun2

        event_instance.recovery_function = r_fun1
        assert event_instance.recovery_function == r_fun1

        with pytest.raises(ValueError):
            event_instance.recovery_function = 5  # type: ignore

    @pytest.mark.parametrize(
        "impact, expected",
        [
            ([], "error"),
            (1, "error"),
            (
                pd.Series({("RegionA", "Sector1"): 1000.0}),
                "error"
            ),
            (pd.Series({("RegionA", "Sector1"): 0.5}), pd.Series({("RegionA", "Sector1"): 0.5})),
        ],
    )
    def test_event_from_series_arbitrary(self, impact, expected):
        if isinstance(expected, str) and expected == "error":
            with pytest.raises(ValueError):
                event.from_series(
                    impact,
                    event_type="arbitrary",
                    recovery_tau=1,
                )
        else:
            event_instance = event.from_series(
                impact,
                event_type="arbitrary",
                recovery_tau=1,
            )
            pd.testing.assert_series_equal(event_instance.prod_cap_delta_arbitrary, expected)
