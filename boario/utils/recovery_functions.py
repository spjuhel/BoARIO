from typing import Optional

import numpy as np


def linear_recovery(
    elapsed_temporal_unit: int,
    init_impact_stock: np.ndarray,
    recovery_time: int,
):
    r"""Linear Initial impact recovery function

    Initial impact is entirely recovered when `recovery_time` has passed since event
    started recovering

    Parameters
    ----------

    init_impact_stock : float
        Initial Initial impact destroyed

    elapsed_temporal_unit : int
        Elapsed time since event started recovering

    recovery_time : int
        Total time it takes the event to fully recover

    """

    return init_impact_stock * (1 - (elapsed_temporal_unit / recovery_time))


def convexe_recovery(
    elapsed_temporal_unit: int,
    init_impact_stock: np.ndarray,
    recovery_time: int,
):
    r"""Convexe Initial impact recovery function

    Initial impact is recovered with characteristic time `recovery_time`. (This doesn't mean Initial impact is fully recovered after this time !)
    This function models a recovery similar as the one happening in the rebuilding case, for the same characteristic time.

    Parameters
    ----------

    init_impact_stock : float
        Initial Initial impact destroyed

    elapsed_temporal_unit : int
        Elapsed time since event started recovering

    recovery_time : int
        Total time it takes the event to fully recover

    """

    return init_impact_stock * (1 - (1 / recovery_time)) ** elapsed_temporal_unit


def convexe_recovery_scaled(
    elapsed_temporal_unit: int,
    init_impact_stock: np.ndarray,
    recovery_time: int,
    scaling_factor: float = 4,
):
    r"""Convexe Initial impact recovery function (scaled to match other recovery duration)

    Initial impact is mostly recovered (>95% by default for most cases) when `recovery_time` has passed since event
    started recovering.

    Parameters
    ----------

    init_impact_stock : float
        Initial Initial impact destroyed

    elapsed_temporal_unit : int
        Elapsed time since event started recovering

    recovery_time : int
        Total time it takes the event to fully recover

    scaling_factor: float
        Used to scale the exponent in the function so that Initial impact is mostly rebuilt after `recovery_time`. A value of 4 insure >95% of Initial impact is recovered for a reasonable range of `recovery_time` values.

    """

    return init_impact_stock * (1 - (1 / recovery_time)) ** (
        scaling_factor * elapsed_temporal_unit
    )


def concave_recovery(
    elapsed_temporal_unit: int,
    init_impact_stock: np.ndarray,
    recovery_time: int,
    steep_factor: float = 0.000001,
    half_recovery_time: Optional[int] = None,
):
    r"""Concave (s-shaped) Initial impact recovery function

    Initial impact is mostly (>95% in most cases) recovered when `recovery_time` has passed since event started recovering.

    Parameters
    ----------

    init_impact_stock : float
        Initial Initial impact destroyed

    elapsed_temporal_unit : int
        Elapsed time since event started recovering

    recovery_time : int
        Total time it takes the event to fully recover

    steep_factor: float
        This coefficient governs the slope of the central part of the s-shape, smaller values lead to a steeper slope. As such it also affect the percentage of Initial impact rebuilt after `recovery_time` has elapsed. A value of 0.000001 should insure 95% of the initial impact is rebuild for a reasonable range of recovery duration.

    half_recovery_time : int
        This can by use to change the time the inflexion point of the s-shape curve is attained. By default it is set to half the recovery duration.

    """

    if half_recovery_time is None:
        tau_h = 2
    else:
        tau_h = recovery_time / half_recovery_time
    exponent = (np.log(recovery_time) - np.log(steep_factor)) / (
        np.log(recovery_time) - np.log(tau_h)
    )
    return (init_impact_stock * recovery_time) / (
        recovery_time + steep_factor * (elapsed_temporal_unit**exponent)
    )
