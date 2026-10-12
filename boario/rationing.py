"""Rationing schemes distributing production across demands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, ClassVar, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from boario.model_base import ARIOBaseModel

import numpy as np
import numpy.typing as npt

__all__ = [
    "RationingScheme",
    "Proportional",
    "DomesticFirst",
    "BiggestFirst",
    "ShortfallFirst",
    "SCHEMES",
    "get_scheme",
]


def _domestic_first_priority(
    model, regions: Sequence[str] | None = None
) -> npt.NDArray:
    """Priority matrix for domestic-first rationing in selected regions.

    Suppliers located in `regions` serve domestic intermediate demand first
    (tier 0) and everything else after (tier 1). Suppliers elsewhere put all
    demand in tier 1, i.e. they ration proportionally.

    Parameters
    ----------
    regions : sequence of str, optional
        Regions whose industries apply domestic-first. All regions if None.

    Returns
    -------
    npt.NDArray
        Integer tiers, same shape as `entire_demand`. Lower is served first.

    Raises
    ------
    ValueError
        If a name in `regions` is not a region of the model.
    """
    n_r, n_s = model.n_regions, model.n_sectors
    if regions is None:
        active = np.ones(n_r, dtype=bool)
    else:
        unknown = set(regions) - set(model.regions)
        if unknown:
            raise ValueError(f"Unknown regions: {sorted(unknown)}")
        active = np.asarray(model.regions.isin(regions))

    region_idx = np.repeat(np.arange(n_r), n_s)  # region of each (region, sector)
    domestic = region_idx[:, None] == region_idx[None, :]
    applies = active[region_idx][:, None]  # per supplier row

    priority = np.ones(model.entire_demand.shape, dtype=int)
    priority[:, : n_r * n_s] = np.where(domestic & applies, 0, 1)
    return priority


def _allocate_by_priority(
    demand: npt.NDArray, production: npt.NDArray, priority: npt.NDArray
) -> npt.NDArray:
    """Allocate production to demands tier by tier.

    Lower `priority` values are served first. Within a tier, rationing is
    proportional to demand.

    Parameters
    ----------
    demand : npt.NDArray
        Demand matrix, one row per supplier.
    production : npt.NDArray
        Production of each supplier, shape ``(demand.shape[0],)``.
    priority : npt.NDArray
        Integer tier of each entry of `demand`, same shape as `demand`.

    Returns
    -------
    npt.NDArray
        Allocation matrix, same shape as `demand`, with ``alloc <= demand``.
    """
    alloc = np.zeros_like(demand)
    remaining = production.copy()
    for level in np.unique(priority):
        tier = np.where(priority == level, demand, 0.0)
        tier_tot = tier.sum(axis=1)
        given = np.minimum(tier_tot, remaining)
        shares = np.divide(
            tier,
            tier_tot[:, None],
            out=np.zeros_like(tier),
            where=tier_tot[:, None] != 0,
        )
        alloc += shares * given[:, None]
        remaining -= given
    return alloc


def _allocate_by_order(
    demand: npt.NDArray, production: npt.NDArray, order: npt.NDArray
) -> npt.NDArray:
    """Allocate production to demands in a strict per-supplier order.

    Parameters
    ----------
    demand : npt.NDArray
        Demand matrix, one row per supplier.
    production : npt.NDArray
        Production of each supplier, shape ``(demand.shape[0],)``.
    order : npt.NDArray
        Column indices per row, served first to last (e.g. from ``np.argsort``).

    Returns
    -------
    npt.NDArray
        Allocation matrix, same shape as `demand`, with ``alloc <= demand``.
    """
    d_sorted = np.take_along_axis(demand, order, axis=1)
    before = np.cumsum(d_sorted, axis=1) - d_sorted
    a_sorted = np.clip(production[:, None] - before, 0.0, d_sorted)
    alloc = np.empty_like(demand)
    np.put_along_axis(alloc, order, a_sorted, axis=1)
    return alloc


def _distribute_proportional(
    demand: npt.NDArray, production: npt.NDArray
) -> npt.NDArray:
    """Distribute production proportionally to demand.

    Each supplier's production is split across its demands in proportion
    to the amounts demanded. Suppliers with no demand distribute nothing.

    Parameters
    ----------
    demand : npt.NDArray
        Demand matrix, one row per supplier.
    production : npt.NDArray
        Production of each supplier, shape ``(demand.shape[0],)``.

    Returns
    -------
    npt.NDArray
        Distributed production, same shape as `demand`.

    Notes
    -----
    Shares are scaled by the full `production`, so allocations can exceed
    demand when production exceeds total demand.
    """
    demand_tot = np.sum(demand, axis=1, keepdims=True)
    demand_shares = np.zeros_like(demand)
    np.divide(demand, demand_tot, where=(demand_tot != 0), out=demand_shares)
    return demand_shares * production[:, np.newaxis]


def _shortfall_priority_key(model, other_demand_key: float = 0.0) -> npt.NDArray:
    """Priority key from the buyers' relative input shortfall.

    The shortfall of buyer ``j`` for input product ``s`` is
    ``1 - inputs_stock / stock_goal``, clipped to ``[0, 1]``. It is already
    an accumulated signal: unmet orders lower the stock, which raises the
    shortfall at the next step.

    Parameters
    ----------
    other_demand_key : float, default 0.0
        Key given to final and rebuilding demand, which have no stock.

    Returns
    -------
    npt.NDArray
        Key matrix, same shape as `entire_demand`. Higher means served first.
    """
    goal = np.tile(model.production_opt, (model.n_sectors, 1)) * model.tech_mat
    with np.errstate(invalid="ignore"):
        goal *= model.inv_duration[:, np.newaxis]
    shortfall = np.zeros_like(goal)
    ok = np.isfinite(goal) & (goal > 0)
    shortfall[ok] = np.clip(1.0 - model.inputs_stock[ok] / goal[ok], 0.0, 1.0)

    key = np.full(model.entire_demand.shape, other_demand_key)
    n_int = model.n_regions * model.n_sectors
    key[:, :n_int] = np.tile(shortfall, (model.n_regions, 1))
    return key


class RationingScheme(Protocol):
    """Rule distributing each supplier's production across its demands."""

    name: ClassVar[str]

    def __call__(self, model: ARIOBaseModel) -> npt.NDArray:
        """Distribute production.

        Parameters
        ----------
        model : ARIOBaseModel
            Model at the current step.

        Returns
        -------
        npt.NDArray
            Distributed production, same shape as ``model.entire_demand``.
        """
        ...


@dataclass(frozen=True)
class Proportional:
    """Ration proportionally to demand."""

    name: ClassVar[str] = "proportional"

    def __call__(self, model: ARIOBaseModel) -> npt.NDArray:
        return _distribute_proportional(model.entire_demand, model.production)


@dataclass(frozen=True)
class DomesticFirst:
    """Serve domestic intermediate demand first in the given regions.

    Parameters
    ----------
    regions : tuple of str, optional
        Regions whose suppliers apply the rule. All regions if None.
    """

    name: ClassVar[str] = "domestic_first"
    regions: tuple[str, ...] | None = None

    def __call__(self, model: ARIOBaseModel) -> npt.NDArray:
        priority = _domestic_first_priority(model, self.regions)
        return _allocate_by_priority(model.entire_demand, model.production, priority)


@dataclass(frozen=True)
class BiggestFirst:
    """Serve the largest demands first."""

    name: ClassVar[str] = "biggest_first"

    def __call__(self, model: ARIOBaseModel) -> npt.NDArray:
        order = np.argsort(-model.entire_demand, axis=1, kind="stable")
        return _allocate_by_order(model.entire_demand, model.production, order)


@dataclass(frozen=True)
class ShortfallFirst:
    """Serve the buyers with the largest relative input shortfall first.

    Parameters
    ----------
    other_demand_key : float, default 0.0
        Key given to final and rebuilding demand.
    """

    name: ClassVar[str] = "shortfall_first"
    other_demand_key: float = 0.0

    def __call__(self, model: ARIOBaseModel) -> npt.NDArray:
        key = _shortfall_priority_key(model, self.other_demand_key)
        order = np.argsort(-key, axis=1, kind="stable")
        return _allocate_by_order(model.entire_demand, model.production, order)


SCHEMES: dict[str, Callable[[], RationingScheme]] = {
    cls.name: cls for cls in (Proportional, DomesticFirst, BiggestFirst, ShortfallFirst)
}


def get_scheme(scheme: str | RationingScheme) -> RationingScheme:
    """Return a scheme instance from its name, or pass an instance through.

    Parameters
    ----------
    scheme : str or RationingScheme
        Registered scheme name, or a ready-made scheme instance.

    Returns
    -------
    RationingScheme
        The scheme instance. Named schemes use their default parameters.

    Raises
    ------
    ValueError
        If `scheme` is a string that is not a registered name.
    """
    if not isinstance(scheme, str):
        return scheme
    try:
        factory = SCHEMES[scheme]
    except KeyError:
        raise ValueError(
            f"Unknown scheme {scheme!r}, available: {sorted(SCHEMES)}"
        ) from None
    return factory()
