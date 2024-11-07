# BoARIO : The Adaptative Regional Input Output model in python.
# Copyright (C) 2022  Samuel Juhel
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import inspect
import math
import warnings
from abc import ABC, abstractmethod
from typing import Callable, List, Literal, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from boario import DEBUG_TRACE, logger
from boario.utils.recovery_functions import (
    concave_recovery,
    convexe_recovery,
    convexe_recovery_scaled,
    linear_recovery,
)

__all__ = [
    "Event",
    "EventKapitalDestroyed",
    "EventArbitraryProd",
    "EventKapitalRecover",
    "EventKapitalRebuild",
    "Impact",
    "IndustriesList",
    "SectorsList",
    "RegionsList",
]

VectorImpact = Union[list, dict, np.ndarray, pd.DataFrame, pd.Series]
ScalarImpact = int | float
Impact = Union[VectorImpact, ScalarImpact]
IndustriesList = Union[List[Tuple[str, str]], pd.MultiIndex]
SectorsList = Union[List[str], pd.Index, str]
RegionsList = Union[List[str], pd.Index, str]
FinalCatList = Union[List[str], pd.Index, str]

REBUILDING_FINALDEMAND_CAT_REGEX = (
    r"(?i)(?=.*household)(?=.*final)(?!.*NPISH|profit).*|HFCE"
)

LOW_DEMAND_THRESH = 10


@overload
def from_series(
    impact: pd.Series,
    *,
    event_type: Literal["recovery"],
    occurrence: int = 1,
    duration: int = 1,
    name: Optional[str] = None,
    event_monetary_factor: int | None = None,
    recovery_tau: int | None = None,
    recovery_function: str | None = "linear",
    households_impact: Impact | None = None,
) -> EventKapitalRecover: ...


@overload
def from_series(
    impact: pd.Series,
    *,
    event_type: Literal["rebuild"],
    occurrence: int = 1,
    duration: int = 1,
    name: Optional[str] = None,
    event_monetary_factor: int | None = None,
    households_impact: Impact | None = None,
    rebuild_tau: int | None = None,
    rebuilding_sectors: dict[str, float] | pd.Series | None = None,
    rebuilding_factor: float | None = 1.0,
) -> EventKapitalRebuild: ...


@overload
def from_series(
    impact: pd.Series,
    *,
    event_type: Literal["arbitrary"],
    occurrence: int = 1,
    duration: int = 1,
    name: Optional[str] = None,
    recovery_tau: int | None = None,
    recovery_function: str | None = "linear",
) -> EventArbitraryProd: ...


def from_series(
    impact: pd.Series,
    *,
    event_type: Literal["recovery"] | Literal["rebuild"] | Literal["arbitrary"],
    occurrence: int = 1,
    duration: int = 1,
    name: Optional[str] = None,
    event_monetary_factor: int | None = None,
    recovery_tau: int | None = None,
    recovery_function: str | None = "linear",
    households_impact: Impact | None = None,
    rebuild_tau: int | None = None,
    rebuilding_sectors: dict[str, float] | pd.Series | None = None,
    rebuilding_factor: float | None = 1.0,
) -> Event:
    """Create an event for an impact given as a pd.Series.

    Parameters
    ----------
    impact : pd.Series
        A pd.Series defining the impact per (region, sector)
    occurrence : int
        The ordinal of occurrence of the event (requires to be > 0). Defaults to 1.
    duration : int
        The duration of the event (entire impact applied during this number of steps). Defaults to 1.
    name : Optional[str]
        A possible name for the event, for convenience. Defaults to None.
    **kwargs :
        Keyword arguments keyword arguments to pass to the instantiating method
    (depends on the type of event).

    Returns
    -------
    Event
        An Event object or one of its subclass

    Raises
    ------
    ValueError
        Raised if impact is empty of contains negative values.

    Examples
    --------
    >>> import pandas as pd
    >>> import pymrio as pym
    >>> from boario.simulation import Simulation
    >>> from boario.extended_model import ARIOPsiModel
    >>> from boario.event import EventKapitalRecover
    >>>
    >>> mriot = pym.load_test()
    >>> mriot.calc_all()
    >>>
    >>> impact_series = pd.Series({('reg1', 'electricity'): 100000.0, ('reg1', 'mining'): 50000.0})
    >>> model = ARIOPsiModel(mriot)
    >>> sim = Simulation(model)
    >>> event = EventKapitalRecover.from_series(impact_series, occurrence=5, duration=10, recovery_time=30, name="Event 1")
    >>> sim.add_event(event)

    """

    if event_type == "rebuild":
        return EventKapitalRebuild._from_series(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            households_impact=households_impact,
            rebuild_tau=rebuild_tau,
            rebuilding_sectors=rebuilding_sectors,
            rebuilding_factor=rebuilding_factor,
        )
    elif event_type == "recovery":
        return EventKapitalRecover._from_series(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            households_impact=households_impact,
            recovery_tau=recovery_tau,
            recovery_function=recovery_function,
        )
    elif event_type == "arbitrary":
        return EventArbitraryProd._from_series(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            recovery_tau=recovery_tau,
            recovery_function=recovery_function,
        )

    else:
        raise ValueError(f"Wrong event type: {event_type}")


def from_scalar_industries(
    impact: int | float,
    *,
    event_type: str,
    affected_industries: IndustriesList,
    impact_distrib: Literal["equal"] | pd.Series = "equal",
    occurrence: int = 1,
    duration: int = 1,
    name: Optional[str] = None,
    event_monetary_factor: int | None = None,
    recovery_tau: int | None = None,
    recovery_function: str | None = "linear",
    households_impact: Impact | None = None,
    rebuild_tau: int | None = None,
    rebuilding_sectors: dict[str, float] | pd.Series | None = None,
    rebuilding_factor: float | None = 1.0,
) -> Event:
    """Creates an Event from a scalar and a list of industries affected.

    The scalar impact is distributed evenly by default. Otherwise it can
    be distributed proportionnaly to the GVA of affected industries, or to
    a custom distribution.

    Parameters
    ----------
    impact : ScalarImpact
        The scalar impact.
    industries : IndustriesList
        The list of industries affected by the impact.
    impact_industries_distrib : Optional[npt.ArrayLike]
        A vector of equal size to the list of industries, stating the
        share of the impact each industry should receive. Defaults to None.
    gva_distrib : Optional[bool]
        A boolean stating if the impact should be distributed proportionnaly to GVA. Defaults to False.
    occurrence : Optional[int]
        The ordinal of occurrence of the event (requires to be > 0). Defaults to 1.
    duration : Optional[int]
        The duration of the event (entire impact applied during this number of steps). Defaults to 1.
    name : Optional[str]
        A possible name for the event, for convenience. Defaults to None.
    **kwargs :
        Keyword arguments
        Other keyword arguments to pass to the instantiate method (depends on the type of event)

    Raises
    ------
    ValueError
        Raised if Impact is null, if len(industries) < 1 or if the sum of impact_industries_distrib differs from 1.0.

    Returns
    -------
    Event
        An Event object or one of its subclass.

    """

    if event_type == "rebuild":
        return EventKapitalRebuild._from_scalar_industries(
            impact=impact,
            affected_industries=affected_industries,
            impact_distrib=impact_distrib,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            households_impact=households_impact,
            rebuild_tau=rebuild_tau,
            rebuilding_sectors=rebuilding_sectors,
            rebuilding_factor=rebuilding_factor,
        )
    elif event_type == "recovery":
        return EventKapitalRecover._from_scalar_industries(
            impact=impact,
            affected_industries=affected_industries,
            impact_distrib=impact_distrib,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            recovery_tau=recovery_tau,
            recovery_function=recovery_function,
        )
    elif event_type == "arbitrary":
        raise NotImplementedError("This type of event is not implemented yet.")
    else:
        raise ValueError(f"Wrong event type: {event_type}")


def from_scalar_regions_sectors(
    impact: int | float,
    *,
    event_type: str,
    affected_regions: RegionsList,
    affected_sectors: SectorsList,
    impact_regional_distrib: Literal["equal"] | pd.Series = "equal",
    impact_sectoral_distrib: Literal["equal"] | pd.Series = "equal",
    occurrence: int = 1,
    duration: int = 1,
    name: Optional[str] = None,
    event_monetary_factor: int | None = None,
    recovery_tau: int | None = None,
    recovery_function: str | None = "linear",
    households_impact: Impact | None = None,
    rebuild_tau: int | None = None,
    rebuilding_sectors: dict[str, float] | pd.Series | None = None,
    rebuilding_factor: float | None = 1.0,
) -> Event:
    """Creates an Event from a scalar, a list of regions and a list of sectors affected.

    Parameters
    ----------
    impact : ScalarImpact
        The scalar impact.
    regions : RegionsList
        The list of regions affected.
    sectors : SectorsList
        The list of sectors affected in each region.
    impact_regional_distrib : Optional[npt.ArrayLike], optional
        A vector of equal size to the list of regions affected, stating the
        share of the impact each industry should receive. Defaults to None.
    impact_sectoral_distrib : Optional[Union[str, npt.ArrayLike]], optional
        Either:

        * ``\"gdp\"``, the impact is then distributed using the gross value added of each sector as a weight.
        * A vector of equal size to the list of sectors affected, stating the share of the impact each industry should receive. Defaults to None.

    occurrence : int, optional
        The ordinal of occurrence of the event (requires to be > 0). Defaults to 1.
    duration : int, optional
        The duration of the event (entire impact applied during this number of steps). Defaults to 1.
    name : Optional[str], optional
        A possible name for the event, for convenience. Defaults to None.
    **kwargs :
        Keyword arguments
        Other keyword arguments to pass to the instantiate method (depends on the type of event)

    Raises
    ------
    ValueError
        Raise if Impact is null, if len(regions) or len(sectors) < 1,

    Returns
    -------
    Event
        An Event object or one of its subclass.
    """

    if event_type == "rebuild":
        return EventKapitalRebuild._from_scalar_regions_sectors(
            impact=impact,
            affected_regions=affected_regions,
            affected_sectors=affected_sectors,
            impact_regional_distrib=impact_regional_distrib,
            impact_sectoral_distrib=impact_sectoral_distrib,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            households_impact=households_impact,
            rebuild_tau=rebuild_tau,
            rebuilding_sectors=rebuilding_sectors,
            rebuilding_factor=rebuilding_factor,
        )
    elif event_type == "recovery":
        return EventKapitalRecover._from_scalar_regions_sectors(
            impact=impact,
            affected_regions=affected_regions,
            affected_sectors=affected_sectors,
            impact_regional_distrib=impact_regional_distrib,
            impact_sectoral_distrib=impact_sectoral_distrib,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            recovery_tau=recovery_tau,
            recovery_function=recovery_function,
        )
    elif event_type == "arbitrary":
        raise NotImplementedError("This type of event is not implemented yet.")
    else:
        raise ValueError(f"Wrong event type: {event_type}")


class Event(ABC):
    r"""An Event object stores all information about a unique shock during simulation
    such as time of occurrence, duration, type of shock, amount of damages.
    Computation of recovery or initially requested rebuilding demand is also
    done in this class.

    .. warning::
       The Event class is abstract and cannot be instantiated directly. Only its non-abstract subclasses can be instantiated.

    .. note::
       Events should be constructed using :py:meth:`~Event.from_series()`, :func:`~event.from_dataframe()`, :func:`~event.from_scalar_industries()` or from :func:`~event.from_scalar_regions_sectors()`.
       Depending on the type of event chosen, these constructors require additional keyword arguments, that are documented for each instantiable Event subclass.
       For instance, :py:class:`EventKapitalRebuild` additionally requires `rebuild_tau` and `rebuilding_sectors`.

    .. seealso::
       Tutorial :ref:`boario-events`
    """

    @abstractmethod
    def __init__(
        self,
        *,
        impact: pd.Series,
        name: str | None = None,
        occurrence: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> None:
        logger.info("Initializing new Event")
        self.name: str | None = name
        r"""An identifying name for the event (for convenience mostly)"""

        self.occurrence = occurrence if occurrence is not None else 1
        self.duration = duration if duration is not None else 1
        self.impact = impact

        self.event_dict: dict = {
            "name": str(self.name),
            "occurrence": self.occurrence,
            "duration": self.duration,
            "aff_regions": list(self.aff_regions),
            "aff_sectors": list(self.aff_sectors),
            "impact": self.total_impact,
            "impact_industries_distrib": list(self.impact_industries_distrib),
            "impact_regional_distrib": list(self.impact_regional_distrib),
        }
        r"""Store relevant information about the event"""

    @classmethod
    def _from_series(
        cls,
        impact: pd.Series,
        *,
        occurrence: Optional[int] = 1,
        duration: Optional[int] = 1,
        name: Optional[str] = None,
        **kwargs,
    ) -> Event:
        if impact.size == 0:
            raise ValueError(
                "Empty impact Series at init, did you not set the impact correctly ?"
            )
        impact = impact[impact != 0]
        if np.less_equal(impact, 0).any():
            if DEBUG_TRACE:
                logger.debug(
                    f"Impact has negative values:\n{impact}\n{impact[impact<0]}"
                )
            raise ValueError("Impact has negative values")
        return cls(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            **kwargs,
        )

    @classmethod
    def _from_scalar_industries(
        cls,
        impact: ScalarImpact,
        *,
        affected_industries: IndustriesList,
        impact_distrib: Literal["equal"] | pd.Series = "equal",
        occurrence: Optional[int] = 1,
        duration: Optional[int] = 1,
        name: Optional[str] = None,
        **kwargs,
    ) -> Event:
        impact_vec = cls.distribute_impact_industries(
            impact=impact,
            affected_industries=affected_industries,
            distrib=impact_distrib,
        )

        return cls._from_series(
            impact=impact_vec,
            occurrence=occurrence,
            duration=duration,
            name=name,
            **kwargs,
        )

    @classmethod
    def distribute_impact_industries(
        cls,
        impact: ScalarImpact,
        affected_industries: IndustriesList,
        distrib: Literal["equal"] | pd.Series = "equal",
    ) -> pd.Series:
        if impact <= 0:
            raise ValueError("Cannot distribute null impact")

        if len(affected_industries) < 1:
            raise ValueError("No affected industries given")

        if isinstance(affected_industries, list):
            affected_industries = pd.MultiIndex.from_tuples(
                affected_industries, names=["region", "sector"]
            )
        impact_vec = pd.Series(impact, dtype="float64", index=affected_industries)
        distrib_vec = cls._level_distrib(affected_industries, distrib)

        return cls._distribute_impact(impact_vec, distrib=distrib_vec)

    @classmethod
    def _distribute_impact(cls, impact_vec: pd.Series, distrib: pd.Series) -> pd.Series:
        if not isinstance(impact_vec, pd.Series):
            raise ValueError(
                f"Impact vector has to be a Series not a {type(impact_vec)}."
            )
        if impact_vec.size < 1:
            raise ValueError(f"Impact vector cannot be null sized.")
        if not math.isclose(distrib.sum(), 1.0, rel_tol=10e-7):
            raise ValueError(
                f"Impact distribution doesn't sum up to 1.0 (but {distrib.sum()})"
            )

        ret = impact_vec * distrib
        if ret.hasnans:
            raise ValueError(
                "Products of impact vector and distrib lead to NaNs, check index matching and values."
            )
        return ret

    @classmethod
    def _from_scalar_regions_sectors(
        cls,
        impact: ScalarImpact,
        *,
        affected_regions: RegionsList,
        affected_sectors: SectorsList,
        impact_regional_distrib: Literal["equal"] | pd.Series = "equal",
        impact_sectoral_distrib: Literal["equal"] | pd.Series = "equal",
        occurrence: int = 1,
        duration: int = 1,
        name: Optional[str] = None,
        **kwargs,
    ) -> Event:
        affected_industries = cls._build_industries_idx(
            regions=affected_regions, sectors=affected_sectors
        )
        regional_distrib = cls._level_distrib(
            affected_industries.levels[0], impact_regional_distrib
        )
        sectoral_distrib = cls._level_distrib(
            affected_industries.levels[1], impact_sectoral_distrib
        )
        industries_distrib = pd.Series(
            np.outer(regional_distrib.values, sectoral_distrib.values).flatten(),  # type: ignore
            index=pd.MultiIndex.from_product(
                [regional_distrib.index, sectoral_distrib.index]
            ),
        )
        impact_vec = cls.distribute_impact_industries(
            impact, affected_industries=affected_industries, distrib=industries_distrib
        )

        return cls._from_series(
            impact=impact_vec,
            occurrence=occurrence,
            duration=duration,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_scalar_regions_sectors(
        cls,
        impact: ScalarImpact,
        *,
        regions: RegionsList,
        sectors: SectorsList,
        impact_regional_distrib: Optional[npt.ArrayLike] = None,
        impact_sectoral_distrib: Optional[Union[str, npt.ArrayLike]] = None,
        occurrence: int = 1,
        duration: int = 1,
        name: Optional[str] = None,
        **kwargs,
    ) -> Event:
        """Creates an Event from a scalar, a list of regions and a list of sectors affected.

        Parameters
        ----------
        impact : ScalarImpact
            The scalar impact.
        regions : RegionsList
            The list of regions affected.
        sectors : SectorsList
            The list of sectors affected in each region.
        impact_regional_distrib : Optional[npt.ArrayLike], optional
            A vector of equal size to the list of regions affected, stating the
            share of the impact each industry should receive. Defaults to None.
        impact_sectoral_distrib : Optional[Union[str, npt.ArrayLike]], optional
            Either:

            * ``\"gdp\"``, the impact is then distributed using the gross value added of each sector as a weight.
            * A vector of equal size to the list of sectors affected, stating the share of the impact each industry should receive. Defaults to None.

        occurrence : int, optional
            The ordinal of occurrence of the event (requires to be > 0). Defaults to 1.
        duration : int, optional
            The duration of the event (entire impact applied during this number of steps). Defaults to 1.
        name : Optional[str], optional
            A possible name for the event, for convenience. Defaults to None.
        **kwargs :
            Keyword arguments
            Other keyword arguments to pass to the instantiate method (depends on the type of event)

        Raises
        ------
        ValueError
            Raise if Impact is null, if len(regions) or len(sectors) < 1,

        Returns
        -------
        Event
            An Event object or one of its subclass.
        """
        if not isinstance(impact, (int, float)):
            raise ValueError("Impact is not scalar.")

        if impact <= 0:
            raise ValueError("Impact is null.")

    @classmethod
    def _build_industries_idx(cls, regions: RegionsList, sectors: SectorsList):
        # TODO: Move this in utils?
        if isinstance(regions, str):
            regions = [regions]

        if isinstance(sectors, str):
            sectors = [sectors]

        _regions = pd.Index(regions, name="region")
        _sectors = pd.Index(sectors, name="sector")

        if len(_regions) < 1:
            raise ValueError("Null sized affected regions ?")

        if len(_sectors) < 1:
            raise ValueError("Null sized affected sectors ?")

        if _sectors.duplicated().any():
            warnings.warn(
                UserWarning(
                    "Multiple presence of the same sector in affected sectors. (Will remove duplicate)"
                )
            )
            _sectors = _sectors.drop_duplicates()

        if _regions.duplicated().any():
            warnings.warn(
                UserWarning(
                    "Multiple presence of the same region in affected region. (Will remove duplicate)"
                )
            )
            _regions = _regions.drop_duplicates()
        return pd.MultiIndex.from_product(
            [_regions, _sectors], names=["region", "sector"]
        )

    @classmethod
    def _level_distrib(
        cls,
        affected_idx: List[str] | pd.Index | pd.MultiIndex,
        distrib: Literal["equal"] | pd.Series,
    ):
        if isinstance(distrib, str) and distrib == "equal":
            return cls._distrib_equi_level(affected_idx)
        else:
            if not isinstance(distrib, pd.Series):
                raise ValueError(
                    "The given impact distribution is incorrect. (Pandas Series required)."
                )
            affected_idx = pd.Index(affected_idx)
            if not affected_idx.isin(distrib.index).all():
                raise ValueError(
                    f"The given impact distribution does not match the impacted industries, regions or sectors:\n affected:\n{affected_idx}\ndistribution:\n{distrib.index}"
                )

        _dist = distrib.loc[affected_idx]
        _dist = _dist.transform(lambda x: x / sum(x))
        return _dist

    @classmethod
    def _distrib_equi_level(
        cls, level_idx: List[str] | pd.Index | pd.MultiIndex
    ) -> pd.Series:
        """Distribute an impact equally between all affected regions.

        Assume impact is given as a vector with all value being the
        total impact to distribute.

        Parameters
        ----------
        impact_vec : pd.Series
            The impact to distribute.

        Returns
        -------
        pd.Series
            The impact vector equally distributed among affected industries.

        """
        return pd.Series(1.0 / len(level_idx), index=level_idx)

    @property
    def impact(self) -> pd.Series:
        r"""A pandas Series with all possible industries as index, holding the impact vector of the event. The impact is defined for each sectors in each region."""
        return self._impact_df

    @impact.setter
    def impact(self, value: pd.Series):
        self._impact_df = value
        self._impact_df.rename_axis(index=["region", "sector"], inplace=True)
        logger.debug("Sorting impact Series")
        self._impact_df.sort_index(inplace=True)
        tmp_idx = self.impact.loc[self.impact > 0].index
        if not isinstance(tmp_idx, pd.MultiIndex):
            raise ValueError("The impact series does not have a MultiIndex index.")
        self._aff_industries: pd.MultiIndex = tmp_idx
        self._aff_regions = self._aff_industries.get_level_values("region").unique()
        self._aff_sectors = self._aff_industries.get_level_values("sector").unique()
        tmp = self.impact.transform(lambda x: x / sum(x), axis=0)
        self.impact_industries_distrib = tmp
        self.total_impact = self.impact.sum()

    @property
    def occurrence(self) -> int:
        r"""The temporal unit of occurrence of the event."""

        return self._occur

    @occurrence.setter
    def occurrence(self, value: int):
        if not value > 0:
            raise ValueError("Occurrence of event cannot be negative or null.")
        else:
            logger.debug(f"Setting occurrence to {value}")
            self._occur = value

    @property
    def duration(self) -> int:
        r"""The duration of the event."""

        return self._duration

    @duration.setter
    def duration(self, value: int):
        if not value > 0:
            raise ValueError("Duration of event cannot be negative or null.")
        else:
            logger.debug(f"Setting duration to {value}")
            self._duration = value

    @property
    def aff_industries(self) -> pd.MultiIndex:
        r"""The industries affected by the event."""
        return self._aff_industries

    @property
    def aff_regions(self) -> pd.Index:
        r"""The array of regions affected by the event"""
        return self._aff_regions

    @property
    def aff_sectors(self) -> pd.Index:
        r"""The array of affected sectors by the event"""
        return self._aff_sectors

    @property
    def impact_regional_distrib(self) -> pd.Series:
        r"""The series specifying how damages are distributed among affected regions"""
        return self._impact_regional_distrib

    @property
    def impact_industries_distrib(self) -> pd.Series:
        r"""The series specifying how damages are distributed among affected industries (regions,sectors)"""
        return self._impact_industries_distrib

    @impact_industries_distrib.setter
    def impact_industries_distrib(self, value: pd.Series):
        self._impact_industries_distrib = value
        tmp = self._impact_industries_distrib.groupby(
            "region",
            observed=False,
        ).sum()
        self._impact_regional_distrib = tmp

    def __repr__(self):
        # TODO: find ways to represent long lists
        return f"""[WIP]
        {self.__class__}(
              name = {self.name},
              occur = {self.occurrence},
              duration = {self.duration}
              aff_regions = {self.aff_regions.to_list()},
              aff_sectors = {self.aff_sectors.to_list()},
             )
        """


class EventKapitalDestroyed(Event, ABC):
    r"""EventKapitalDestroyed is an abstract class to hold events with where some capital (from industries or households) is destroyed. See :py:class:`EventKapitalRecover` and :py:class:`EventKapitalRebuild` for its instantiable classes.

    .. note::
       For this type of event, the impact value represent the amount of capital destroyed in monetary terms.

    .. note::
       We distinguish between impacts on household and industrial (productive) capital. We assume destruction of the former not to reduce production capacity contrary to the latter (but possibly induce reconstruction demand). Impacts on household capital is null by default, but can be set via the ``households_impacts`` argument in the constructor. The amount of production capacity lost is computed as the share of capital lost over total capital of the industry.

    .. note::
       The user can specify a monetary factor via the ``event_monetary_factor`` argument for the event if it differs from the monetary factor of the MRIOT used. By default the constructor assumes the two factors to be the same (i.e., if the MRIOT is in €M, the so is the impact).

    .. seealso::
       Tutorial :ref:`boario-events`
    """

    def __init__(
        self,
        *,
        impact: pd.Series,
        households_impact: Optional[pd.Series] = None,
        name: str | None = None,
        occurrence: int = 1,
        duration: int = 1,
        event_monetary_factor: Optional[int] = None,
    ) -> None:
        if event_monetary_factor is None:
            logger.info(f"No event monetary factor given. Assuming it is 1.")
            self.event_monetary_factor = 1
            r"""The monetary factor for the impact of the event (e.g. 10**6, 10**3, ...)"""

        else:
            self.event_monetary_factor = event_monetary_factor

        self._check_negligeable_impact(impact)
        super().__init__(
            impact=impact,
            name=name,
            occurrence=occurrence,
            duration=duration,
        )

        self._impact_households = None
        # The only thing we have to do is affecting/computing the regional_sectoral_productive_capital_destroyed
        self.total_productive_capital_destroyed = self.total_impact
        logger.info(
            f"Total impact on productive capital is {self.total_productive_capital_destroyed} (with monetary factor: {self.event_monetary_factor})"
        )
        if households_impact is not None:
            if not isinstance(households_impact, pd.Series):
                raise ValueError(
                    "Households impacts have to be a Series with regions and sectors affected as multiindex."
                )
            self._check_negligeable_impact(households_impact)
            self.impact_households = households_impact
        if self.impact_households is not None:
            logger.info(
                f"Total impact on households is {self.impact_households.sum()} (with monetary factor: {self.event_monetary_factor})"
            )

    @property
    def impact_households(self) -> pd.Series | None:
        r"""A pandas Series with all possible (regions, final_demand_cat) as index, holding the households impacts vector of the event. The impact is defined for each region and each final demand category."""
        return self._impact_households

    @impact_households.setter
    def impact_households(self, value: pd.Series | None):
        self._impact_households = value

    def _check_negligeable_impact(self, impact: pd.Series):
        if (impact < LOW_DEMAND_THRESH / self.event_monetary_factor).all():
            raise ValueError(
                "Impact is too small to be considered by the model. Check you units perhaps ?"
            )
        if negligeable := (
            impact < LOW_DEMAND_THRESH / self.event_monetary_factor
        ).sum():
            warnings.warn(
                f"Impact for some industries ({negligeable} total), is smaller than {LOW_DEMAND_THRESH / self.event_monetary_factor} and will be considered as 0. by the model."
            )


class EventKapitalRebuild(EventKapitalDestroyed):
    r"""EventKapitalRebuild holds a :py:class:`EventKapitalDestroyed` event where the destroyed capital requires to be rebuilt, and creates a reconstruction demand.

    This subclass requires and enables new arguments to pass to the constructor:

    * A characteristic time for reconstruction (``tau_rebuild``)
    * A set of sectors responsible for the reconstruction (``rebuilding_sectors``)
    * A ``rebuilding_factor`` in order to modulate the reconstruction demand. By default, this factor is 1, meaning that the entire impact value is translated as an additional demand.

    .. note::
       The ``tau_rebuild`` of an event takes precedence over the one defined for a model.

    .. seealso::
       Tutorial :ref:`boario-events`

    """

    def __init__(
        self,
        *,
        impact: pd.Series,
        households_impact: pd.Series | None = None,
        name: str | None = None,
        occurrence: int = 1,
        duration: int = 1,
        event_monetary_factor: Optional[int] = None,
        rebuild_tau: int,
        rebuilding_sectors: dict[str, float] | pd.Series,
        rebuilding_factor: float = 1.0,
    ) -> None:
        super().__init__(
            impact=impact,
            households_impact=households_impact,
            name=name,
            occurrence=occurrence,
            duration=duration,
            event_monetary_factor=event_monetary_factor,
        )
        self.rebuild_tau = rebuild_tau
        self.rebuilding_sectors = rebuilding_sectors
        self.rebuilding_factor = rebuilding_factor
        self.event_dict["rebuilding_sectors"] = {
            sec: share for sec, share in self.rebuilding_sectors.items()
        }

    @property
    def rebuild_tau(self) -> int:
        r"""The characteristic time for rebuilding."""
        return self._rebuild_tau

    @rebuild_tau.setter
    def rebuild_tau(self, value: int):
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"``rebuild_tau`` should be a strictly positive integer. Value given is {value}."
            )
        else:
            self._rebuild_tau = value

    @property
    def rebuilding_sectors(self) -> pd.Series:
        r"""The (optional) array of rebuilding sectors"""
        return self._rebuilding_sectors

    @rebuilding_sectors.setter
    def rebuilding_sectors(self, value: dict[str, float] | pd.Series):
        if value is None:
            raise ValueError(f"Rebuilding sectors cannot be empty/none.")
        if isinstance(value, dict):
            reb_sectors = pd.Series(value)
        else:
            reb_sectors = value
        if not is_numeric_dtype(reb_sectors):
            raise TypeError(
                "Rebuilding sectors should be given as ``dict[str, float] | pd.Series``."
            )
        if not np.isclose(reb_sectors.sum(), 1.0):
            raise ValueError(f"Reconstruction shares among sectors do not sum up to 1.")

        self._rebuilding_sectors = reb_sectors


class EventKapitalRecover(EventKapitalDestroyed):
    r"""EventKapitalRecover holds a :py:class:`EventKapitalDestroyed` event where the destroyed capital does not create a reconstruction demand.

    This subclass requires and enables new arguments to pass to the constructor:

    * A characteristic time for the recovery (``recovery_tau``)
    * Optionally a ``recovery_function`` (linear by default).

    .. seealso::
       Tutorial :ref:`boario-events`
    """

    def __init__(
        self,
        *,
        impact: pd.Series,
        recovery_tau: int,
        recovery_function: str = "linear",
        households_impact: Optional[pd.Series] = None,
        name: str | None = None,
        occurrence: int = 1,
        duration: int = 1,
        event_monetary_factor: int | None = None,
    ) -> None:
        super().__init__(
            impact=impact,
            households_impact=households_impact,
            name=name,
            occurrence=occurrence,
            duration=duration,
            event_monetary_factor=event_monetary_factor,
        )
        self.recovery_tau = recovery_tau
        self.recovery_function = recovery_function

    @property
    def recovery_tau(self) -> int:
        return self._recovery_tau

    @recovery_tau.setter
    def recovery_tau(self, value: int):
        if (not isinstance(value, int)) or (value <= 0):
            raise ValueError(f"Invalid recovery tau: {value} (positive int required).")
        self._recovery_tau = value

    @property
    def recovery_function(self) -> Callable:
        r"""The recovery function used for recovery (`Callable`)"""
        return self._recovery_fun

    @recovery_function.setter
    def recovery_function(self, r_fun: str | Callable | None):
        if r_fun is None:
            r_fun = "linear"
        if isinstance(r_fun, str):
            if r_fun == "linear":
                fun = linear_recovery
            elif r_fun == "convexe":
                fun = convexe_recovery_scaled
            elif r_fun == "convexe noscale":
                fun = convexe_recovery
            elif r_fun == "concave":
                fun = concave_recovery
            else:
                raise NotImplementedError(
                    "No implemented recovery function corresponding to {}".format(r_fun)
                )
        elif callable(r_fun):
            r_fun_argsspec = inspect.getfullargspec(r_fun)
            r_fun_args = r_fun_argsspec.args + r_fun_argsspec.kwonlyargs
            if not all(
                args in r_fun_args
                for args in [
                    "init_impact_stock",
                    "elapsed_temporal_unit",
                    "recovery_tau",
                ]
            ):
                raise ValueError(
                    "Recovery function has to have at least the following keyword arguments: {}".format(
                        [
                            "init_impact_stock",
                            "elapsed_temporal_unit",
                            "recovery_tau",
                        ]
                    )
                )
            fun = r_fun

        else:
            raise ValueError("Given recovery function is not a str or callable")

        self._recovery_fun = fun


class EventArbitraryProd(Event):
    r"""An EventArbitraryProd object holds an event with arbitrary impact on production capacity.

    Such events can be used to represent temporary loss of production capacity in a completely exogenous way (e.g., loss of working hours from a heatwave).

    .. warning::
       This type of event suffers from a problem with the recovery and does not function properly at the moment.

    .. note::
       For this type of event, the impact value represent the share of production capacity lost of an industry.

    .. note::
       In addition to the base arguments of an Event, EventArbitraryProd requires a ``recovery_time`` (1 step by default) and a ``recovery_function`` (linear by default).

    .. seealso::
       Tutorial :ref:`boario-events`
    """

    def __init__(
        self,
        *,
        impact: pd.Series,
        recovery_tau: int = 1,
        recovery_function: str = "linear",
        name: str | None = None,
        occurrence: int = 1,
        duration: int = 1,
    ) -> None:
        if (impact > 1.0).any():
            raise ValueError(
                "Impact is greater than 100% (1.) for at least an industry."
            )

        super().__init__(
            impact=impact,
            name=name,
            occurrence=occurrence,
            duration=duration,
        )

        self._prod_cap_delta_arbitrary_0 = (
            self.impact.copy()
        )  # np.zeros(shape=len(self.possible_sectors))
        self.prod_cap_delta_arbitrary = (
            self.impact.copy()
        )  # type: ignore # np.zeros(shape=len(self.possible_sectors))
        self.recovery_tau = recovery_tau
        r"""The characteristic recovery duration after the event is over"""

        self.recovery_function = recovery_function

        logger.info("Initialized")

    @property
    def recovery_tau(self) -> int:
        return self._recovery_tau

    @recovery_tau.setter
    def recovery_tau(self, value: int):
        if (not isinstance(value, int)) or (value <= 0):
            raise ValueError(f"Invalid recovery tau: {value} (positive int required).")
        self._recovery_tau = value

    @property
    def prod_cap_delta_arbitrary(self) -> pd.Series:
        r"""The optional array storing arbitrary (as in not related to productive_capital destroyed) production capacity loss"""
        return self._prod_cap_delta_arbitrary

    @prod_cap_delta_arbitrary.setter
    def prod_cap_delta_arbitrary(self, value: pd.Series):
        self._prod_cap_delta_arbitrary = value

    @property
    def recovery_function(self) -> Callable:
        r"""The recovery function used for recovery (`Callable`)"""
        return self._recovery_fun

    @recovery_function.setter
    def recovery_function(self, r_fun: str | Callable | None):
        if r_fun is None:
            r_fun = "instant"
        if self.recovery_tau is None:
            raise AttributeError(
                "Impossible to set recovery function if no recovery time is given."
            )
        if isinstance(r_fun, str):
            if r_fun == "linear":
                fun = linear_recovery
            elif r_fun == "convexe":
                fun = convexe_recovery_scaled
            elif r_fun == "convexe noscale":
                fun = convexe_recovery
            elif r_fun == "concave":
                fun = concave_recovery
            else:
                raise NotImplementedError(
                    "No implemented recovery function corresponding to {}".format(r_fun)
                )
        elif callable(r_fun):
            r_fun_argsspec = inspect.getfullargspec(r_fun)
            r_fun_args = r_fun_argsspec.args + r_fun_argsspec.kwonlyargs
            if not all(
                args in r_fun_args
                for args in [
                    "init_impact_stock",
                    "elapsed_temporal_unit",
                    "recovery_tau",
                ]
            ):
                raise ValueError(
                    "Recovery function has to have at least the following keyword arguments: {}".format(
                        [
                            "init_impact_stock",
                            "elapsed_temporal_unit",
                            "recovery_tau",
                        ]
                    )
                )
            fun = r_fun

        else:
            raise ValueError("Given recovery function is not a str or callable")

        self._recovery_fun = fun
