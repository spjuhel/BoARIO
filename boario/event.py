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
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_numeric_dtype


from boario import logger
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
ScalarImpact = Union[int, float, np.integer]
Impact = Union[VectorImpact, ScalarImpact]
IndustriesList = Union[List[Tuple[str, str]], pd.MultiIndex]
SectorsList = Union[List[str], pd.Index, str]
RegionsList = Union[List[str], pd.Index, str]
FinalCatList = Union[List[str], pd.Index, str]

REBUILDING_FINALDEMAND_CAT_REGEX = (
    r"(?i)(?=.*household)(?=.*final)(?!.*NPISH|profit).*|HFCE"
)

LOW_DEMAND_THRESH = 10


class Event(ABC):
    r"""An Event object stores all information about a unique shock during simulation
    such as time of occurrence, duration, type of shock, amount of damages.
    Computation of recovery or initially requested rebuilding demand is also
    done in this class.

    .. warning::
       The Event class is abstract and cannot be instantiated directly. Only its non-abstract subclasses can be instantiated.

    .. note::
       Events should be constructed using :py:meth:`~Event.from_series()`, :py:meth:`~Event.from_dataframe()`, :py:meth:`~Event.from_scalar_industries()` or from :py:meth:`~Event.from_scalar_regions_sectors()`.
       Depending on the type of event chosen, these constructors require additional keyword arguments, that are documented for each instantiable Event subclass.
       For instance, :py:class:`EventKapitalRebuild` additionally requires `rebuild_tau` and `rebuilding_sectors`.

    .. seealso::
       Tutorial :ref:`boario-events`
    """

    possible_sectors: pd.Index = pd.Index([])
    r"""List of sectors present in the MRIOT used by the model"""

    possible_regions: pd.Index = pd.Index([])
    r"""List of regions present in the MRIOT used by the model"""

    possible_final_demand_cat: pd.Index = pd.Index([])
    r"""List of final demand categories present in the MRIOT used by the model"""

    temporal_unit_range: int = 0
    r"""Maximum temporal unit simulated"""

    z_shape: tuple[int, int] = (0, 0)
    r"""Shape of the Z (intermediate consumption) matrix in the model"""

    y_shape: tuple[int, int] = (0, 0)
    r"""Shape of the Y (final demand) matrix in the model"""

    x_shape: tuple[int, int] = (0, 0)
    r"""Shape of the x (production) vector in the model"""

    regions_idx: npt.NDArray = np.array([])
    r"""lexicographic region indexes"""

    sectors_idx: npt.NDArray = np.array([])
    r"""lexicographic sector indexes"""

    model_monetary_factor: int = 1
    r"""Amount of unitary currency used in the MRIOT (e.g. 1000000 if in € millions)"""

    gva_df: pd.Series = pd.Series([], dtype="float64")
    r"""GVA per (region,sector)"""

    sectors_gva_shares: npt.NDArray = np.array([])
    r"""Fraction of total (regional) GVA for each sectors"""

    Z_distrib: npt.NDArray = np.array([])
    r"""Normalized intermediate consumption matrix"""

    Y_distrib: npt.NDArray = np.array([])
    r"""Normalized final consumption matrix"""

    mrio_name: str = ""
    r"""MRIOT identification"""

    @abstractmethod
    def __init__(
        self,
        *,
        impact: pd.Series,
        name: str | None = "Unnamed",
        occurrence: int = 1,
        duration: int = 1,
    ) -> None:
        logger.info("Initializing new Event")
        logger.debug("Checking required Class attributes are defined")

        if np.size(self.possible_regions) == 0 or np.size(self.possible_sectors) == 0:
            raise AttributeError(
                "It appears that no model has been instantiated as some class attributes are not initialized (possible_regions, possible_sectors). Events require to instantiate a model and a simulation context before they can be instantiated"
            )

        if self.temporal_unit_range == 0:
            raise AttributeError(
                "It appears that no simulation context has been instantiated as some class attributes are not initialized (temporal_unit_range). Events require to instantiate a model and a simulation context before they can be instantiated"
            )

        self.name: str = name if name else "unnamed"
        r"""An identifying name for the event (for convenience mostly)"""

        self.occurrence = occurrence
        self.duration = duration
        self.impact_df = impact

        self.happened: bool = False
        r"""States if the event happened"""

        self.over: bool = False
        r"""States if the event is over"""

        self.event_dict: dict = {
            "name": str(self.name),
            "occurrence": self.occurrence,
            "duration": self.duration,
            "aff_regions": list(self.aff_regions),
            "aff_sectors": list(self.aff_sectors),
            "impact": self.total_impact,
            "impact_industries_distrib": list(self.impact_industries_distrib),
            "impact_regional_distrib": list(self.impact_regional_distrib),
            "globals_vars": {
                "possible_sectors": list(self.possible_sectors),
                "possible_regions": list(self.possible_regions),
                "temporal_unit_range": self.temporal_unit_range,
                "z_shape": self.z_shape,
                "y_shape": self.y_shape,
                "x_shape": self.x_shape,
                "model_monetary_factor": self.model_monetary_factor,
                "mrio_used": self.mrio_name,
            },
        }
        r"""Store relevant information about the event"""

    @classmethod
    @abstractmethod
    def _instantiate(
        cls,
        impact: pd.Series,
        *,
        occurrence: int = 1,
        duration: int = 1,
        name: Optional[str] = None,
        **_,
    ):
        return cls(impact=impact, occurrence=occurrence, duration=duration, name=name)

    @classmethod
    def from_series(
        cls,
        impact: pd.Series,
        *,
        occurrence: int = 1,
        duration: int = 1,
        name: Optional[str] = None,
        **kwargs,
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
        if impact.size == 0:
            raise ValueError(
                "Empty impact Series at init, did you not set the impact correctly ?"
            )
        impact = impact[impact != 0]
        if np.less_equal(impact, 0).any():
            logger.debug(f"Impact has negative values:\n{impact}\n{impact[impact<0]}")
            raise ValueError("Impact has negative values")
        return cls._instantiate(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            **kwargs,
        )

    @classmethod
    def from_dataframe(
        cls,
        impact: pd.DataFrame,
        *,
        occurrence: int = 1,
        duration: int = 1,
        name: Optional[str] = None,
        **kwargs,
    ) -> Event:
        """Convenience function for DataFrames. See :meth:`~boario.event.Event.from_series`. This constructor only apply ``.squeeze()`` to the given DataFrame.

        Parameters
        ----------
        impact : pd.DataFrame
           A pd.DataFrame defining the impact per (region, sector)
        occurrence : int
            The ordinal of occurrence of the event (requires to be > 0). Defaults to 1.
        duration : int
            The duration of the event (entire impact applied during this number of steps). Defaults to 1.
        name : Optional[str]
            A possible name for the event, for convenience. Defaults to None.
        **kwargs :
            Keyword arguments
            Other keyword arguments to pass to the instantiate method (depends on the type of event)

        Raises
        ------
        ValueError
            If impact cannot be squeezed to a Series

        Returns
        -------
        Event
           An Event object (or one of its subclass).
        """
        impact = impact.squeeze()
        if not isinstance(impact, pd.Series):
            raise ValueError("Could not squeeze impact dataframe to a serie.")

        return cls.from_series(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            **kwargs,
        )

    @classmethod
    def distribute_impact_by_gva(cls, impact_vec: pd.Series) -> pd.Series:
        """Distribute a vector of impact by the GVA of affected industries.

        Each values of the given impact are mutliplied by the share of the GVA
        the industry has over the GVA of all affected industries.

        Parameters
        ----------
        impact_vec : pd.Series
            The impact values to be reweigthed. Current use-case assumes all values are the total impact.

        Returns
        -------
        pd.Series
            The impact where each value was multiplied by the share of GVA of each affected industry (over total GVA affected).

        """
        gva = cls.gva_df.loc[impact_vec.index]
        gva = gva.transform(lambda x: x / sum(x))
        return impact_vec * gva

    @classmethod
    def distribute_impact_equally(cls, impact_vec: pd.Series) -> pd.Series:
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
        dfg = impact_vec.groupby("region")
        return dfg.transform(lambda x: x / (dfg.ngroups * x.count()))

    @classmethod
    def from_scalar_industries(
        cls,
        impact: ScalarImpact,
        *,
        industries: IndustriesList,
        impact_industries_distrib: Optional[npt.ArrayLike] = None,
        gva_distrib: Optional[bool] = False,
        occurrence: Optional[int] = 1,
        duration: Optional[int] = 1,
        name: Optional[str] = None,
        **kwargs,
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
        if impact <= 0:
            raise ValueError("Impact is null")

        if len(industries) < 1:
            raise ValueError("Null sized affected industries ?")

        if isinstance(industries, list):
            industries = pd.MultiIndex.from_tuples(
                industries, names=["region", "sector"]
            )

        impact_vec = pd.Series(impact, dtype="float64", index=industries)

        if impact_industries_distrib:
            if np.sum(impact_industries_distrib) != 1.0:
                raise ValueError("Impact distribution doesn't sum up to 1.0")
            else:
                impact_vec *= impact_industries_distrib  # type: ignore

        elif gva_distrib:
            impact_vec = cls.distribute_impact_by_gva(impact_vec)

        else:
            impact_vec = cls.distribute_impact_equally(impact_vec)

        return cls.from_series(
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

        industries = pd.MultiIndex.from_product(
            [_regions, _sectors], names=["region", "sector"]
        )

        impact_vec = pd.Series(impact, dtype="float64", index=industries)

        assert isinstance(_regions, pd.Index)
        if impact_regional_distrib is None:
            regional_distrib = pd.Series(1.0 / len(_regions), index=_regions)
        elif not isinstance(impact_regional_distrib, pd.Series):
            impact_regional_distrib = pd.Series(impact_regional_distrib, index=_regions)
            regional_distrib = pd.Series(0.0, index=_regions)
            regional_distrib.loc[impact_regional_distrib.index] = (
                impact_regional_distrib
            )
        else:
            regional_distrib = pd.Series(0.0, index=_regions)
            try:
                regional_distrib.loc[impact_regional_distrib.index] = (
                    impact_regional_distrib
                )
            except KeyError:
                regional_distrib.loc[_regions] = impact_regional_distrib.values

        assert isinstance(_sectors, pd.Index)
        if impact_sectoral_distrib is None:
            sectoral_distrib = pd.Series(1.0 / len(_sectors), index=_sectors)
        elif (
            isinstance(impact_sectoral_distrib, str)
            and impact_sectoral_distrib == "gdp"
        ):
            gva = cls.gva_df.loc[(_regions, _sectors)]
            sectoral_distrib = gva.groupby("region").transform(lambda x: x / sum(x))
        elif not isinstance(impact_sectoral_distrib, pd.Series):
            impact_sectoral_distrib = pd.Series(impact_sectoral_distrib, index=_sectors)
            sectoral_distrib = pd.Series(0.0, index=_sectors)
            sectoral_distrib.loc[impact_sectoral_distrib.index] = (
                impact_sectoral_distrib
            )
        else:
            sectoral_distrib = pd.Series(0.0, index=_sectors)
            try:
                sectoral_distrib.loc[impact_sectoral_distrib.index] = (
                    impact_sectoral_distrib
                )
            except KeyError:
                sectoral_distrib.loc[_sectors] = impact_sectoral_distrib.values

        logger.debug(f"{sectoral_distrib}")
        logger.debug(f"{regional_distrib}")
        if isinstance(sectoral_distrib.index, pd.MultiIndex):
            industries_distrib = sectoral_distrib * regional_distrib
        else:
            industries_distrib = pd.Series(
                np.outer(regional_distrib.values, sectoral_distrib.values).flatten(),  # type: ignore
                index=pd.MultiIndex.from_product(
                    [regional_distrib.index, sectoral_distrib.index]
                ),
            )

        impact_vec *= industries_distrib

        return cls.from_series(
            impact=impact_vec,
            occurrence=occurrence,
            duration=duration,
            name=name,
            **kwargs,
        )

    @property
    def impact_df(self) -> pd.Series:
        r"""A pandas Series with all possible industries as index, holding the impact vector of the event. The impact is defined for each sectors in each region."""
        return self._impact_df

    @impact_df.setter
    def impact_df(self, value: pd.Series):
        self._impact_df = pd.Series(
            0,
            dtype="float64",
            index=pd.MultiIndex.from_product(
                [self.possible_regions, self.possible_sectors],
                names=["region", "sector"],
            ),
        )
        self._impact_df[value.index] = value
        logger.debug("Sorting impact Series")
        self._impact_df.sort_index(inplace=True)
        self.aff_industries = self.impact_df.loc[self.impact_df > 0].index  # type: ignore
        self.impact_industries_distrib = self.impact_df.transform(lambda x: x / sum(x))
        self.total_impact = self.impact_df.sum()
        self.impact_vector = self.impact_df.values

    @property
    def aff_industries(self) -> pd.MultiIndex:
        r"""The industries affected by the event."""

        return self._aff_industries

    @aff_industries.setter
    def aff_industries(self, index: pd.MultiIndex):
        if not isinstance(index, pd.MultiIndex):
            raise ValueError(
                "Given impact vector does not have a (region,sector) MultiIndex"
            )
        if index.names != ["region", "sector"]:
            raise ValueError("MultiIndex must have levels named 'region' and 'sector'")

        regions = index.get_level_values("region").unique()
        sectors = index.get_level_values("sector").unique()
        if not set(regions).issubset(self.possible_regions):
            raise ValueError(
                f"Regions {set(regions) - set(self.possible_regions)} are not in the possible regions list"
            )
        if not set(sectors).issubset(self.possible_sectors):
            raise ValueError(
                f"Sectors {set(sectors) - set(self.possible_sectors)} are not in the possible sectors list"
            )

        self.aff_regions = regions
        self.aff_sectors = sectors
        logger.debug(
            f"Setting _aff_industries. There are {np.size(index)} affected industries"
        )
        self._aff_industries = index
        self._aff_industries_idx = np.array(
            [
                np.size(self.possible_sectors) * ri + si
                for ri in self._aff_regions_idx
                for si in self._aff_sectors_idx
            ]
        )

    @property
    def occurrence(self) -> int:
        r"""The temporal unit of occurrence of the event."""

        return self._occur

    @occurrence.setter
    def occurrence(self, value: int):
        if not 0 < value <= self.temporal_unit_range:
            raise ValueError(
                "Occurrence of event is not in the range of simulation steps (cannot be 0) : {} not in ]0-{}]".format(
                    value, self.temporal_unit_range
                )
            )
        else:
            logger.debug(f"Setting occurence to {value}")
            self._occur = value

    @property
    def duration(self) -> int:
        r"""The duration of the event."""

        return self._duration

    @duration.setter
    def duration(self, value: int):
        if not 0 <= self.occurrence + value <= self.temporal_unit_range:
            raise ValueError(
                "Occurrence + duration of event is not in the range of simulation steps : {} + {} not in [0-{}]".format(
                    self.occurrence, value, self.temporal_unit_range
                )
            )
        else:
            logger.debug(f"Setting duration to {value}")
            self._duration = value

    @property
    def aff_regions(self) -> pd.Index:
        r"""The array of regions affected by the event"""

        return self._aff_regions

    @property
    def aff_regions_idx(self) -> npt.NDArray:
        r"""The array of lexicographically ordered affected region indexes"""

        return self._aff_regions_idx

    @aff_regions.setter
    def aff_regions(self, value: npt.ArrayLike | str):
        if isinstance(value, str):
            value = [value]
        value = pd.Index(value)  # type: ignore
        impossible_regions = np.setdiff1d(value, self.possible_regions)
        if impossible_regions.size > 0:
            raise ValueError(
                "These regions are not in the model : {}".format(impossible_regions)
            )
        else:
            self._aff_regions = pd.Index(value, name="region")
            logger.debug(f"Setting _aff_regions to {self._aff_regions}")
            self._aff_regions_idx = np.searchsorted(self.possible_regions, value)

    @property
    def aff_sectors(self) -> pd.Index:
        r"""The array of affected sectors by the event"""

        return self._aff_sectors

    @property
    def aff_sectors_idx(self) -> npt.NDArray:
        r"""The array of lexicographically ordered affected sectors indexes"""

        return self._aff_sectors_idx

    @aff_sectors.setter
    def aff_sectors(self, value: npt.ArrayLike | str):
        if isinstance(value, str):
            value = [value]
        value = pd.Index(value, name="sector")  # type: ignore
        impossible_sectors = np.setdiff1d(value, self.possible_sectors)
        if impossible_sectors.size > 0:
            raise ValueError(
                "These sectors are not in the model : {}".format(impossible_sectors)
            )
        else:
            self._aff_sectors = pd.Index(value, name="sector")
            logger.debug(f"Setting _aff_sectors to {self._aff_sectors}")
            self._aff_sectors_idx = np.searchsorted(self.possible_sectors, value)

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
        self._impact_regional_distrib = self._impact_industries_distrib.groupby(
            "region",
            observed=False,
        ).sum()

    def __repr__(self):
        # TODO: find ways to represent long lists
        return f"""[WIP]
        Event(
              name = {self.name},
              occur = {self.occurrence},
              duration = {self.duration}
              aff_regions = {self.aff_regions.to_list()},
              aff_sectors = {self.aff_sectors.to_list()},
             )
        """


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
        recovery_time: int = 1,
        recovery_function: str = "linear",
        name: str | None = "Unnamed",
        occurrence: int = 1,
        duration: int = 1,
    ) -> None:
        raise NotImplementedError(
            "This type of Event suffers from a major bug and cannot be used at the moment."
        )
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
            self.impact_vector.copy()
        )  # np.zeros(shape=len(self.possible_sectors))
        self.prod_cap_delta_arbitrary = (
            self.impact_vector.copy()
        )  # type: ignore # np.zeros(shape=len(self.possible_sectors))
        self.recovery_time = recovery_time
        r"""The characteristic recovery duration after the event is over"""

        self.recovery_function = recovery_function

        logger.info("Initialized")

    @classmethod
    def _instantiate(
        cls,
        impact: pd.Series,
        *,
        recovery_time: int = 1,
        recovery_function: str = "linear",
        occurrence: int = 1,
        duration: int = 1,
        name: Optional[str] = None,
        **_,
    ):
        return cls(
            impact=impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            recovery_time=recovery_time,
            recovery_function=recovery_function,
        )

    @property
    def prod_cap_delta_arbitrary(self) -> npt.NDArray:
        r"""The optional array storing arbitrary (as in not related to productive_capital destroyed) production capacity loss"""
        return self._prod_cap_delta_arbitrary

    @prod_cap_delta_arbitrary.setter
    def prod_cap_delta_arbitrary(self, value: npt.NDArray):
        self._prod_cap_delta_arbitrary = value

    @property
    def recoverable(self) -> bool:
        r"""A boolean stating if an event can start recover"""
        return self._recoverable

    @recoverable.setter
    def recoverable(self, current_temporal_unit: int):
        reb = (self.occurrence + self.duration) <= current_temporal_unit
        if reb and not self.recoverable:
            logger.info(
                "Temporal_Unit : {} ~ Event named {} that occured at {} in {} has started recovering (arbitrary production capacity loss)".format(
                    current_temporal_unit,
                    self.name,
                    self.occurrence,
                    self.aff_regions.to_list(),
                )
            )
        self._recoverable = reb

    @property
    def recovery_function(self) -> Callable:
        r"""The recovery function used for recovery (`Callable`)"""
        return self._recovery_fun

    @recovery_function.setter
    def recovery_function(self, r_fun: str | Callable | None):
        if r_fun is None:
            r_fun = "instant"
        if self.recovery_time is None:
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
                    "recovery_time",
                ]
            ):
                raise ValueError(
                    "Recovery function has to have at least the following keyword arguments: {}".format(
                        [
                            "init_impact_stock",
                            "elapsed_temporal_unit",
                            "recovery_time",
                        ]
                    )
                )
            fun = r_fun

        else:
            raise ValueError("Given recovery function is not a str or callable")

        r_fun_partial = partial(
            fun,
            init_impact_stock=self._prod_cap_delta_arbitrary_0,
            recovery_time=self.recovery_time,
        )
        self._recovery_fun = r_fun_partial

    def recovery(self, current_temporal_unit: int):
        """Apply the recovery function to the capital destroyed for the current temporal unit.

        Parameters
        ----------
        current_temporal_unit : int
            The current temporal unit

        Raises
        ------
        RuntimeError
            Raised if no recovery function has been set.

        """

        elapsed = current_temporal_unit - (self.occurrence + self.duration)
        if elapsed < 0:
            raise RuntimeError("Trying to recover before event is over")
        if self.recovery_function is None:
            raise RuntimeError(
                "Trying to recover event while recovery function isn't set yet"
            )
        res = self.recovery_function(elapsed_temporal_unit=elapsed)
        precision = int(math.log10(self.model_monetary_factor)) + 1
        res = np.around(res, precision)
        if not np.any(res):
            self.over = True
        self._prod_cap_delta_arbitrary = res


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
        households_impact: Optional[Impact] = None,
        name: str | None = "Unnamed",
        occurrence: int = 1,
        duration: int = 1,
        event_monetary_factor: Optional[int] = None,
    ) -> None:
        if event_monetary_factor is None:
            logger.info(
                f"No event monetary factor given. Assuming it is the same as the model ({self.model_monetary_factor})"
            )
            self.event_monetary_factor = self.model_monetary_factor
            r"""The monetary factor for the impact of the event (e.g. 10**6, 10**3, ...)"""

        else:
            self.event_monetary_factor = event_monetary_factor
            if self.event_monetary_factor != self.model_monetary_factor:
                logger.warning(
                    f"Event monetary factor ({self.event_monetary_factor}) differs from model monetary factor ({self.model_monetary_factor}). Be careful to define your impact with the correct unit (ie in event monetary factor)."
                )

        if (impact < LOW_DEMAND_THRESH / self.event_monetary_factor).all():
            raise ValueError(
                "Impact is too small to be considered by the model. Check you units perhaps ?"
            )

        super().__init__(
            impact=impact,
            name=name,
            occurrence=occurrence,
            duration=duration,
        )

        # The only thing we have to do is affecting/computing the regional_sectoral_productive_capital_destroyed
        self.impact_df = self.impact_df * (
            self.event_monetary_factor / self.model_monetary_factor
        )
        self.total_productive_capital_destroyed = self.total_impact
        logger.info(
            f"Total impact on productive capital is {self.total_productive_capital_destroyed} (in model unit)"
        )
        self.remaining_productive_capital_destroyed = (
            self.total_productive_capital_destroyed
        )
        self._regional_sectoral_productive_capital_destroyed_0 = (
            self.impact_vector.copy()
        )
        self.regional_sectoral_productive_capital_destroyed = self.impact_vector.copy()  # type: ignore
        self.households_impact_df: pd.Series = pd.Series(
            0,
            dtype="float64",
            index=pd.MultiIndex.from_product(
                [self.possible_regions, self.possible_final_demand_cat],
                names=["region", "final_demand_cat"],
            ),
        )
        r"""A pandas Series with all possible (regions, final_demand_cat) as index, holding the households impacts vector of the event. The impact is defined for each region and each final demand category."""

        if households_impact:
            try:
                rebuilding_demand_idx = self.possible_final_demand_cat[
                    self.possible_final_demand_cat.str.match(
                        REBUILDING_FINALDEMAND_CAT_REGEX
                    )
                ]  # .values[0]
                if len(rebuilding_demand_idx) > 1:
                    raise ValueError(
                        f"The rebuilding demand index ({rebuilding_demand_idx}) contains multiple values which is a problem. Contact the dev to update the matching regexp."
                    )

            except IndexError:
                logger.warning(
                    f"No final demand category matched common rebuilding final demand category, hence we will put it in the first available ({self.possible_final_demand_cat[0]})."
                )
                rebuilding_demand_idx = self.possible_final_demand_cat[0]
            if isinstance(households_impact, pd.Series):
                logger.debug("Given household impact is a pandas Series")
                self.households_impact_df.loc[
                    households_impact.index, rebuilding_demand_idx
                ] = households_impact  # type: ignore
            elif isinstance(households_impact, dict):
                logger.debug("Given household impact is a dict")
                households_impact = pd.Series(households_impact, dtype="float64")
                self.households_impact_df.loc[
                    households_impact.index, rebuilding_demand_idx
                ] = households_impact  # type: ignore
            elif isinstance(households_impact, pd.DataFrame):
                logger.debug("Given household impact is a dataframe")
                households_impact = households_impact.squeeze()
                if not isinstance(households_impact, pd.Series):
                    raise ValueError(
                        "The given households_impact DataFrame is not a Series after being squeezed"
                    )
                self.households_impact_df.loc[
                    households_impact.index, rebuilding_demand_idx
                ] = households_impact  # type: ignore
            elif isinstance(households_impact, (list, np.ndarray)):
                if np.size(households_impact) != np.size(self.aff_regions):
                    raise ValueError(
                        f"Length mismatch between households_impact ({np.size(households_impact)} and affected regions ({np.size(self.aff_regions)}))"
                    )
                else:
                    self.households_impact_df.loc[
                        self.aff_regions, rebuilding_demand_idx
                    ] = households_impact  # type: ignore
            elif isinstance(households_impact, (float, int)):
                if self.impact_regional_distrib is not None:
                    logger.warning(
                        f"households impact ({households_impact}) given as scalar, distributing among region following `impact_regional_distrib` ({self.impact_regional_distrib}) to {self.aff_regions}"
                    )
                    logger.debug(f"{rebuilding_demand_idx}")
                    self.households_impact_df.loc[:, rebuilding_demand_idx] = (
                        households_impact * self.impact_regional_distrib
                    ).to_numpy()  # type: ignore
            self.households_impact_df *= (
                self.event_monetary_factor / self.model_monetary_factor
            )
        logger.info(
            f"Total impact on households is {self.households_impact_df.sum()} (in model unit)"
        )

    @property
    def regional_sectoral_productive_capital_destroyed(self) -> npt.NDArray:
        r"""The (optional) array of productive_capital destroyed per industry (ie region x sector)"""
        return self._regional_sectoral_productive_capital_destroyed

    @regional_sectoral_productive_capital_destroyed.setter
    def regional_sectoral_productive_capital_destroyed(self, value: npt.ArrayLike):
        value = np.array(value)
        logger.debug(
            f"Changing regional_sectoral_productive_capital_destroyed with {value}\n Sum is {value.sum()}"
        )
        if value.shape != self.x_shape:
            raise ValueError(
                "Incorrect shape give for regional_sectoral_productive_capital_destroyed: {} given and {} expected".format(
                    value.shape, self.x_shape
                )
            )
        self._regional_sectoral_productive_capital_destroyed = value


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
        households_impact: Impact | None = None,
        name: str | None = "Unnamed",
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
        self._rebuildable = False
        self.rebuild_tau = rebuild_tau
        self.rebuilding_sectors = rebuilding_sectors
        self.rebuilding_factor = rebuilding_factor

        industrial_rebuilding_demand = np.zeros(shape=self.z_shape)
        tmp = np.zeros(self.z_shape, dtype="float")

        mask = np.ix_(
            np.union1d(
                self._rebuilding_industries_RoW_idx, self._rebuilding_industries_idx
            ),
            self._aff_industries_idx,
        )
        industrial_rebuilding_demand = np.outer(
            self.rebuilding_sectors_shares,
            self.regional_sectoral_productive_capital_destroyed,
        )
        industrial_rebuilding_demand = industrial_rebuilding_demand * rebuilding_factor
        tmp[mask] = self.Z_distrib[mask] * industrial_rebuilding_demand[mask]

        precision = int(math.log10(self.model_monetary_factor)) + 1
        np.around(tmp, precision, tmp)

        self.rebuilding_demand_indus = tmp

        households_rebuilding_demand = np.zeros(shape=self.y_shape)
        tmp = np.zeros(self.y_shape, dtype="float")
        mask = np.ix_(  # type: ignore
            np.union1d(  # type: ignore
                self._rebuilding_industries_RoW_idx, self._rebuilding_industries_idx
            ),
            list(range(self.y_shape[1])),
        )

        households_rebuilding_demand = np.outer(
            self.rebuilding_sectors_shares, self.households_impact_df.to_numpy()
        )
        households_rebuilding_demand = households_rebuilding_demand * rebuilding_factor
        tmp[mask] = self.Y_distrib[mask] * households_rebuilding_demand[mask]
        np.around(tmp, precision, tmp)
        self.rebuilding_demand_house = tmp
        logger.debug(
            f"house rebuilding demand is: {pd.DataFrame(self.rebuilding_demand_house, index=pd.MultiIndex.from_product([self.possible_regions,self.possible_sectors]))}"
        )

        self.event_dict["rebuilding_sectors"] = {
            sec: share
            for sec, share in zip(
                self.rebuilding_sectors, self.rebuilding_sectors_shares
            )
        }

    @classmethod
    def _instantiate(
        cls,
        impact: pd.Series,
        *,
        households_impact: Optional[Impact] = None,
        occurrence: int = 1,
        duration: int = 1,
        name: str | None = None,
        event_monetary_factor: Optional[int] = None,
        rebuild_tau: int,
        rebuilding_sectors: dict[str, float] | pd.Series,
        rebuilding_factor: float = 1.0,
        **_,
    ):
        return cls(
            impact=impact,
            households_impact=households_impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            rebuild_tau=rebuild_tau,
            rebuilding_sectors=rebuilding_sectors,
            rebuilding_factor=rebuilding_factor,
        )

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
    def rebuilding_sectors(self) -> pd.Index:
        r"""The (optional) array of rebuilding sectors"""
        return self._rebuilding_sectors

    @property
    def rebuilding_sectors_idx(self) -> npt.NDArray:
        r"""The (optional) array of indexes of the rebuilding sectors (in lexicographic order)"""
        return self._rebuilding_sectors_idx

    @property
    def rebuilding_sectors_shares(self) -> npt.NDArray:
        r"""The array specifying how rebuilding demand is distributed along the rebuilding sectors"""
        return self._rebuilding_sectors_shares

    @rebuilding_sectors.setter
    def rebuilding_sectors(self, value: dict[str, float] | pd.Series):
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
        impossible_sectors = np.setdiff1d(reb_sectors.index, self.possible_sectors)
        if impossible_sectors.size > 0:
            raise ValueError(
                "These sectors are not in the model : {}".format(impossible_sectors)
            )
        else:
            self._rebuilding_sectors = reb_sectors.index
            self._rebuilding_sectors_idx = np.searchsorted(
                self.possible_sectors, reb_sectors.index
            )

            self._rebuilding_sectors_shares = np.zeros(self.x_shape)
            self._rebuilding_industries_idx = np.array(
                [
                    np.size(self.possible_sectors) * ri + si
                    for ri in self._aff_regions_idx
                    for si in self._rebuilding_sectors_idx
                ],
                dtype="int64",
            )
            self._rebuilding_industries_RoW_idx = np.array(
                [
                    np.size(self.possible_sectors) * ri + si
                    for ri in range(np.size(self.possible_regions))
                    if ri not in self._aff_regions_idx
                    for si in self._rebuilding_sectors_idx
                ],
                dtype="int64",
            )
            self._rebuilding_sectors_shares[self._rebuilding_industries_idx] = np.tile(
                np.array(reb_sectors.values), np.size(self.aff_regions)
            )
            if self._rebuilding_industries_RoW_idx.size != 0:
                self._rebuilding_sectors_shares[self._rebuilding_industries_RoW_idx] = (
                    np.tile(
                        np.array(reb_sectors.values),
                        (np.size(self.possible_regions) - np.size(self.aff_regions)),
                    )
                )

    @property
    def rebuilding_demand_house(self) -> npt.NDArray:
        r"""The optional array of rebuilding demand from households"""
        return self._rebuilding_demand_house

    @rebuilding_demand_house.setter
    def rebuilding_demand_house(self, value: npt.ArrayLike):
        value = np.array(value)
        if value.shape != self.y_shape:
            raise ValueError(
                "Incorrect shape give for rebuilding_demand_house: {} given and {} expected".format(
                    value.shape, self.y_shape
                )
            )
        value[value < LOW_DEMAND_THRESH / self.model_monetary_factor] = 0.0
        self._rebuilding_demand_house = value

    @property
    def rebuilding_demand_indus(self) -> npt.NDArray:
        r"""The optional array of rebuilding demand from industries (to rebuild their productive_capital)"""
        return self._rebuilding_demand_indus

    @rebuilding_demand_indus.setter
    def rebuilding_demand_indus(self, value: npt.ArrayLike):
        value = np.array(value)
        if value.shape != self.z_shape:
            raise ValueError(
                "Incorrect shape give for rebuilding_demand_indus: {} given and {} expected".format(
                    value.shape, self.z_shape
                )
            )
        value[value < LOW_DEMAND_THRESH / self.model_monetary_factor] = 0.0
        self._rebuilding_demand_indus = value
        # Also update productive_capital destroyed
        self.regional_sectoral_productive_capital_destroyed = value.sum(axis=0) * (
            1 / self.rebuilding_factor
        )

    @property
    def rebuildable(self) -> bool:
        r"""A boolean stating if an event can start rebuild"""
        return self._rebuildable

    @rebuildable.setter
    def rebuildable(self, current_temporal_unit: int):
        reb = (self.occurrence + self.duration) <= current_temporal_unit
        if reb and not self.rebuildable:
            logger.info(
                "Temporal_Unit : {} ~ Event named {} that occured at {} in {} for {} damages has started rebuilding".format(
                    current_temporal_unit,
                    self.name,
                    self.occurrence,
                    self.aff_regions.to_list(),
                    self.total_productive_capital_destroyed,
                )
            )
        self._rebuildable = reb


class EventKapitalRecover(EventKapitalDestroyed):
    r"""EventKapitalRecover holds a :py:class:`EventKapitalDestroyed` event where the destroyed capital does not create a reconstruction demand.

    This subclass requires and enables new arguments to pass to the constructor:

    * A characteristic time for the recovery (``recovery_time``)
    * Optionally a ``recovery_function`` (linear by default).

    .. seealso::
       Tutorial :ref:`boario-events`
    """

    def __init__(
        self,
        *,
        impact: pd.Series,
        recovery_time: int,
        recovery_function: str = "linear",
        households_impact: Optional[Impact] = None,
        name: str | None = "Unnamed",
        occurrence=1,
        duration=1,
        event_monetary_factor=None,
    ) -> None:
        super().__init__(
            impact=impact,
            households_impact=households_impact,
            name=name,
            occurrence=occurrence,
            duration=duration,
            event_monetary_factor=event_monetary_factor,
        )
        self.recovery_time = recovery_time
        self.recovery_function = recovery_function
        self._recoverable = False

    @classmethod
    def _instantiate(
        cls,
        impact: pd.Series,
        *,
        households_impact: Optional[Impact] = None,
        occurrence: int = 1,
        duration: int = 1,
        name: str | None = None,
        event_monetary_factor: Optional[int] = None,
        recovery_time: int,
        recovery_function: str = "linear",
        **_,
    ):
        return cls(
            impact=impact,
            households_impact=households_impact,
            occurrence=occurrence,
            duration=duration,
            name=name,
            event_monetary_factor=event_monetary_factor,
            recovery_time=recovery_time,
            recovery_function=recovery_function,
        )

    @property
    def recoverable(self) -> bool:
        r"""A boolean stating if an event can start recover"""
        return self._recoverable

    @recoverable.setter
    def recoverable(self, current_temporal_unit: int):
        reb = (self.occurrence + self.duration) <= current_temporal_unit
        if reb and not self.recoverable:
            logger.info(
                "Temporal_Unit : {} ~ Event named {} that occured at {} in {} for {} damages has started recovering (no rebuilding demand)".format(
                    current_temporal_unit,
                    self.name,
                    self.occurrence,
                    self.aff_regions.to_list(),
                    self.total_productive_capital_destroyed,
                )
            )
        self._recoverable = reb

    @property
    def recovery_function(self) -> Callable:
        r"""The recovery function used for recovery (`Callable`)"""
        return self._recovery_fun

    @recovery_function.setter
    def recovery_function(self, r_fun: str | Callable | None):
        if r_fun is None:
            r_fun = "linear"
        if self.recovery_time is None:
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
                    "recovery_time",
                ]
            ):
                raise ValueError(
                    "Recovery function has to have at least the following keyword arguments: {}".format(
                        [
                            "init_impact_stock",
                            "elapsed_temporal_unit",
                            "recovery_time",
                        ]
                    )
                )
            fun = r_fun

        else:
            raise ValueError("Given recovery function is not a str or callable")

        r_fun_partial = partial(
            fun,
            init_impact_stock=self._regional_sectoral_productive_capital_destroyed_0,
            recovery_time=self.recovery_time,
        )
        self._recovery_fun = r_fun_partial

    def recovery(self, current_temporal_unit: int):
        elapsed = current_temporal_unit - (self.occurrence + self.duration)
        if elapsed < 0:
            raise RuntimeError("Trying to recover before event is over")
        if self.recovery_function is None:
            raise RuntimeError(
                "Trying to recover event while recovery function isn't set yet"
            )
        res = self.recovery_function(elapsed_temporal_unit=elapsed)
        precision = int(math.log10(self.model_monetary_factor)) + 1
        res = np.around(res, precision)
        res[res < 0] = 0.0
        if not np.any(res):
            self.over = True
        self.regional_sectoral_productive_capital_destroyed = res
