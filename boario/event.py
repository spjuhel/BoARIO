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
import abc
from typing import Callable, Optional, Union, List, Tuple
from numpy.typing import ArrayLike
import numpy as np
import pandas as pd
from boario import logger
import math
import inspect
from functools import partial

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

Impact = Union[int, float, list, dict, np.ndarray, pd.DataFrame, pd.Series]
IndustriesList = Union[List[Tuple[str, str]], pd.MultiIndex, np.ndarray]
SectorsList = RegionsList = Union[List[str], pd.Index, np.ndarray]


def linear_recovery(
    elapsed_temporal_unit: int, init_kapital_destroyed: np.ndarray, recovery_time: int
):
    """Linear Kapital recovery function

    Kapital is entirely recovered when `recovery_time` has passed since event
    started recovering

    Parameters
    ----------
    init_kapital_destroyed : float
        Initial kapital destroyed
    elapsed_temporal_unit : int
        Elapsed time since event started recovering
    recovery_time : int
        Total time it takes the event to fully recover

    Examples
    --------
    FIXME: Add docs.

    """

    return init_kapital_destroyed * (1 - (elapsed_temporal_unit / recovery_time))


def convexe_recovery(
    elapsed_temporal_unit: int, init_kapital_destroyed: np.ndarray, recovery_time: int
):
    """Convexe Kapital recovery function

    Kapital is recovered with characteristic time `recovery_time`. (This doesn't mean Kapital is fully recovered after this time !)
    This function models a recovery similar as the one happening in the rebuilding case, for the same characteristic time.

    Parameters
    ----------
    init_kapital_destroyed : float
        Initial kapital destroyed
    elapsed_temporal_unit : int
        Elapsed time since event started recovering
    recovery_time : int
        Total time it takes the event to fully recover

    """
    return init_kapital_destroyed * (1 - (1 / recovery_time)) ** elapsed_temporal_unit


def convexe_recovery_scaled(
    elapsed_temporal_unit: int,
    init_kapital_destroyed: np.ndarray,
    recovery_time: int,
    scaling_factor: float = 4,
):
    """Convexe Kapital recovery function (scaled to match other recovery duration)

    Kapital is mostly recovered (>95% by default for most cases) when `recovery_time` has passed since event
    started recovering.

    Parameters
    ----------
    init_kapital_destroyed : float
        Initial kapital destroyed
    elapsed_temporal_unit : int
        Elapsed time since event started recovering
    recovery_time : int
        Total time it takes the event to fully recover
    scaling_factor: float
        Used to scale the exponent in the function so that kapital is mostly rebuilt after `recovery_time`.
    A value of 4 insure >95% of kapital is recovered for a reasonable range of `recovery_time` values.

    """
    return init_kapital_destroyed * (1 - (1 / recovery_time)) ** (
        scaling_factor * elapsed_temporal_unit
    )


def concave_recovery(
    elapsed_temporal_unit: int,
    init_kapital_destroyed: np.ndarray,
    recovery_time: int,
    steep_factor: float = 0.000001,
    half_recovery_time: Optional[int] = None,
):
    """Concave (s-shaped) Kapital recovery function

    Kapital is mostly (>95% in most cases) recovered when `recovery_time` has passed since event started recovering.

    Parameters
    ----------
    init_kapital_destroyed : float
        Initial kapital destroyed
    elapsed_temporal_unit : int
        Elapsed time since event started recovering
    recovery_time : int
        Total time it takes the event to fully recover
    steep_factor: float
        This coefficient governs the slope of the central part of the s-shape, smaller values lead to a steeper slope. As such it also affect the percentage of kapital rebuilt
    after `recovery_time` has elapsed. A value of 0.000001 should insure 95% of the kapital is rebuild for a reasonable range of recovery duration.
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
    return (init_kapital_destroyed * recovery_time) / (
        recovery_time + steep_factor * (elapsed_temporal_unit**exponent)
    )


class Event(metaclass=abc.ABCMeta):
    """Create an event shocking the model from a dictionary.

    Events store all information about a unique shock during simulation such as
    time of occurrence, duration, type of shock, amount of damages. Computation
    of recovery or initialy requested rebuilding demand is also done in this
    class.

    Parameters
    ----------

    event : dict
        A dictionary holding the necessary information to define an event.

    Examples
    --------
    FIXME: Add docs.

    """

    # Class Attributes
    __required_class_attributes = [
        "possible_sectors",
        "possible_regions",
        "temporal_unit_range",
        "z_shape",
        "y_shape",
        "x_shape",
        "regions_idx",
        "sectors_idx",
        "monetary_factor",
        "sectors_gva_shares",
        "Z_distrib",
        "mrio_name",
    ]
    possible_sectors: SectorsList = pd.Index([])
    """List of sectors present in the MRIO used by the model"""
    possible_regions: RegionsList = pd.Index([])
    """List of regions present in the MRIO used by the model"""
    temporal_unit_range: int = 0
    """Maximum temporal unit simulated"""
    z_shape: tuple[int, int] = (0, 0)
    """Shape of the Z (intermediate consumption) matrix in the model"""
    y_shape: tuple[int, int] = (0, 0)
    """Shape of the Y (final demand) matrix in the model"""
    x_shape: tuple[int, int] = (0, 0)
    """Shape of the x (production) vector in the model"""
    regions_idx: np.ndarray = np.array([])
    """lexicographic region indexes"""
    sectors_idx: np.ndarray = np.array([])
    """lexicographic sector indexes"""
    monetary_factor: int = 0
    """Amount of unitary currency used in the MRIO (e.g. 1000000 if in € millions)"""
    sectors_gva_shares: np.ndarray = np.array([])
    """Fraction of total (regional) GVA for each sectors"""
    Z_distrib: np.ndarray = np.array([])
    """Normalized intermediate consumption matrix"""
    mrio_name: str = ""
    """MRIO identification"""

    def __init__(
        self,
        impact: Impact,
        aff_regions: Optional[RegionsList] = None,
        aff_sectors: Optional[SectorsList] = None,
        aff_industries: Optional[IndustriesList] = None,
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="custom",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
    ) -> None:
        logger.debug("Initializing new Event")
        logger.debug("Checking required Class attributes are defined")
        for v in Event.__required_class_attributes:
            if Event.__dict__[v] is None:
                raise AttributeError(
                    "Required Event Class attribute {} is not set yet so instantiating an Event isn't possible".format(
                        v
                    )
                )

        self.name = name
        self.occurrence = occurrence
        self.duration = duration
        self.impact_df = pd.Series(
            0,
            dtype="float64",
            index=pd.MultiIndex.from_product(
                [self.possible_regions, self.possible_sectors],
                names=["region", "sector"],
            ),
        )
        ################## DATAFRAME INIT #################
        # CASE VECTOR 1 (everything is there and regrouped) (only df creation)
        if isinstance(impact, pd.Series):
            logger.debug("Given impact is a pandas Series")
            self.impact_df.loc[impact.index] = impact
            if self.name == "Unnamed" and not impact.name is None:
                self.name = impact.name
        elif isinstance(impact, dict):
            logger.debug("Given impact is a dict, converting it to pandas Series")
            impact = pd.Series(impact)
            self.impact_df.loc[impact.index] = impact
        elif isinstance(impact, pd.DataFrame):
            logger.debug("Given Impact is a pandas DataFrame, squeezing it to a Series")
            impact = impact.squeeze()
            if not isinstance(impact, pd.Series):
                raise ValueError(
                    "The given impact DataFrame is not a Series after being squeezed"
                )
            self.impact_df.loc[impact.index] = impact
        # CASE VECTOR 2 (everything is there but not regrouped) AND CASE SCALAR (Only df creation)
        elif (
            isinstance(impact, (int, float, list, np.ndarray))
            and aff_industries is not None
        ):
            logger.debug(
                f"Given Impact is a {type(impact)} and list of impacted industries given. Proceeding."
            )
            self.impact_df.loc[aff_industries] = impact
        elif (
            isinstance(impact, (int, float, list, np.ndarray))
            and aff_regions is not None
            and aff_sectors is not None
        ):
            logger.debug(
                f"Given Impact is a {type(impact)} and lists of impacted regions and sectors given. Proceeding."
            )
            if isinstance(aff_regions, str):
                aff_regions = [aff_regions]
            if isinstance(aff_sectors, str):
                aff_sectors = [aff_sectors]

            self.impact_df.loc[
                pd.MultiIndex.from_product([aff_regions, aff_sectors])
            ] = impact
        else:
            raise ValueError("Invalid input format. Could not initiate pandas Series.")

        # Check for <0 values and remove 0.
        if (self.impact_df < 0).any():
            logger.warning(
                "Found negative values in impact vector. This should raise concern"
            )

        # SORT DF
        # at this point impact_df is built, and can be sorted. Note that if impact was a scalar, impact_df contains copies of this scalar.
        logger.debug("Sorting Impact Series")
        self.impact_df = self.impact_df.sort_index()

        # Init self.impact_sectoral_distrib_type,
        self.impact_sectoral_distrib_type = impact_sectoral_distrib_type
        #################################################

        # SET INDEXES ATTR
        # note that the following also sets aff_regions and aff_sectors
        assert isinstance(self.impact_df.index, pd.MultiIndex)
        # Only look for industries where impact is greater than 0
        self.aff_industries = self.impact_df.loc[self.impact_df > 0].index

        ###### SCALAR DISTRIBUTION ######################
        # if impact_industries_distrib is given, set it. We assume impact is scalar !
        # CASE SCALAR + INDUS DISTRIB
        if impact_industries_distrib is not None and not isinstance(
            impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)
        ):
            logger.debug("Impact is Scalar and impact_industries_distrib was given")
            self.impact_industries_distrib = np.array(impact_industries_distrib)
            self.impact_df.loc[self.aff_industries] = (
                self.impact_df.loc[self.aff_industries] * self.impact_industries_distrib
            )
        # if impact_reg_dis and sec_dis are give, deduce the rest. We also assume impact is scalar !
        # CASE SCALAR + REGION and SECTOR DISTRIB
        elif (
            impact_regional_distrib is not None
            and impact_sectoral_distrib is not None
            and not isinstance(
                impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)
            )
        ):
            logger.debug(
                "Impact is Scalar and impact_regional_distrib and impact_sectoral_distrib were given"
            )
            if len(impact_regional_distrib) != len(self.aff_regions) or len(
                impact_sectoral_distrib
            ) != len(self.aff_sectors):
                raise ValueError(
                    "Lengths of `impact_regional_distrib` and/or `impact_sectoral_distrib` are incompatible with `aff_regions` and/or `aff_sectors`."
                )
            else:
                self.impact_regional_distrib = np.array(impact_regional_distrib)
                self.impact_sectoral_distrib = np.array(impact_sectoral_distrib)
                self.impact_industries_distrib = (
                    self.impact_regional_distrib[:, np.newaxis]
                    * self.impact_sectoral_distrib
                ).flatten()
                self.impact_df.loc[self.aff_industries] = (
                    self.impact_df.loc[self.aff_industries]
                    * self.impact_industries_distrib
                )
        # CASE SCALAR + 'gdp' distrib
        elif (

            impact_regional_distrib is not None
            and impact_sectoral_distrib_type is not None
            and impact_sectoral_distrib_type == "gdp"
            and not isinstance(
                impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)
            )
        ):
            logger.debug("Impact is Scalar and impact_sectoral_distrib_type is 'gdp'")

            self.impact_regional_distrib = np.array(impact_regional_distrib)
            
            shares = self.sectors_gva_shares.reshape(
                (len(self.possible_regions), len(self.possible_sectors))
            )
            self.impact_sectoral_distrib = (
                shares[self._aff_regions_idx][:, self._aff_sectors_idx]
                / shares[self._aff_regions_idx][:, self._aff_sectors_idx].sum(axis=1)[
                    :, np.newaxis
                ]
            )
            self.impact_industries_distrib = (
                self.impact_regional_distrib[:, np.newaxis]
                * self.impact_sectoral_distrib
            ).flatten()
            self.impact_df.loc[self.aff_industries] = (
                self.impact_df.loc[self.aff_industries] * self.impact_industries_distrib
            )
            self.impact_sectoral_distrib_type = "gdp"
        # CASE SCALAR + NO DISTRIB + list of industries
        # if neither was given, we use default values. Again impact should be scalar here !
        elif isinstance(aff_industries, (list, np.ndarray)) and not isinstance(
            impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)
        ):
            logger.debug(
                "Impact is Scalar and no distribution was given but a list of affected industries was given"
            )
            self._default_distribute_impact_from_industries_list()
            self.impact_sectoral_distrib_type = "default (shared equally between affected regions and then affected sectors)"
        # CASE SCALAR + NO DISTRIB + list of region + list of sectors
        elif (
            aff_regions is not None
            and aff_sectors is not None
            and not isinstance(
                impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)
            )
        ):
            logger.debug(
                "Impact is Scalar and no distribution was given but lists of regions and sectors affected were given"
            )
            self._default_distribute_impact_from_industries_list()
            self.impact_sectoral_distrib_type = "default (shared equally between affected regions and then affected sectors)"
        elif not isinstance(impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)):
            raise ValueError(f"Invalid input format: Could not compute impact")

        self._finish_init()
        ##################################################

        self.happened = False
        self.over = False
        self.event_dict = {
            "name": str(self.name),
            "occurrence": self.occurrence,
            "duration": self.duration,
            "aff_regions": list(self.aff_regions),
            "aff_sectors": list(self.aff_sectors),
            "impact": self.total_impact,
            "impact_industries_distrib": list(self.impact_industries_distrib),
            "impact_regional_distrib": list(self.impact_regional_distrib),
            "impact_sectoral_distrib_type": self.impact_sectoral_distrib_type,
        }
        self.event_dict["globals_vars"] = {
            "possible_sectors": list(self.possible_sectors),
            "possible_regions": list(self.possible_regions),
            "temporal_unit_range": self.temporal_unit_range,
            "z_shape": self.z_shape,
            "y_shape": self.y_shape,
            "x_shape": self.x_shape,
            "monetary_factor": self.monetary_factor,
            "mrio_used": self.mrio_name,
        }

    def _default_distribute_impact_from_industries_list(self):
        # at this point, impact should still be scalar.
        logger.debug("Using default impact distribution to industries")
        self.impact_regional_distrib = np.full(
            len(self.aff_regions), 1 / len(self.aff_regions)
        )
        self.impact_df.loc[self.aff_industries] = (
            self.impact_df.loc[self.aff_industries] * 1 / len(self.aff_regions)
        )
        impact_sec_vec = np.array(
            [
                1 / len(self.impact_df.loc[(reg, slice(None))])
                for _ in self.aff_sectors
                for reg in self.aff_regions
            ]
        )
        self.impact_df.loc[self.aff_industries] = (
            self.impact_df.loc[self.aff_industries] * impact_sec_vec
        )

    def _finish_init(self):
        logger.debug("Finishing Event init")
        self.impact_vector = self.impact_df.to_numpy()
        self.total_impact = self.impact_vector.sum()
        self.impact_industries_distrib = (
            self.impact_vector[self.impact_vector > 0] / self.total_impact
        )
        self.impact_regional_distrib = (
            self.impact_df.loc[self.aff_industries].groupby("region").sum().values
            / self.total_impact
        )

    @property
    def aff_industries(self) -> pd.MultiIndex:
        """A pandas MultiIndex with the regions affected as first level, and sectors affected as second level"""
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
                f"Regions {set(regions)-set(self.possible_regions)} are not in the possible regions list"
            )
        if not set(sectors).issubset(self.possible_sectors):
            raise ValueError(
                f"Sectors {set(sectors)-set(self.possible_sectors)} are not in the possible sectors list"
            )

        self.aff_regions = regions
        self.aff_sectors = sectors
        logger.debug(
            f"Setting _aff_industries. There are {len(index)} affected industries"
        )
        self._aff_industries = index
        self._aff_industries_idx = np.array(
            [
                len(self.possible_sectors) * ri + si
                for ri in self._aff_regions_idx
                for si in self._aff_sectors_idx
            ]
        )

    @property
    def occurrence(self) -> int:
        """The temporal unit of occurrence of the event."""
        return self._occur

    @occurrence.setter
    def occurrence(self, value: int):
        if not 0 <= value <= self.temporal_unit_range:
            raise ValueError(
                "Occurrence of event is not in the range of simulation steps : {} not in [0-{}]".format(
                    value, self.temporal_unit_range
                )
            )
        else:
            logger.debug(f"Setting occurence to {value}")
            self._occur = value

    @property
    def duration(self) -> int:
        """The duration of the event."""
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
        """The array of regions affected by the event"""
        return self._aff_regions

    @property
    def aff_regions_idx(self) -> np.ndarray:
        """The array of lexicographically ordered affected region indexes"""
        return self._aff_regions_idx

    @aff_regions.setter
    def aff_regions(self, value: ArrayLike | str):
        if isinstance(value, str):
            value = [value]
        value = pd.Index(value)
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
        """The array of affected sectors by the event"""
        return self._aff_sectors

    @property
    def aff_sectors_idx(self) -> np.ndarray:
        """The array of lexicographically ordered affected sectors indexes"""
        return self._aff_sectors_idx

    @aff_sectors.setter
    def aff_sectors(self, value: ArrayLike | str):
        if isinstance(value, str):
            value = [value]
        value = pd.Index(value, name="sector")
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
    def impact_regional_distrib(self) -> np.ndarray:
        """The array specifying how damages are distributed among affected regions"""
        return self._impact_regional_distrib

    @impact_regional_distrib.setter
    def impact_regional_distrib(self, value: ArrayLike):
        if self.aff_regions is None:
            raise AttributeError("Affected regions attribute isn't set yet")
        value = np.array(value)
        if value.size != self.aff_regions.size:
            raise ValueError(
                "There are {} affected regions by the event and length of given damage distribution is {}".format(
                    self.aff_regions.size, value.size
                )
            )
        s = value.sum()
        if not math.isclose(s, 1):
            raise ValueError(
                "Damage distribution doesn't sum up to 1 but to {}, which is not valid".format(
                    s
                )
            )
        self._impact_regional_distrib = value

    # @property
    # def impact_sectoral_distrib(self) -> np.ndarray:
    #     """The array specifying how damages are distributed among affected sectors"""
    #     return self._impact_sectoral_distrib

    # @impact_sectoral_distrib.setter
    # def impact_sectoral_distrib(self, value: ArrayLike):
    #     if self.aff_sectors is None:
    #         raise AttributeError("Affected sectors attribute isn't set yet")
    #     value = np.array(value)
    #     if value.size != self.aff_sectors.size:
    #         raise ValueError(
    #             "There are {} affected sectors by the event and length of given damage distribution is {}".format(
    #                 self.aff_sectors.size, value.size
    #             )
    #         )
    #     s = value.sum()
    #     if not math.isclose(s, 1):
    #         raise ValueError(
    #             "Damage distribution doesn't sum up to 1 but to {}, which is not valid".format(
    #                 s
    #             )
    #         )
    #     self._impact_sectoral_distrib = value

    @property
    def impact_sectoral_distrib_type(self) -> str:
        """The type of damages distribution among sectors (currently only 'gdp')"""
        return self._impact_sectoral_distrib_type

    @impact_sectoral_distrib_type.setter
    def impact_sectoral_distrib_type(self, value: str):
        logger.debug(f"Setting _impact_sectoral_distrib_type to {value}")
        self._impact_sectoral_distrib_type = value

    def __repr__(self):
        # TODO: find ways to represent long lists
        return f""" [Representation WIP]
        Event(
              name = {self.name},
              occur = {self.occurrence},
              duration = {self.duration}
              aff_regions = {self.aff_regions},
              aff_sectors = {self.aff_sectors},
             )
        """


class EventArbitraryProd(Event):
    def __init__(
        self,
        impact: Impact,
        aff_regions: Optional[RegionsList] = None,
        aff_sectors: Optional[SectorsList] = None,
        aff_industries: Optional[IndustriesList] = None,
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
    ) -> None:
        super().__init__(
            impact,
            aff_regions,
            aff_sectors,
            aff_industries,
            impact_industries_distrib,
            impact_regional_distrib,
            impact_sectoral_distrib_type,
            impact_sectoral_distrib,
            name,
            occurrence,
            duration,
        )
        raise NotImplementedError()

    @property
    def prod_cap_delta_arbitrary(self) -> np.ndarray:
        """The optional array storing arbitrary (as in not related to kapital destroyed) production capacity loss"""
        return self._prod_cap_delta_arbitrary

    @prod_cap_delta_arbitrary.setter
    def prod_cap_delta_arbitrary(self, value: dict[str, float]):
        if self.aff_regions is None:
            raise AttributeError("Affected regions attribute isn't set yet")
        aff_sectors = np.array(list(value.keys()))
        aff_shares = np.array(list(value.values()))
        impossible_sectors = np.setdiff1d(aff_sectors, self.possible_sectors)
        if impossible_sectors.size > 0:
            raise ValueError(
                "These sectors are not in the model : {}".format(impossible_sectors)
            )
        self._aff_sectors = aff_sectors
        self._aff_sectors_idx = np.searchsorted(self.possible_sectors, aff_sectors)
        aff_industries_idx = np.array(
            [
                len(self.possible_sectors) * ri + si
                for ri in self.regions_idx
                for si in self._aff_sectors_idx
            ]
        )
        self._prod_cap_delta_arbitrary = np.zeros(shape=len(self.possible_sectors))
        self._prod_cap_delta_arbitrary[aff_industries_idx] = np.tile(
            aff_shares, self._aff_regions.size
        )


class EventKapitalDestroyed(Event, metaclass=abc.ABCMeta):
    def __init__(
        self,
        impact: Impact,
        aff_regions: Optional[RegionsList] = None,
        aff_sectors: Optional[SectorsList] = None,
        aff_industries: Optional[IndustriesList] = None,
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
    ) -> None:
        super().__init__(
            impact,
            aff_regions,
            aff_sectors,
            aff_industries,
            impact_industries_distrib,
            impact_regional_distrib,
            impact_sectoral_distrib_type,
            impact_sectoral_distrib,
            name,
            occurrence,
            duration,
        )
        # The only thing we have to do is affecting/computing the regional_sectoral_kapital_destroyed
        self.total_kapital_destroyed = self.total_impact
        self.total_kapital_destroyed /= self.monetary_factor
        self.remaining_kapital_destroyed = self.total_kapital_destroyed
        self._regional_sectoral_kapital_destroyed_0 = (
            self.impact_vector / self.monetary_factor
        )
        self.regional_sectoral_kapital_destroyed = (
            self.impact_vector / self.monetary_factor
        )

    @property
    def regional_sectoral_kapital_destroyed(self) -> np.ndarray:
        """The (optional) array of kapital destroyed per industry (ie region x sector)"""
        return self._regional_sectoral_kapital_destroyed

    @regional_sectoral_kapital_destroyed.setter
    def regional_sectoral_kapital_destroyed(self, value: ArrayLike):
        value = np.array(value)
        if value.shape != self.x_shape:
            raise ValueError(
                "Incorrect shape give for regional_sectoral_kapital_destroyed: {} given and {} expected".format(
                    value.shape, self.x_shape
                )
            )
        self._regional_sectoral_kapital_destroyed = value


class EventKapitalRebuild(EventKapitalDestroyed):
    def __init__(
        self,
        impact: Impact,
        rebuilding_sectors: Union[dict[str, float], pd.Series],
        rebuild_tau=60,
        aff_regions: Optional[RegionsList] = None,
        aff_sectors: Optional[SectorsList] = None,
        aff_industries: Optional[IndustriesList] = None,
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
    ) -> None:
        super().__init__(
            impact,
            aff_regions,
            aff_sectors,
            aff_industries,
            impact_industries_distrib,
            impact_regional_distrib,
            impact_sectoral_distrib_type,
            impact_sectoral_distrib,
            name,
            occurrence,
            duration,
        )

        self.rebuild_tau = rebuild_tau
        self.rebuilding_sectors = rebuilding_sectors

        rebuilding_demand = np.zeros(shape=self.z_shape)
        tmp = np.zeros(self.z_shape, dtype="float")
        mask = np.ix_(
            np.union1d(
                self._rebuilding_industries_RoW_idx, self._rebuilding_industries_idx
            ),
            self._aff_industries_idx,
        )
        rebuilding_demand = np.outer(
            self.rebuilding_sectors_shares, self.regional_sectoral_kapital_destroyed
        )
        logger.debug(
            f"kapital destroyed vec is {self.regional_sectoral_kapital_destroyed}"
        )
        logger.debug(f"rebuilding sectors shares are {self.rebuilding_sectors_shares}")
        logger.debug(f"Z shape is {self.z_shape}")
        logger.debug(f"Z_distrib[mask] has shape {self.Z_distrib[mask].shape}")
        logger.debug(f"reb_demand: {rebuilding_demand}")
        logger.debug(f"reb_demand: {rebuilding_demand.shape}")
        # reb_tiled = np.tile(rebuilding_demand, (len(self.possible_regions), 1))
        # reb_tiled = reb_tiled[mask]
        tmp[mask] = self.Z_distrib[mask] * rebuilding_demand[mask]
        self.rebuilding_demand_indus = tmp
        self.rebuilding_demand_house = np.zeros(shape=self.y_shape)

    @property
    def rebuilding_sectors(self) -> pd.Index:
        """The (optional) array of rebuilding sectors"""
        return self._rebuilding_sectors

    @property
    def rebuilding_sectors_idx(self) -> np.ndarray:
        """The (optional) array of indexes of the rebuilding sectors (in lexicographic order)"""
        return self._rebuilding_sectors_idx

    @property
    def rebuilding_sectors_shares(self) -> np.ndarray:
        """The array specifying how rebuilding demand is distributed along the rebuilding sectors"""
        return self._rebuilding_sectors_shares

    @rebuilding_sectors.setter
    def rebuilding_sectors(self, value: dict[str, float] | pd.Series):
        if isinstance(value, dict):
            reb_sectors = pd.Series(value)
        else:
            reb_sectors = value
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
                    len(self.possible_sectors) * ri + si
                    for ri in self._aff_regions_idx
                    for si in self._rebuilding_sectors_idx
                ]
            )
            self._rebuilding_industries_RoW_idx = np.array(
                [
                    len(self.possible_sectors) * ri + si
                    for ri in range(len(self.possible_regions))
                    if ri not in self._aff_regions_idx
                    for si in self._rebuilding_sectors_idx
                ]
            )
            self._rebuilding_sectors_shares[self._rebuilding_industries_idx] = np.tile(
                np.array(reb_sectors.values), len(self.aff_regions)
            )
            self._rebuilding_sectors_shares[
                self._rebuilding_industries_RoW_idx
            ] = np.tile(
                np.array(reb_sectors.values),
                (len(self.possible_regions) - len(self.aff_regions)),
            )

    @property
    def rebuilding_demand_house(self) -> np.ndarray:
        """The optional array of rebuilding demand from households"""
        return self._rebuilding_demand_house

    @rebuilding_demand_house.setter
    def rebuilding_demand_house(self, value: ArrayLike):
        value = np.array(value)
        if value.shape != self.y_shape:
            raise ValueError(
                "Incorrect shape give for rebuilding_demand_house: {} given and {} expected".format(
                    value.shape, self.y_shape
                )
            )
        self._rebuilding_demand_house = value

    @property
    def rebuilding_demand_indus(self) -> np.ndarray:
        """The optional array of rebuilding demand from industries (to rebuild their kapital)"""
        return self._rebuilding_demand_indus

    @rebuilding_demand_indus.setter
    def rebuilding_demand_indus(self, value: ArrayLike):
        value = np.array(value)
        if value.shape != self.z_shape:
            raise ValueError(
                "Incorrect shape give for rebuilding_demand_indus: {} given and {} expected".format(
                    value.shape, self.z_shape
                )
            )
        self._rebuilding_demand_indus = value
        # Also update kapital destroyed
        self._regional_sectoral_kapital_destroyed = value.sum(axis=0)

    @property
    def rebuildable(self) -> bool:
        """A boolean stating if an event can start rebuild"""
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
                    self._aff_regions,
                    self.total_kapital_destroyed,
                )
            )
        self._rebuildable = reb


class EventKapitalRecover(EventKapitalDestroyed):
    def __init__(
        self,
        impact: Impact,
        recovery_time: int,
        recovery_function: str = "linear",
        aff_regions: Optional[RegionsList] = None,
        aff_sectors: Optional[SectorsList] = None,
        aff_industries: Optional[IndustriesList] = None,
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
    ) -> None:
        super().__init__(
            impact,
            aff_regions,
            aff_sectors,
            aff_industries,
            impact_industries_distrib,
            impact_regional_distrib,
            impact_sectoral_distrib_type,
            impact_sectoral_distrib,
            name,
            occurrence,
            duration,
        )
        self.recovery_time = recovery_time
        self.recovery_function = recovery_function
        self._recoverable = False

    @property
    def recoverable(self) -> bool:
        """A boolean stating if an event can start recover"""
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
                    self._aff_regions,
                    self.total_kapital_destroyed,
                )
            )
        self._recoverable = reb

    @property
    def recovery_function(self) -> Callable:
        """The recovery function used for recovery (`Callable`)"""
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
                    "init_kapital_destroyed",
                    "elapsed_temporal_unit",
                    "recovery_time",
                ]
            ):
                raise ValueError(
                    "Recovery function has to have at least the following keyword arguments: {}".format(
                        [
                            "init_kapital_destroyed",
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
            init_kapital_destroyed=self._regional_sectoral_kapital_destroyed_0,
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
        precision = int(math.log10(self.monetary_factor)) + 1
        res = np.around(res, precision)
        if not np.any(res):
            self.over = True
        self.regional_sectoral_kapital_destroyed = res
