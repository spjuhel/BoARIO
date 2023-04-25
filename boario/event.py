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
from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, List, Tuple, get_origin, get_args
import numpy.typing as npt
import numpy as np
import pandas as pd
from boario import logger
import math
import inspect
from functools import partial

from boario.utils.recovery_functions import (
    concave_recovery,
    convexe_recovery,
    linear_recovery,
    convexe_recovery_scaled,
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

Impact = Union[int, float, list, dict, np.ndarray, pd.DataFrame, pd.Series]
IndustriesList = Union[List[Tuple[str, str]], pd.MultiIndex, npt.NDArray]
SectorsList = Union[List[str], pd.Index, npt.NDArray]
RegionsList = Union[List[str], pd.Index, npt.NDArray]
FinalCatList = Union[List[str], pd.Index, npt.NDArray]

rebuilding_finaldemand_cat_regex = r".*[hH]ousehold.*|HFCE"


class Event(ABC):
    # Class Attributes

    possible_sectors: pd.Index = pd.Index([])
    r"""List of sectors present in the MRIO used by the model"""

    possible_regions: pd.Index = pd.Index([])
    r"""List of regions present in the MRIO used by the model"""

    possible_final_demand_cat: pd.Index = pd.Index([])
    r"""List of final demand categories present in the MRIO used by the model"""

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

    model_monetary_factor: int = 0
    r"""Amount of unitary currency used in the MRIO (e.g. 1000000 if in € millions)"""

    sectors_gva_shares: npt.NDArray = np.array([])
    r"""Fraction of total (regional) GVA for each sectors"""

    Z_distrib: npt.NDArray = np.array([])
    r"""Normalized intermediate consumption matrix"""

    Y_distrib: npt.NDArray = np.array([])
    r"""Normalized final consumption matrix"""

    mrio_name: str = ""
    r"""MRIO identification"""

    @abstractmethod
    def __init__(
        self,
        *,
        impact: Impact,
        aff_regions: RegionsList = [],
        aff_sectors: SectorsList = [],
        aff_industries: IndustriesList = [],
        impact_industries_distrib: Optional[npt.ArrayLike] = None,
        impact_regional_distrib: Optional[npt.ArrayLike] = None,
        impact_sectoral_distrib_type: str = "custom",
        impact_sectoral_distrib: Optional[npt.ArrayLike] = None,
        name: str = "Unnamed",
        occurrence: int = 1,
        duration: int = 1,
        event_monetary_factor: Optional[int] = None,
    ) -> None:
        r"""Create an event shocking the model from a dictionary.

        An Event object stores all information about a unique shock during simulation such as
        time of occurrence, duration, type of shock, amount of damages. Computation
        of recovery or initially requested rebuilding demand is also done in this
        class.

        Parameters
        ----------

        event : dict
            A dictionary holding the necessary information to define an event.

        Examples
        --------
            FIXME: Add docs.

        """
        if len(self.possible_regions) == 0 or len(self.possible_sectors) == 0:
            raise AttributeError(
                "It appears that no model has been instantiated as some class attributes are not initialized (possible_regions, possible_sectors). Events require to instantiate a model and a simulation context before they can be instantiated"
            )

        if self.temporal_unit_range == 0:
            raise AttributeError(
                "It appears that no simulation context has been instantiated as some class attributes are not initialized (temporal_unit_range). Events require to instantiate a model and a simulation context before they can be instantiated"
            )

        self._aff_sectors_idx = np.array([])
        self._aff_sectors = pd.Index([])
        self._aff_regions_idx = np.array([])
        self._aff_regions = pd.Index([])
        impact_regional_distrib = (
            np.array(impact_regional_distrib)
            if impact_regional_distrib
            else np.array([])
        )
        impact_sectoral_distrib = (
            np.array(impact_sectoral_distrib)
            if impact_sectoral_distrib
            else np.array([])
        )
        impact_industries_distrib = (
            np.array(impact_industries_distrib)
            if impact_sectoral_distrib
            else np.array([])
        )
        logger.info("Initializing new Event")
        logger.debug("Checking required Class attributes are defined")

        if event_monetary_factor is None:
            logger.info(
                f"No event monetary factor given. Assuming it is the same as the model ({self.model_monetary_factor})"
            )
            self.event_monetary_factor = self.model_monetary_factor
        else:
            self.event_monetary_factor = event_monetary_factor
            if self.event_monetary_factor != self.model_monetary_factor:
                logger.warning(
                    f"Event monetary factor ({self.event_monetary_factor}) differs from model monetary factor ({self.model_monetary_factor}). Be careful to define your impact with the correct unit (ie in event monetary factor)."
                )

        self.name: str = name
        r"""An identifying name for the event (for convenience mostly)"""

        self.occurrence = occurrence
        self.duration = duration
        self.impact_df: pd.Series = pd.Series(
            0,
            dtype="float64",
            index=pd.MultiIndex.from_product(
                [self.possible_regions, self.possible_sectors],
                names=["region", "sector"],
            ),
        )
        r"""A pandas Series with all possible industries as index, holding the impact vector of the event. The impact is defined for each sectors in each region."""

        ################## DATAFRAME INIT #################
        # CASE VECTOR 1 (everything is there and regrouped) (only df creation)
        if isinstance(impact, pd.Series):
            logger.debug("Given impact is a pandas Series")
            self.impact_df.loc[impact.index] = impact
            if self.name == "Unnamed" and not impact.name is None:
                self.name = str(impact.name)
        elif isinstance(impact, dict):
            logger.debug("Given impact is a dict, converting it to pandas Series")
            impact = pd.Series(impact)
            self.impact_df.loc[impact.index] = impact
        elif isinstance(impact, pd.DataFrame):
            logger.debug("Given impact is a pandas DataFrame, squeezing it to a Series")
            impact = impact.squeeze()
            if not isinstance(impact, pd.Series):
                raise ValueError(
                    "The given impact DataFrame is not a Series after being squeezed"
                )
            self.impact_df.loc[impact.index] = impact
        # CASE VECTOR 2 (everything is there but not regrouped) AND CASE SCALAR (Only df creation)
        elif (
            isinstance(impact, (int, float, list, np.ndarray))
            and len(aff_industries) > 0
        ):
            logger.debug(
                f"Given impact is a {type(impact)} and list of impacted industries given. Proceeding."
            )
            self.impact_df.loc[aff_industries] = impact
        elif (
            isinstance(impact, (int, float, list, np.ndarray))
            and len(aff_regions) > 0
            and len(aff_sectors) > 0
        ):
            logger.debug(
                f"Given impact is a {type(impact)} and lists of impacted regions and sectors given. Proceeding."
            )
            if isinstance(aff_regions, str):
                aff_regions = [aff_regions]
            if isinstance(aff_sectors, str):
                aff_sectors = [aff_sectors]

            self.impact_df.loc[
                pd.MultiIndex.from_product([aff_regions, aff_sectors])
            ] = impact
        else:
            raise ValueError(
                "Invalid input format. Could not initiate pandas Series. Check https://spjuhel.github.io/BoARIO/boario-events.html for in depths explanation on how to define Events."
            )

        # Check for <0 values and remove 0.
        if (self.impact_df < 0).any():
            logger.warning(
                "Found negative values in impact vector. This should raise concern"
            )

        # SORT DF
        # at this point impact_df is built, and can be sorted. Note that if impact was a scalar, impact_df contains copies of this scalar.
        logger.debug("Sorting impact Series")
        self.impact_df = self.impact_df.sort_index()

        # Init self.impact_sectoral_distrib_type,
        self.impact_sectoral_distrib_type = impact_sectoral_distrib_type
        #################################################

        # SET INDEXES ATTR
        # note that the following also sets aff_regions and aff_sectors
        assert isinstance(self.impact_df.index, pd.MultiIndex)
        # Only look for industries where impact is greater than 0
        self.aff_industries = self.impact_df.loc[self.impact_df > 0].index

        logger.debug(
            f"impact df at the moment:\n {self.impact_df.loc[self.aff_industries]}"
        )

        ###### SCALAR DISTRIBUTION ######################
        # if impact_industries_distrib is given, set it. We assume impact is scalar !
        # CASE SCALAR + INDUS DISTRIB
        if len(impact_industries_distrib) > 0 and not isinstance(
            impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)
        ):
            logger.debug("impact is Scalar and impact_industries_distrib was given")
            self.impact_industries_distrib = np.array(impact_industries_distrib)
            self.impact_df.loc[self.aff_industries] = (
                self.impact_df.loc[self.aff_industries] * self.impact_industries_distrib
            )
        # if impact_reg_dis and sec_dis are give, deduce the rest. We also assume impact is scalar !
        # CASE SCALAR + REGION and SECTOR DISTRIB
        elif (
            len(impact_regional_distrib) > 0
            and len(impact_sectoral_distrib) > 0
            and not isinstance(
                impact,
                (pd.Series, dict, pd.DataFrame, list, np.ndarray),
            )
        ):
            logger.debug(
                "impact is Scalar and impact_regional_distrib and impact_sectoral_distrib were given"
            )
            if len(impact_regional_distrib) != len(self.aff_regions) or len(
                impact_sectoral_distrib
            ) != len(self.aff_sectors):
                raise ValueError(
                    "Lengths of `impact_regional_distrib` and/or `impact_sectoral_distrib` are incompatible with `aff_regions` and/or `aff_sectors`."
                )
            else:
                self.impact_regional_distrib = impact_regional_distrib
                self.impact_sectoral_distrib = impact_sectoral_distrib
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
            len(impact_regional_distrib) > 0
            and len(impact_sectoral_distrib_type) > 0
            and impact_sectoral_distrib_type == "gdp"
            and not isinstance(
                impact,
                (pd.Series, dict, pd.DataFrame, list, np.ndarray),
            )
        ):
            logger.debug("impact is Scalar and impact_sectoral_distrib_type is 'gdp'")

            self.impact_regional_distrib = impact_regional_distrib

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
                "impact is Scalar and no distribution was given but a list of affected industries was given"
            )
            self._default_distribute_impact_from_industries_list()
            self.impact_sectoral_distrib_type = "default (shared equally between affected regions and then affected sectors)"
        # CASE SCALAR + NO DISTRIB + list of region + list of sectors
        elif (
            len(aff_regions) > 0
            and len(aff_sectors) > 0
            and not isinstance(
                impact,
                (pd.Series, dict, pd.DataFrame, list, np.ndarray),
            )
        ):
            logger.debug(
                "impact is Scalar and no distribution was given but lists of regions and sectors affected were given"
            )
            self._default_distribute_impact_from_industries_list()
            self.impact_sectoral_distrib_type = "default (shared equally between affected regions and then affected sectors)"
        elif not isinstance(impact, (pd.Series, dict, pd.DataFrame, list, np.ndarray)):
            raise ValueError(f"Invalid input format: Could not compute impact")

        self._finish_init()
        ##################################################

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
            "impact_sectoral_distrib_type": self.impact_sectoral_distrib_type,
            "globals_vars": {
                "possible_sectors": list(self.possible_sectors),
                "possible_regions": list(self.possible_regions),
                "temporal_unit_range": self.temporal_unit_range,
                "z_shape": self.z_shape,
                "y_shape": self.y_shape,
                "x_shape": self.x_shape,
                "model_monetary_factor": self.model_monetary_factor,
                "event_monetary_factor": self.event_monetary_factor,
                "mrio_used": self.mrio_name,
            },
        }
        r"""Store relevant information about the event"""

    def _default_distribute_impact_from_industries_list(self):
        # at this point, impact should still be scalar.
        logger.debug("Using default impact distribution to industries")
        logger.debug(
            f"impact df at the moment:\n {self.impact_df.loc[self.aff_industries]}"
        )
        self.impact_regional_distrib = np.full(
            len(self.aff_regions), 1 / len(self.aff_regions)
        )

        logger.debug(
            f"self.impact_regional_distrib: {list(self.impact_regional_distrib)}"
        )
        logger.debug(f"len aff_regions: {len(self.aff_regions)}")
        self.impact_df.loc[self.aff_industries] = (
            self.impact_df.loc[self.aff_industries] * 1 / len(self.aff_regions)
        )
        impact_sec_vec = np.array(
            [
                1 / len(self.aff_industries.to_series().loc[reg])
                for reg in self.aff_regions
            ]
        )
        self.impact_df.loc[self.aff_industries] = (
            self.impact_df.loc[self.aff_industries] * impact_sec_vec
        )
        logger.debug(
            f"impact df after default distrib:\n {self.impact_df.loc[self.aff_industries]}"
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
        r"""The industries affected by the event.

        Parameters
        ----------

        index : pd.MultiIndex
             The affected industries as a pandas MultiIndex

        Returns
        -------
             A pandas MultiIndex with the regions affected as first level, and sectors affected as second level

        """

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
    def impact_regional_distrib(self) -> npt.NDArray:
        r"""The array specifying how damages are distributed among affected regions"""

        return self._impact_regional_distrib

    @impact_regional_distrib.setter
    def impact_regional_distrib(self, value: npt.ArrayLike):
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

    @property
    def impact_sectoral_distrib_type(self) -> str:
        r"""The type of damages distribution among sectors (currently only 'gdp')"""

        return self._impact_sectoral_distrib_type

    @impact_sectoral_distrib_type.setter
    def impact_sectoral_distrib_type(self, value: str):
        logger.debug(f"Setting _impact_sectoral_distrib_type to {value}")
        self._impact_sectoral_distrib_type = value

    def __repr__(self):
        # TODO: find ways to represent long lists
        return f"""
        Event(
              name = {self.name},
              occur = {self.occurrence},
              duration = {self.duration}
              aff_regions = {self.aff_regions.to_list()},
              aff_sectors = {self.aff_sectors.to_list()},
             )
        """


class EventArbitraryProd(Event):
    def __init__(
        self,
        *,
        impact: Impact,
        recovery_time: int = 1,
        recovery_function: str = "linear",
        aff_regions: RegionsList = [],
        aff_sectors: SectorsList = [],
        aff_industries: IndustriesList = [],
        impact_industries_distrib: Optional[npt.ArrayLike] = None,
        impact_regional_distrib: Optional[npt.ArrayLike] = None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib: Optional[npt.ArrayLike] = None,
        name: str = "Unnamed",
        occurrence: int = 1,
        duration: int = 1,
    ) -> None:
        if not isinstance(impact, pd.Series):
            raise NotImplementedError(
                "Arbitrary production capacity shock currently require to be setup from a vector indexed by the industries (regions,sectors) affected, and where values are the share of production capacity lost."
            )

        if (impact > 1.0).any():
            raise ValueError(
                "Impact is greater than 100% (1.) for at least an industry."
            )

        super().__init__(
            impact=impact,
            aff_regions=aff_regions,
            aff_sectors=aff_sectors,
            aff_industries=aff_industries,
            impact_industries_distrib=impact_industries_distrib,
            impact_regional_distrib=impact_regional_distrib,
            impact_sectoral_distrib_type=impact_sectoral_distrib_type,
            impact_sectoral_distrib=impact_sectoral_distrib,
            name=name,
            occurrence=occurrence,
            duration=duration,
            event_monetary_factor=None,
        )

        self._prod_cap_delta_arbitrary_0 = (
            self.impact_vector.copy()
        )  # np.zeros(shape=len(self.possible_sectors))
        self.prod_cap_delta_arbitrary = (
            self.impact_vector.copy()
        )  # np.zeros(shape=len(self.possible_sectors))
        self.recovery_time = recovery_time
        r"""The characteristic recovery duration after the event is over"""

        self.recovery_function = recovery_function

        logger.info("Initialized")

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
    def __init__(
        self,
        *,
        impact: Impact,
        households_impact: Optional[Impact] = None,
        aff_regions: RegionsList = [],
        aff_sectors: SectorsList = [],
        aff_industries: IndustriesList = [],
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
        event_monetary_factor=None,
    ) -> None:
        super().__init__(
            impact=impact,
            aff_regions=aff_regions,
            aff_sectors=aff_sectors,
            aff_industries=aff_industries,
            impact_industries_distrib=impact_industries_distrib,
            impact_regional_distrib=impact_regional_distrib,
            impact_sectoral_distrib_type=impact_sectoral_distrib_type,
            impact_sectoral_distrib=impact_sectoral_distrib,
            name=name,
            occurrence=occurrence,
            duration=duration,
            event_monetary_factor=event_monetary_factor,
        )
        # The only thing we have to do is affecting/computing the regional_sectoral_productive_capital_destroyed
        self.total_productive_capital_destroyed = self.total_impact
        self.total_productive_capital_destroyed *= (
            self.event_monetary_factor / self.model_monetary_factor
        )
        logger.info(
            f"Total impact on productive capital is {self.total_productive_capital_destroyed} (in model unit)"
        )
        self.remaining_productive_capital_destroyed = (
            self.total_productive_capital_destroyed
        )
        self._regional_sectoral_productive_capital_destroyed_0 = self.impact_vector * (
            self.event_monetary_factor / self.model_monetary_factor
        )
        self.regional_sectoral_productive_capital_destroyed = self.impact_vector * (
            self.event_monetary_factor / self.model_monetary_factor
        )
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
                        rebuilding_finaldemand_cat_regex
                    )
                ]  # .values[0]

            except IndexError:
                rebuilding_demand_idx = self.possible_final_demand_cat[0]
            if isinstance(households_impact, pd.Series):
                logger.debug("Given household impact is a pandas Series")
                self.households_impact_df.loc[
                    households_impact.index, rebuilding_demand_idx
                ] = households_impact
            elif isinstance(households_impact, dict):
                logger.debug("Given household impact is a dict")
                households_impact = pd.Series(households_impact)
                self.households_impact_df.loc[
                    households_impact.index, rebuilding_demand_idx
                ] = households_impact
            elif isinstance(households_impact, pd.DataFrame):
                logger.debug("Given household impact is a dataframe")
                households_impact = households_impact.squeeze()
                if not isinstance(households_impact, pd.Series):
                    raise ValueError(
                        "The given households_impact DataFrame is not a Series after being squeezed"
                    )
                self.households_impact_df.loc[
                    households_impact.index, rebuilding_demand_idx
                ] = households_impact
            elif isinstance(households_impact, (list, np.ndarray)):
                if len(households_impact) != len(self.aff_regions):
                    raise ValueError(
                        f"Length mismatch between households_impact ({len(households_impact)} and affected regions ({len(self.aff_regions)}))"
                    )
                else:
                    self.households_impact_df.loc[
                        self.aff_regions, rebuilding_demand_idx
                    ] = households_impact
            elif isinstance(households_impact, (float, int)):
                if self.impact_regional_distrib is not None:
                    logger.warning(
                        f"households impact given as scalar, distributing among region following `impact_regional_distrib` ({self.impact_regional_distrib}) "
                    )
                    self.households_impact_df.loc[
                        self.aff_regions, rebuilding_demand_idx
                    ] = (households_impact * self.impact_regional_distrib)
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
    def __init__(
        self,
        *,
        impact: Impact,
        rebuilding_sectors: Union[dict[str, float], pd.Series],
        rebuild_tau=60,
        households_impact: Optional[Impact] = None,
        aff_regions: RegionsList = [],
        aff_sectors: SectorsList = [],
        aff_industries: IndustriesList = [],
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
        rebuilding_factor: float = 1.0,
        event_monetary_factor=None,
    ) -> None:
        super().__init__(
            impact=impact,
            households_impact=households_impact,
            aff_regions=aff_regions,
            aff_sectors=aff_sectors,
            aff_industries=aff_industries,
            impact_industries_distrib=impact_industries_distrib,
            impact_regional_distrib=impact_regional_distrib,
            impact_sectoral_distrib_type=impact_sectoral_distrib_type,
            impact_sectoral_distrib=impact_sectoral_distrib,
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
        assert np.isclose(reb_sectors.sum(), 1.0)
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
        value[value < 10 / self.model_monetary_factor] = 0.0
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
        value[value < 10 / self.model_monetary_factor] = 0.0
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
    def __init__(
        self,
        *,
        impact: Impact,
        recovery_time: int,
        recovery_function: str = "linear",
        households_impact: Impact | None = None,
        aff_regions: RegionsList = [],
        aff_sectors: SectorsList = [],
        aff_industries: IndustriesList = [],
        impact_industries_distrib=None,
        impact_regional_distrib=None,
        impact_sectoral_distrib_type="equally shared",
        impact_sectoral_distrib=None,
        name="Unnamed",
        occurrence=1,
        duration=1,
        event_monetary_factor=None,
    ) -> None:
        super().__init__(
            impact=impact,
            households_impact=households_impact,
            aff_regions=aff_regions,
            aff_sectors=aff_sectors,
            aff_industries=aff_industries,
            impact_industries_distrib=impact_industries_distrib,
            impact_regional_distrib=impact_regional_distrib,
            impact_sectoral_distrib_type=impact_sectoral_distrib_type,
            impact_sectoral_distrib=impact_sectoral_distrib,
            name=name,
            occurrence=occurrence,
            duration=duration,
            event_monetary_factor=event_monetary_factor,
        )
        self.recovery_time = recovery_time
        self.recovery_function = recovery_function
        self._recoverable = False

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
