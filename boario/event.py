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
from typing import Optional
from numpy.typing import ArrayLike
import warnings
import numpy as np
from boario import logger
import math

__all__ = ['Event']

class Event(object):
    possible_sectors : np.ndarray = None #type: ignore
    possible_regions : np.ndarray = None #type: ignore
    temporal_unit_range : int = None #type: ignore
    z_shape : tuple[int,int] = None #type: ignore
    y_shape : tuple[int,int] = None #type: ignore
    x_shape : tuple[int,int] = None #type: ignore
    regions_idx : np.ndarray = None #type: ignore
    sectors_idx : np.ndarray = None #type: ignore
    monetary_unit : int = None #type: ignore
    sectors_gva_shares : np.ndarray = None #type: ignore
    Z_distrib: np.ndarray = None #type: ignore
    mrio_name: str = ""

    def __init__(self, event:dict) -> None:
        super().__init__()
        if event.get("globals_var") is not None:
            for k,v in event["globals_var"].items():
                if Event.__dict__[k] != v:
                    logger.warning("""You are trying to load an event which was simulated under different global vars, this might break: key:{} with value {} in dict and {} in Event class""".format(k,v,Event.__dict__[k]))
        for k,v in Event.__dict__.items():
            if k!="__doc__" and v is None:
                raise AttributeError("Event Class attribute {} is not set yet so instantiating an Event isn't possible".format(k))
        self.shock_type = event["shock_type"]
        self.name = event.get("name","unnamed")
        self.occurrence = event.get("occur",1)
        self.duration = event.get("duration",1)
        self.recovery = event.get("recovery")
        self.aff_regions = event["aff_regions"] #TODO make it a list if it is a str
        self.aff_sectors = event["aff_sectors"]
        self.dmg_regional_distrib = event.get("dmg_regional_distrib",[1/self.aff_regions.size for _ in range(self.aff_regions.size)])
        self.dmg_sectoral_distrib_type = event.get("dmg_sectoral_distrib_type","equally shared")
        self.dmg_sectoral_distrib = event.get("dmg_sectoral_distrib",[1/self.aff_sectors.size for _ in range(self.aff_sectors.size)])
        self.rebuilding_sectors = event["rebuilding_sectors"]
        self.total_kapital_destroyed = event.get("kapital_damage")
        self.rebuilding_demand_house = None
        self.rebuilding_demand_indus = None
        self.production_share_allocated = None
        self.prod_cap_delta_arbitrary = None
        self.happened = False
        self._rebuildable_kind : bool = False
        self._rebuildable : bool = False
        self.rebuild_tau = event.get("rebuild_tau")
        self.over = False
        self.init_shock(event)
        self.event_dict = event
        self.event_dict["globals_vars"] = {
            "possible_sectors" : list(self.possible_sectors),
            "possible_regions" : list(self.possible_regions),
            "temporal_unit_range" : self.temporal_unit_range,
            "z_shape" : self.z_shape,
            "y_shape" : self.y_shape,
            "x_shape" : self.x_shape,
            "monetary_unit" : self.monetary_unit,
            "mrio_used" : self.mrio_name
            }


    @property
    def occurrence(self)->int:
        return self._occur

    @occurrence.setter
    def occurrence(self, value:int):
        if not 0 <= value <= self.temporal_unit_range:
            raise ValueError("Occurrence of event is not in the range of simulation steps : {} not in [0-{}]".format(value, self.temporal_unit_range))
        else:
            self._occur=value

    @property
    def duration(self)->int:
        return self._duration

    @duration.setter
    def duration(self, value:int):
        if not 0 <= self.occurrence + value <= self.temporal_unit_range:
            raise ValueError("Occurrence + duration of event is not in the range of simulation steps : {} + {} not in [0-{}]".format(self.occurrence, value, self.temporal_unit_range))
        else:
            self._duration=value

    @property
    def aff_regions(self)->np.ndarray:
        return self._aff_regions

    @property
    def aff_regions_idx(self)->np.ndarray:
        return self._aff_regions_idx

    @aff_regions.setter
    def aff_regions(self, value:ArrayLike|str):
        if isinstance(value,str):
            value = [value]
        value = np.array(value)
        impossible_regions = np.setdiff1d(value,self.possible_regions)
        if impossible_regions.size > 0:
            raise ValueError("These regions are not in the model : {}".format(impossible_regions))
        else:
            self._aff_regions = value
            self._aff_regions_idx = np.searchsorted(self.possible_regions, value)

    @property
    def aff_sectors(self)->np.ndarray:
        return self._aff_sectors

    @property
    def aff_sectors_idx(self)->np.ndarray:
        return self._aff_sectors_idx

    @aff_sectors.setter
    def aff_sectors(self, value:ArrayLike|str):
        if isinstance(value,str):
            value = [value]
        value = np.array(value)
        impossible_sectors = np.setdiff1d(value,self.possible_sectors)
        if impossible_sectors.size > 0 :
            raise ValueError("These sectors are not in the model : {}".format(impossible_sectors))
        else:
            self._aff_sectors = value
            self._aff_sectors_idx = np.searchsorted(self.possible_sectors, value)

    @property
    def dmg_regional_distrib(self)->np.ndarray:
        return self._dmg_regional_distrib

    @dmg_regional_distrib.setter
    def dmg_regional_distrib(self,value:ArrayLike):
        if self.aff_regions is None:
            raise AttributeError("Affected regions attribute isn't set yet")
        value = np.array(value)
        if value.size != self.aff_regions.size:
            raise ValueError("There are {} affected regions by the event and length of given damage distribution is {}".format(self.aff_regions.size,value.size))
        s = value.sum()
        if not math.isclose(s,1):
            raise ValueError("Damage distribution doesn't sum up to 1 but to {}, which is not valid".format(s))
        self._dmg_regional_distrib = value

    @property
    def dmg_sectoral_distrib(self)->np.ndarray:
        return self._dmg_sectoral_distrib

    @dmg_sectoral_distrib.setter
    def dmg_sectoral_distrib(self,value:ArrayLike):
        if self.aff_sectors is None:
            raise AttributeError("Affected sectors attribute isn't set yet")
        value = np.array(value)
        if value.size != self.aff_sectors.size:
            raise ValueError("There are {} affected sectors by the event and length of given damage distribution is {}".format(self.aff_sectors.size,value.size))
        s = value.sum()
        if not math.isclose(s,1):
            raise ValueError("Damage distribution doesn't sum up to 1 but to {}, which is not valid".format(s))
        self._dmg_sectoral_distrib = value

    @property
    def dmg_sectoral_distrib_type(self)->str:
        return self._dmg_sectoral_distrib_type

    @dmg_sectoral_distrib_type.setter
    def dmg_sectoral_distrib_type(self,value:str):
        self._dmg_sectoral_distrib_type=value

    @property
    def rebuilding_sectors(self)->Optional[np.ndarray]:
        return self._rebuilding_sectors

    @property
    def rebuilding_sectors_idx(self)->Optional[np.ndarray]:
        return self._rebuilding_sectors_idx

    @property
    def rebuilding_sectors_shares(self)->Optional[np.ndarray]:
        return self._rebuilding_sectors_shares

    @rebuilding_sectors.setter
    def rebuilding_sectors(self, value:dict[str,float]|None):
        if value is None:
            self._rebuilding_sectors = None
            self._rebuilding_sectors_idx = None
        else:
            reb_sectors = np.array(list(value.keys()))
            reb_shares = np.array(list(value.values()))
            impossible_sectors = np.setdiff1d(reb_sectors,self.possible_sectors)
            if impossible_sectors.size > 0 :
                raise ValueError("These sectors are not in the model : {}".format(impossible_sectors))
            else:
                self._rebuilding_sectors = reb_sectors
                self._rebuilding_sectors_idx = np.searchsorted(self.possible_sectors, reb_sectors)
                self._rebuilding_sectors_shares = reb_shares

    @property
    def rebuilding_demand_house(self)->Optional[np.ndarray]:
        return self._rebuilding_demand_house

    @rebuilding_demand_house.setter
    def rebuilding_demand_house(self,value:ArrayLike|None):
        if value is None:
            self._rebuilding_demand_house = None
        else:
            value = np.array(value)
            if value.shape != self.y_shape:
                raise ValueError("Incorrect shape give for rebuilding_demand_house: {} given and {} expected".format(value.shape, self.y_shape))
            self._rebuilding_demand_house = value

    @property
    def rebuilding_demand_indus(self)->Optional[np.ndarray]:
        return self._rebuilding_demand_indus

    @rebuilding_demand_indus.setter
    def rebuilding_demand_indus(self,value:ArrayLike|None):
        if value is None:
            self._rebuilding_demand_indus = value
        else:
            value = np.array(value)
            if value.shape != self.z_shape:
                raise ValueError("Incorrect shape give for rebuilding_demand_indus: {} given and {} expected".format(value.shape, self.z_shape))
            self._rebuilding_demand_indus = value

    @property
    def rebuildable(self)->Optional[bool]:
        return self._rebuildable

    @rebuildable.setter
    def rebuildable(self,current_temporal_unit:int):
        if self._rebuildable_kind:
            reb = (self.occurrence + self.duration) <= current_temporal_unit
            if reb and not self.rebuildable :
                logger.info("Temporal_Unit : {} ~ Event named {} that occured at {} in {} for {} damages has started rebuilding".format(current_temporal_unit,self.name,self.occurrence, self._aff_regions, self.total_kapital_destroyed))
            self._rebuildable = reb

    @property
    def prod_cap_delta_arbitrary(self)->Optional[np.ndarray]:
        return self._prod_cap_delta_arbitrary

    @prod_cap_delta_arbitrary.setter
    def prod_cap_delta_arbitrary(self, value:dict[str,float]|None):
        if value is None:
            self._prod_cap_delta_arbitrary = None
        else:
            if self.aff_regions is None:
                raise AttributeError("Affected regions attribute isn't set yet")
            aff_sectors = np.array(list(value.keys()))
            aff_shares = np.array(list(value.values()))
            impossible_sectors = np.setdiff1d(aff_sectors,self.possible_sectors)
            if impossible_sectors.size > 0 :
                raise ValueError("These sectors are not in the model : {}".format(impossible_sectors))
            self._aff_sectors = aff_sectors
            self._aff_sectors_idx = np.searchsorted(self.possible_sectors, aff_sectors)
            aff_industries_idx = np.array([self.possible_sectors.size * ri + si for ri in self.regions_idx for si in self._aff_sectors_idx])
            self._prod_cap_delta_arbitrary = np.zeros(shape=self.possible_sectors.size)
            self._prod_cap_delta_arbitrary[aff_industries_idx] = np.tile(aff_shares,self._aff_regions.size)

    def init_shock(self, event:dict):
        """Initiate shock from the event. Methods called by __init__.

        Sets the rebuilding demand for households and industry.

        First, if multiple regions are affected, it computes the vector of how damages are distributed across these.
        Then it computes the vector of how regional damages are distributed across affected sectors.
        It produces a ``n_regions`` * ``n_sectors`` sized vector hence stores the damage (i.e. capital destroyed) for all industries.

        This method also computes the `rebuilding demand` matrices from households and industries, i.e. the demand addressed to the rebuilding
        industries consequent to the shock.

        See :ref:`How to define Events <boario-events>` for further detail on how to parameter these distribution.

        Parameters
        ----------
        event_to_add_id : int
            The id (rank it the ``events`` list) of the event to shock the model with.

        Raises
        ------
        ValueError
            Raised if the production share allocated to rebuilding (in either
            the impacted regions or the others) is not in [0,1].
        """

        if self.shock_type == "kapital_destroyed_rebuild":
            self._rebuildable_kind = True
            if self.total_kapital_destroyed is None:
                raise AttributeError("Total kapital destroyed isn't set yet, this shouldn't happen.")
            regions_idx = np.arange(self.possible_regions.size)
            aff_industries_idx = np.array([self.possible_sectors.size * ri + si for ri in self._aff_regions_idx for si in self._aff_sectors_idx])
            if not isinstance(self.total_kapital_destroyed, (int,float)):
                raise ValueError("Kapital damages is {}, which is not valid".format(type(self.total_kapital_destroyed)))
            self.total_kapital_destroyed /= self.monetary_unit
            self.remaining_kapital_destroyed = self.total_kapital_destroyed
            regional_damages = np.array(self.dmg_regional_distrib) * self.total_kapital_destroyed
            # GDP CASE
            if self.dmg_sectoral_distrib_type == "gdp":
                shares = self.sectors_gva_shares.reshape((self.possible_regions.size,self.possible_sectors.size))
                self.dmg_sectoral_distrib = (shares[self._aff_regions_idx][:,self._aff_sectors_idx]/shares[self._aff_regions_idx][:,self._aff_sectors_idx].sum(axis=1)[:,np.newaxis])

            regional_sectoral_damages = regional_damages * self.dmg_sectoral_distrib

            # REBUILDING
            if self._rebuilding_sectors_idx is None:
                raise ValueError("Rebuilding sectors are not set for this event")
            rebuilding_industries_idx = np.array([self.possible_sectors.size * ri + si for ri in self._aff_regions_idx for si in self._rebuilding_sectors_idx])
            rebuilding_industries_RoW_idx = np.array([self.possible_sectors.size * ri + si for ri in regions_idx if ri not in self._aff_regions_idx for si in self._rebuilding_sectors_idx])
            rebuilding_demand = np.outer(self._rebuilding_sectors_shares,regional_sectoral_damages)
            tmp = np.zeros(self.z_shape,dtype="float")
            mask = np.ix_(np.union1d(rebuilding_industries_RoW_idx,rebuilding_industries_idx),aff_industries_idx)

            tmp[mask] = self.Z_distrib[mask] * np.tile(rebuilding_demand, (self.possible_regions.size,1))
            self.rebuilding_demand_indus = tmp

            self.rebuilding_demand_house = np.zeros(shape=self.y_shape)
            #self.rebuilding_share_house = np.zeros(shape=self.z_shape)
            #self.rebuildind_share_indus = np.zeros(shape=self.z_shape)
            #self.production_share_allocated = np.zeros(shape=self.x_shape)

        elif self.shock_type == "production_capacity_loss":
            self.prod_cap_delta = event["prod_cap_delta"]

    def __repr__(self):
        #TODO: find ways to represent long lists
        return f''' [Representation WIP]
        Event(
              name = {self.name},
              occur = {self.occurrence},
              duration = {self.duration}
              aff_regions = {self.aff_regions},
              aff_sectors = {self.aff_sectors},
             )
        '''
