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
from typing import Union
from boario.model_base import ARIOBaseModel
from boario.extended_models import ARIOModelPsi
import warnings
import numpy as np
from boario import logger
from boario.logging_conf import DEBUGFORMATTER

__all__ = ['Event']

class Event(object):
    def __init__(self, event:dict, mrio:Union[ARIOBaseModel,ARIOModelPsi]) -> None:
        super().__init__()

        self.name = event['name']
        self.occurence_time = event['occur']
        self.duration = event['duration']
        self.q_damages = event['q_dmg']
        self.aff_regions = event['aff_regions']
        if type(self.aff_regions) is str:
            self.aff_regions = [self.aff_regions]
        self.aff_sectors = event['aff_sectors']
        if type(self.aff_sectors) is str:
            self.aff_sectors = [self.aff_sectors]
        self.dmg_distrib_across_regions = event['dmg_distrib_regions']
        self.dmg_distrib_across_sectors_type = event['dmg_distrib_sectors_type']
        self.dmg_distrib_across_sectors = event['dmg_distrib_sectors']
        self.rebuilding_sectors = event['rebuilding_sectors']
        self.final_demand_rebuild = np.zeros(shape=mrio.Y_0.shape)
        self.final_demand_rebuild_share = np.zeros(shape=mrio.Y_0.shape)
        self.industry_rebuild_share = np.zeros(shape=mrio.Z_0.shape)
        self.industry_rebuild = np.zeros(shape=mrio.Z_0.shape)
        self.production_share_allocated = np.zeros(shape=mrio.production.shape)
        self.rebuildable = False

    def __repr__(self):
        #TODO: find ways to represent long lists
        return f''' [Representation WIP]
        Event(
              name = {self.name},
              occurence_time = {self.occurence_time},
              q_damages = {self.q_damages},
              aff_regions = {self.aff_regions},
              aff_sectors = {self.aff_sectors},
              occurence_time = {self.occurence_time},
              duration = {self.duration},
              dmg_distrib_across_sectors_type = {self.dmg_distrib_across_sectors_type}
             )
        '''

    def check_values(self, sim) -> None:
        if self.occurence_time < 0:
            raise ValueError("Event occurence time is negative, check events json")
        if self.occurence_time > sim.n_temporal_units_to_sim:
            raise ValueError("Event occurence time is outside simulation, check events and sim json")
        if self.q_damages < 0:
            raise ValueError("Event damages are negative, check events json")
        if not set(self.aff_regions).issubset(sim.model.regions):
            tmp = set(self.aff_regions).difference(set(sim.model.regions))
            raise ValueError("""A least one affected region is not a valid region in the mrio table, check events json

            suspicious regions : {}
            """.format(tmp))
        if not set(self.aff_sectors).issubset(sim.model.sectors):
            tmp = set(self.aff_sectors).difference(set(sim.model.sectors))
            raise ValueError("""A least one affected sector is not a valid sector in the mrio table, check events json

            suspicious sectors : {}
            """.format(tmp))
        if not set(self.rebuilding_sectors).issubset(sim.model.sectors):
            tmp = set(self.rebuilding_sectors).difference(set(sim.model.sectors))
            raise ValueError("""A least one rebuilding sector is not a valid sector in the mrio table, check events json

            suspicious sectors : {}
            """.format(tmp))
        if self.duration < 0:
            raise ValueError("Event duration is negative, check events json")
        if self.occurence_time+self.duration > sim.n_temporal_units_to_sim:
            raise ValueError("Event occurence time + duration is outside simulation, check events and sim json")

        if self.dmg_distrib_across_regions is None:
            if not len(self.aff_regions) == 1:
                raise ValueError("Parameter 'dmg_distrib_across_regions' is None yet there are more than one region affected")
        elif type(self.dmg_distrib_across_regions) == str:
            if self.dmg_distrib_across_regions !=  "shared":
                raise ValueError("damage <-> region distribution %s not implemented",self.dmg_distrib_across_regions)
        elif type(self.dmg_distrib_across_regions) == list:
            if len(self.dmg_distrib_across_regions) != len(self.aff_regions):
                raise ValueError("Number of affected regions and size of damage distribution list are not equal")
            if sum(self.dmg_distrib_across_regions) != 1.0:
                warnings.warn("The total distribution of damage across regions is not 1.0")
        else:
            raise TypeError("'dmg_distrib_regions' is of type %s, possible types are str or list[float]", type(self.dmg_distrib_across_regions))

        if self.dmg_distrib_across_sectors_type != 'gdp':
            if self.dmg_distrib_across_sectors is None:
                if not len(self.aff_sectors) == 1:
                    raise ValueError("Parameter 'dmg_distrib_across_sectors' is None yet there are more than one sector affected")
            elif type(self.dmg_distrib_across_sectors) == str:
                if self.dmg_distrib_across_sectors !=  "GDP":
                    raise ValueError("damage <-> sectors distribution %s not implemented",self.dmg_distrib_across_sectors)
            elif type(self.dmg_distrib_across_sectors) == list:
                if len(self.dmg_distrib_across_sectors) != len(self.aff_sectors):
                    raise ValueError("Number of affected sectors and size of damage distribution list are not equal")
                if sum(self.dmg_distrib_across_sectors) != 1.0:
                    warnings.warn("The total distribution of damage across sectors is not 1.0")
            else:
                raise TypeError("'dmg_distrib_sectors' is of type %s, possible types are str or list[float]", type(self.dmg_distrib_across_sectors))
