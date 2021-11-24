import warnings

class Event(object):
    def __init__(self, event:dict) -> None:
        super().__init__()

        self.name = event['name']
        self.occurence_time = event['occur']
        self.q_damages = event['q_dmg']
        self.aff_regions = event['aff-regions']
        if type(self.aff_regions) is str:
            self.aff_regions = [self.aff_regions]
        self.aff_sectors = event['aff-sectors']
        if type(self.aff_sectors) is str:
            self.aff_sectors = [self.aff_sectors]
        self.duration = event['duration']
        self.dmg_distrib_across_regions = event['dmg-distrib-regions']
        self.dmg_distrib_across_sectors = event['dmg-distrib-sectors']
        self.rebuilding_sectors = event['rebuilding-sectors']

    def check_values(self, model):
        if self.occurence_time < 0:
            raise ValueError("Event occurence time is negative, check events json")
        if self.occurence_time > model.n_timesteps_to_sim:
            raise ValueError("Event occurence time is outside simulation, check events and sim json")
        if self.q_damages < 0:
            raise ValueError("Event damages are negative, check events json")
        if not set(self.aff_regions).issubset(model.mrio.regions):
            raise ValueError("A least one affected region is not a valid region in the mrio table, check events json")
        if not set(self.aff_sectors).issubset(model.mrio.sectors):
            raise ValueError("A least one affected sector is not a valid sector in the mrio table, check events json")
        if not set(self.rebuilding_sectors).issubset(model.mrio.sectors):
            raise ValueError("A least one rebuilding sector is not a valid sector in the mrio table, check events json")
        if self.duration < 0:
            raise ValueError("Event duration is negative, check events json")
        if self.occurence_time+self.duration > model.n_timesteps_to_sim:
            raise ValueError("Event occurence time + duration is outside simulation, check events and sim json")

        if self.dmg_distrib_across_regions is None:
            if not len(self.aff_regions) == 1:
                raise ValueError("Parameter 'dmg-distrib-across_regions' is None yet there are more than one region affected")
        elif type(self.dmg_distrib_across_regions) == str:
            if self.dmg_distrib_across_regions !=  "shared":
                raise ValueError("damage <-> region distribution %s not implemented",self.dmg_distrib_across_regions)
        elif type(self.dmg_distrib_across_regions) == list:
            if len(self.dmg_distrib_across_regions) != len(self.aff_regions):
                raise ValueError("Number of affected regions and size of damage distribution list are not equal")
            if sum(self.dmg_distrib_across_regions) != 1.0:
                warnings.warn("The total distribution of damage across regions is not 1.0")
        else:
            raise TypeError("'dmg-distrib-regions' is of type %s, possible types are str or list[float]", type(self.dmg_distrib_across_regions))

        if self.dmg_distrib_across_sectors is None:
            if not len(self.aff_sectors) == 1:
                raise ValueError("Parameter 'dmg-distrib-across_sectors' is None yet there are more than one sector affected")
        elif type(self.dmg_distrib_across_sectors) == str:
            if self.dmg_distrib_across_sectors !=  "GDP":
                raise ValueError("damage <-> sectors distribution %s not implemented",self.dmg_distrib_across_sectors)
        elif type(self.dmg_distrib_across_sectors) == list:
            if len(self.dmg_distrib_across_sectors) != len(self.aff_sectors):
                raise ValueError("Number of affected sectors and size of damage distribution list are not equal")
            if sum(self.dmg_distrib_across_sectors) != 1.0:
                warnings.warn("The total distribution of damage across sectors is not 1.0")
        else:
            raise TypeError("'dmg-distrib-sectors' is of type %s, possible types are str or list[float]", type(self.dmg_distrib_across_sectors))
