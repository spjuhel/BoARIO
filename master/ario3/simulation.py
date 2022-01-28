'''
Simulation module
'''

import json
import pickle
import pathlib
from typing import Union

import numpy as np
import progressbar
import pymrio as pym
from pymrio.core.mriosystem import IOSystem

from ario3.event import Event
from ario3.mriosystem import MrioSystem
from ario3.utils.logger import init_logger

__all__=['Simulation']

logger = init_logger(__name__,pathlib.Path.cwd()/"run.log")

class Simulation(object):
    '''Simulation instance'''
    def __init__(self, params: Union[dict, str, pathlib.Path], mrio_system: Union[IOSystem, str, pathlib.Path, None] = None) -> None:
        """Initiate a simulation with an MRIO pickle file.

        :param params:
        :param mrio_system:

        """
        logger.debug("Initializing Simulation instance")
        super().__init__()
        if isinstance(params, str):
            params_path=pathlib.Path(params)
        if isinstance(params, pathlib.Path):
            params_path=params
            if not params_path.is_dir():
                raise FileNotFoundError("This path should be a directory containing the different simulation parameters files:", params)
            simulation_params_path = params_path / 'params.json'
            if not simulation_params_path.exists():
                raise FileNotFoundError("Simulation parameters file not found, it should be here: ",simulation_params_path.absolute())
            else:
                with simulation_params_path.open() as f:
                    simulation_params = json.load(f)
        if isinstance(params, dict):
            simulation_params = params
            if simulation_params['mrio_params_file'] is None:
                logger.warn("Params given as a dict but 'mrio_params_file' does not exist. Will try with default one.")
        else:
            raise TypeError("params must be either a dict, a str or a pathlib.Path, not a %s", str(type(params)))

        mrio_params = None
        if simulation_params['mrio_params_file'] is not None:
            mrio_params_path = pathlib.Path(simulation_params['mrio_params_file'])
            if not mrio_params_path.exists():
                logger.warning('MRIO parameters file specified in simulation params was not found, trying the default one.')
            else:
                with mrio_params_path.open() as f:
                    mrio_params = json.load(f)
            if pathlib.Path('mrio_sectors_params.json').exists():
                if simulation_params['mrio_params_file'] is not None and simulation_params['mrio_params_file'] != ('mrio_sectors_params.json'):
                    logger.warning("A json file (%s) for MRIO parameters has been found but differs from the file specified in the simulation parameters (%s)", 'mrio_sectors_params.json', simulation_params['mrio_params_file'])
                    logger.warning("Loading the json file")
                    mrio_params_path = pathlib.Path('mrio_sectors_params.json')
                    with mrio_params_path.open() as f:
                        mrio_params = json.load(f)
        if mrio_params is None:
            raise FileNotFoundError("Couldn't load the MRIO params (tried with simulation params and the default file)")

        if isinstance(mrio_system, str):
            mrio_system = pathlib.Path(mrio_system)
        if isinstance(mrio_system, pathlib.Path):
            if not mrio_system.exists():
                raise FileNotFoundError("This file does not exist: ",mrio_system)
            if mrio_system.suffix == '.pkl':
                with mrio_system.open(mode='rb') as fp:
                    mrio = pickle.load(fp)
            else:
                if not mrio_system.is_dir():
                    raise FileNotFoundError("This file should be a pickle file or a directory loadable by pymrio: ", mrio_system.absolute())
                mrio = pym.load_all(mrio_system.absolute())
        elif isinstance(mrio_system, IOSystem):
            mrio = mrio_system
        else:
            raise TypeError("mrio_system must be either a str, a pathlib.Path or an pymrio.IOSystem, not a %s", str(type(mrio_system)))

        if not isinstance(mrio, IOSystem):
            raise TypeError("At this point, mrio should be an IOSystem, not a %s", str(type(mrio)))

        self.params = params
        results_storage = pathlib.Path(self.params['storage_dir']+"/"+self.params['results_storage'])
        if not results_storage.exists():
            results_storage.mkdir()
        self.mrio = MrioSystem(mrio, mrio_params, simulation_params, results_storage)
        self.events = []
        self.events_timings = set()
        self.n_timesteps_to_sim = simulation_params['n_timesteps']
        self.current_t = 0
        self.detailled = False
        self.scheme = 'proportional'

    def loop(self):
        widgets = [
            'Processed: ', progressbar.Counter('Year: %(value)d '), ' ~ ', progressbar.Percentage(), ' ', progressbar.ETA(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets)
        print(json.dumps(self.params, indent=4))
        for t in bar(range(self.n_timesteps_to_sim)):
            assert self.current_t == t
            if not self.next_step():
                logger.warning("Production seems to have crashed, ending simulation")
                break

        self.mrio.rebuild_demand_evolution.flush()
        self.mrio.final_demand_unmet_evolution.flush()
        self.mrio.classic_demand_evolution.flush()
        self.mrio.production_evolution.flush()
        self.mrio.limiting_stocks_evolution.flush()
        self.mrio.rebuild_production_evolution.flush()
        if self.params['register_stocks']:
            self.mrio.stocks_evolution.flush()
        self.mrio.overproduction_evolution.flush()
        self.mrio.production_cap_evolution.flush()
        bar.finish()

    def next_step(self):
        if self.current_t in self.events_timings:
            current_events = [e for e in self.events if e.occurence_time==self.current_t]
            for e in current_events:
                # print(e)
                self.shock(e)
        if self.params['register_stocks']:
            self.mrio.write_stocks(self.current_t)
        self.mrio.write_overproduction(self.current_t)
        self.mrio.write_rebuild_demand(self.current_t)
        self.mrio.write_classic_demand(self.current_t)
        self.mrio.calc_production_cap()
        constraints = self.mrio.calc_production()
        self.mrio.write_limiting_stocks(self.current_t, constraints)
        self.mrio.write_production(self.current_t)
        self.mrio.write_production_max(self.current_t)
        self.mrio.distribute_production(self.current_t, self.scheme)
        self.mrio.calc_orders(constraints)
        self.mrio.calc_overproduction()
        if self.current_t > (self.n_timesteps_to_sim // 5):
            if self.mrio.check_crash() > ((self.mrio.n_sectors*self.mrio.n_regions) / 3):
                return False
        self.current_t+=1
        return True


    def read_events_from_list(self, events_list):
        for ev_dic in events_list:
            if ev_dic['aff-sectors'] == 'all':
                ev_dic['aff-sectors'] = list(self.mrio.sectors)
            ev = Event(ev_dic)
            ev.check_values(self)
            self.events.append(ev)
            self.events_timings.add(ev_dic['occur'])
        with (pathlib.Path(self.params["storage_dir"]+"/"+self.params['results_storage'])/"simulated_events.json").open('w') as f:
            json.dump(events_list, f, indent=4)

    def read_events(self, events_file):
        if not events_file.exists():
            raise FileNotFoundError("This file does not exist: ",events_file)
        else:
            with events_file.open('r') as f:
                events = json.load(f)
        if events['events']:
            for event in events['events']:
                if event['aff-sectors'] == 'all':
                    event['aff-sectors'] = self.mrio.sectors
                ev=Event(event)
                ev.check_values(self)
                self.events.append(ev)
                self.events_timings.add(event['occur'])

    def shock(self, event_to_add):
        """Sets the rebuilding demand and the share of production allocated toward it in the mrio system.

        :param event:
        :returns:

        """
        impacted_region_prod_share = self.params['impacted_region_base_production_toward_rebuilding']
        RoW_prod_share = self.params['row_base_production_toward_rebuilding']
        event_to_add.check_values(self)
        if (impacted_region_prod_share > 1.0 or impacted_region_prod_share < 0.0):
            raise ValueError("Impacted production share should be in [0.0,1.0], (%f)", impacted_region_prod_share)
        if (RoW_prod_share > 1.0 or RoW_prod_share < 0.0):
            raise ValueError("RoW production share should be in [0.0,1.0], (%f)", RoW_prod_share)
        regions_idx = np.arange(self.mrio.regions.size)
        aff_regions_idx = np.searchsorted(self.mrio.regions, event_to_add.aff_regions)
        n_regions_aff = aff_regions_idx.size
        aff_sectors_idx = np.searchsorted(self.mrio.sectors, event_to_add.aff_sectors)
        n_sectors_aff = aff_sectors_idx.size
        aff_industries_idx = np.array([self.mrio.n_sectors * ri + si for ri in aff_regions_idx for si in aff_sectors_idx])
        q_dmg = event_to_add.q_damages / self.mrio.monetary_unit
        # print(f'''
        # regions_idx = {regions_idx},
        # aff_regions_idx = {aff_regions_idx}
        # n_regions_aff = {aff_regions_idx.size}
        # aff_sectors_idx = {aff_sectors_idx}
        # n_sectors_aff = {aff_sectors_idx.size}
        # aff_industries_idx = {aff_industries_idx}
        # q_dmg = {q_dmg}
        # ''')

        # DAMAGE DISTRIBUTION ACROSS REGIONS
        if event_to_add.dmg_distrib_across_regions is None:
            q_dmg_regions = np.array([1.0]) * q_dmg
        elif type(event_to_add.dmg_distrib_across_regions) == list:
            q_dmg_regions = np.array(event_to_add.dmg_distrib_across_regions) * q_dmg
        elif type(event_to_add.dmg_distrib_across_regions) == str and event_to_add.dmg_distrib_across_regions == 'shared':
            q_dmg_regions = np.full(shape=aff_regions_idx.shape,fill_value=q_dmg/aff_regions_idx.size)
        else:
            raise ValueError("This should not happen")
        q_dmg_regions = q_dmg_regions.reshape((n_regions_aff,1))

        #TODO: Check this one !
        # DAMAGE DISTRIBUTION ACROSS SECTORS
        if event_to_add.dmg_distrib_across_sectors_type == "gdp":
            shares = self.mrio.gdp_share_sector.reshape((self.mrio.n_regions,self.mrio.n_sectors))
            q_dmg_regions_sectors = q_dmg_regions * (shares[aff_regions_idx][:,aff_sectors_idx]/shares[aff_regions_idx][:,aff_sectors_idx].sum(axis=1)[:,np.newaxis])
        elif event_to_add.dmg_distrib_across_sectors is None:
            q_dmg_regions_sectors = q_dmg_regions
        elif type(event_to_add.dmg_distrib_across_sectors) == list:
            q_dmg_regions_sectors = q_dmg_regions * np.array(event_to_add.dmg_distrib_across_sectors)
        elif type(event_to_add.dmg_distrib_across_sectors) == str and event_to_add.dmg_distrib_across_sectors == 'GDP':
            shares = self.mrio.gdp_share_sector.reshape((self.mrio.n_regions,self.mrio.n_sectors))
            q_dmg_regions_sectors = q_dmg_regions * (shares[aff_regions_idx][:,aff_sectors_idx]/shares[aff_regions_idx][:,aff_sectors_idx].sum(axis=1)[:,np.newaxis])
        else:
            raise ValueError("damage <-> sectors distribution %s not implemented", event_to_add.dmg_distrib_across_sectors)

        rebuilding_sectors_idx = np.searchsorted(self.mrio.sectors, list(event_to_add.rebuilding_sectors.keys()))
        rebuilding_industries_idx = np.array([self.mrio.n_sectors * ri + si for ri in aff_regions_idx for si in rebuilding_sectors_idx])
        rebuilding_industries_RoW_idx = np.array([self.mrio.n_sectors * ri + si for ri in regions_idx if ri not in aff_regions_idx for si in rebuilding_sectors_idx])
        rebuild_share = np.array([event_to_add.rebuilding_sectors[k] for k in sorted(event_to_add.rebuilding_sectors.keys())])
        rebuilding_demand = np.outer(rebuild_share, q_dmg_regions_sectors)
        new_rebuilding_demand = np.full(self.mrio.Z_0.shape, 0.0)
        mask = np.ix_(np.union1d(rebuilding_industries_RoW_idx, rebuilding_industries_idx), aff_industries_idx)
        new_rebuilding_demand[mask] = self.mrio.Z_distrib[mask] * np.tile(rebuilding_demand, (self.mrio.n_regions,1)) #np.full(self.mrio.Z_0.shape,0.0)
        #new_rebuilding_demand[] = q_dmg_regions_sectors * rebuild_share.reshape(rebuilding_sectors_idx.size,1)
        new_prod_max_toward_rebuilding = np.full(self.mrio.production.shape, 0.0)
        new_prod_max_toward_rebuilding[rebuilding_industries_idx] = impacted_region_prod_share
        new_prod_max_toward_rebuilding[rebuilding_industries_RoW_idx] = RoW_prod_share
        if self.mrio.rebuilding_demand is not None:
            self.mrio.rebuilding_demand = np.append(self.mrio.rebuilding_demand, new_rebuilding_demand[np.newaxis,:], axis=0)
            self.mrio.prod_max_toward_rebuilding = np.append(self.mrio.prod_max_toward_rebuilding, new_prod_max_toward_rebuilding[np.newaxis,:], axis=0)
        else:
            self.mrio.rebuilding_demand = new_rebuilding_demand[np.newaxis,:]
            assert self.mrio.prod_max_toward_rebuilding is None
            self.mrio.prod_max_toward_rebuilding = new_prod_max_toward_rebuilding[np.newaxis,:]

        self.mrio.update_kapital_lost()

    def reset_sim_with_same_events(self):
        self.current_t = 0
        self.mrio.reset_module(self.params, pathlib.Path(self.params['storage_dir']+"/"+self.params['results_storage']))

    def reset_sim_full(self):
        self.reset_sim_with_same_events()
        self.events = []
        self.events_timings = set()

    def update_params(self, new_params):
        self.params = new_params
        results_storage = pathlib.Path(self.params['storage_dir']+"/"+self.params['results_storage'])
        if not results_storage.exists():
            results_storage.mkdir()
        self.mrio.update_params(self.params)

    def write_index(self, index_file):
        self.mrio.write_index(index_file)
