import json
import pickle
import pathlib

import numpy as np
import progressbar
import pymrio as pym

from ario3.event import event as event_module
from ario3.mrio import mrio_module


class Simulation(object):
    def __init__(self,
                 mrio_path:pathlib.Path,
                 params) -> None:
        super().__init__()
        if not mrio_path.exists():
            raise FileNotFoundError("This file does not exist: ",mrio_path)
        if mrio_path.suffix == '.pkl':
            with mrio_path.open(mode='rb') as fp:
                mrio = pickle.load(fp)
        else:
            if not mrio_path.is_dir():
                raise FileNotFoundError("This file should be a pickle file or a directory loadable by pymrio: ", mrio_path)
            mrio = pym.load_all(mrio_path.absolute())

        if not(type(params) == dict):
            if not params.is_dir():
                raise FileNotFoundError("This file should be a directory containing the different simulation parameters files:", params)

            mrio_params_path = params / 'exio_sectors_params.json'
            if not mrio_params_path.exists():
                raise FileNotFoundError("MRIO parameters file not found, it should be here: ",mrio_params_path)
            else:
                with mrio_params_path.open() as f:
                    mrio_params = json.load(f)

            simulation_params_path = params / 'params.json'
            if not simulation_params_path.exists():
                raise FileNotFoundError("Simulation parameters file not found, it should be here: ",simulation_params_path)
            else:
                with simulation_params_path.open() as f:
                    simulation_params = json.load(f)
        else:
            simulation_params = params
            mrio_params_path = pathlib.Path(simulation_params['exio_params_file'])
            if not mrio_params_path.exists():
                raise FileNotFoundError("MRIO parameters file not found, it should be here: ",mrio_params_path)
            else:
                with mrio_params_path.open() as f:
                    mrio_params = json.load(f)

        self.params = simulation_params
        results_storage = pathlib.Path(self.params['results_storage'])
        if not results_storage.exists():
            results_storage.mkdir()
        self.mrio = mrio_module.Mrio_System(mrio, mrio_params, simulation_params, results_storage) #type: ignore
        self.events = []
        self.events_timings = set()
        self.n_timesteps_to_sim = simulation_params['n_timesteps']
        self.current_t = 0
        self.detailled = False
        self.scheme = 'proportional'

    #Not so much actually
    def loop_fast(self):
        widgets = [
            'Processed: ', progressbar.Counter('Year: %(value)d '), ' ~ ', progressbar.Percentage(), ' ', progressbar.ETA(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets)
        for t in bar(range(self.n_timesteps_to_sim)):
            assert self.current_t == t
            self.next_step_fast()
        bar.finish()

    def loop(self):
        widgets = [
            'Processed: ', progressbar.Counter('Year: %(value)d '), ' ~ ', progressbar.Percentage(), ' ', progressbar.ETA(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets)
        for t in bar(range(self.n_timesteps_to_sim)):
            assert self.current_t == t
            self.next_step()
        bar.finish()

    def loop_test(self):
        widgets = [
            'Processed: ', progressbar.Counter('Year: %(value)d '), ' ~ ', progressbar.Percentage(), ' ', progressbar.ETA(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets)
        for t in bar(range(self.n_timesteps_to_sim)):
            assert self.current_t == t
            self.next_step_test()
        bar.finish()


    def next_step_fast(self):
        if self.current_t in self.events_timings:
            current_events = [e for e in self.events if e.occurence_time==self.current_t]
            for e in current_events:
                self.shock(e)
        self.mrio.write_overproduction(self.current_t)
        self.mrio.write_rebuild_demand(self.current_t)
        self.mrio.write_classic_demand(self.current_t)
        self.mrio.calc_production_cap_fast()
        self.mrio.calc_production_fast()
        self.mrio.write_production(self.current_t)
        self.mrio.write_production_max(self.current_t)
        self.mrio.calc_orders_fast()

        self.mrio.distribute_production(self.current_t, self.scheme)
        self.mrio.calc_overproduction_fast()
        self.current_t+=1

    def next_step_test(self):
        if self.current_t in self.events_timings:
            current_events = [e for e in self.events if e.occurence_time==self.current_t]
            for e in current_events:
                self.shock(e)
        self.mrio.write_overproduction(self.current_t)
        self.mrio.write_rebuild_demand(self.current_t)
        self.mrio.write_classic_demand(self.current_t)
        self.mrio.calc_production_cap()
        self.mrio.calc_production()
        self.mrio.write_production(self.current_t)
        self.mrio.write_production_max(self.current_t)
        self.mrio.calc_orders()

        self.mrio.distribute_production(self.current_t, self.scheme)
        self.mrio.calc_overproduction()
        self.current_t+=1

    def next_step(self):
        if self.current_t in self.events_timings:
            current_events = [e for e in self.events if e.occurence_time==self.current_t]
            for e in current_events:
                self.shock(e)
        self.mrio.write_stocks(self.current_t)
        self.mrio.write_overproduction(self.current_t)
        self.mrio.write_rebuild_demand(self.current_t)
        self.mrio.write_classic_demand(self.current_t)
        self.mrio.calc_production_cap()
        constraints = self.mrio.calc_production()
        self.mrio.write_limiting_stocks(self.current_t, constraints)
        self.mrio.write_production(self.current_t)
        self.mrio.write_production_max(self.current_t)
        self.mrio.calc_orders(constraints)

        self.mrio.distribute_production(self.current_t, self.scheme)
        self.mrio.calc_overproduction()
        self.current_t+=1

    def read_events_from_list(self, events_list):
        for ev_dic in events_list:
            ev = event_module.Event(ev_dic)
            ev.check_values(self)
            self.events.append(ev)
            self.events_timings.add(ev_dic['occur'])

    def read_events(self, events_file):
        if not events_file.exists():
            raise FileNotFoundError("This file does not exist: ",events_file)
        else:
            with events_file.open('r') as f:
                events = json.load(f)
        if events['events']:
            for event in events['events']:
                ev=event_module.Event(event)
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
        if event_to_add.dmg_distrib_across_sectors is None:
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
