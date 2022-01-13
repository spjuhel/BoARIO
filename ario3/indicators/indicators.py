from ario3.simulation.base import Simulation
import json
import pathlib
import numpy as np
import pandas as pd
import itertools
from ario3.utils import misc

class Indicators(object):
    def __init__(self, data_dict) -> None:
        super().__init__()
        #if indexes['fd_cat'] is None:
        #    indexes['fd_cat'] = np.array(["Final demand"])

        steps = [i for i in range(data_dict["n_timesteps_to_sim"])]

        prod_df = pd.DataFrame(data_dict["prod"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['prod']
        prodmax_df = pd.DataFrame(data_dict["prodmax"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['prodmax']
        overprod_df = pd.DataFrame(data_dict["overprod"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['overprod']
        c_demand_df = pd.DataFrame(data_dict["c_demand"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['c_demand']
        r_demand_df = pd.DataFrame(data_dict["r_demand"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['r_demand']
        r_prod_df = pd.DataFrame(data_dict["r_prod"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['r_prod']
        fd_unmet_df = pd.DataFrame(data_dict["fd_unmet"], columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['fd_unmet']
        #print(data_dict["stocks_evolution"].shape)
        #stocks_df = pd.DataFrame(data_dict["stocks"].reshape(data_dict["n_timesteps_to_sim"]*data_dict["n_sectors"],-1),
        #                         index=pd.MultiIndex.from_product([steps, data_dict["sectors"]], names=['step', 'stock of']),
        #                         columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        del data_dict['stocks']
        #stocks_df.index = pd.MultiIndex.from_product([steps, data_dict["sectors"]], names=['step', 'stock of'])
        prod_df['step'] = prod_df.index
        prodmax_df['step'] = prodmax_df.index
        overprod_df['step'] = overprod_df.index
        c_demand_df['step'] = c_demand_df.index
        r_demand_df['step'] = r_demand_df.index
        r_prod_df['step'] = r_prod_df.index
        fd_unmet_df['step'] = fd_unmet_df.index
        df = prod_df.copy().set_index('step').melt(ignore_index=False)
        del prod_df
        df=df.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'production'})
        df['demand'] = c_demand_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'demand'})['demand']
        del c_demand_df
        df['rebuild_demand'] = r_demand_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'rebuild_demand'})['rebuild_demand']
        del r_demand_df
        df['production_max'] = prodmax_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'production_max'})['production_max']
        del prodmax_df
        df['rebuild_production'] = r_prod_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'rebuild_production'})['rebuild_production']
        del r_prod_df
        df['overprod'] = overprod_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'overprod'})['overprod']
        del overprod_df

        df = df.reset_index().melt(id_vars=['region', 'sector', 'step'])
        self.df=df
        del df
        #stocks_df = stocks_df.astype(np.float32).reset_index()
        #stocks_df['step'] = stocks_df['step'].astype("uint8")
        #stocks_df['stock of'] = stocks_df['stock of'].astype("category")
        #stocks_df = stocks_df.set_index(['step', 'stock of'])
        #stocks_df = stocks_df.pct_change().fillna(0).add(1).cumprod().sub(1).melt(ignore_index=False).rename(columns={
        #    'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'})
        #stocks_df['region'] = stocks_df['region'].astype("category")
        #stocks_df['sector'] = stocks_df['sector'].astype("category")

        df_loss = fd_unmet_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'fd_cat', 'value':'fdloss'}).reset_index()
        self.df_loss = df_loss
        del df_loss
        self.df_limiting = pd.DataFrame(data_dict["limiting_stocks"].reshape(data_dict["n_timesteps_to_sim"]*data_dict["n_sectors"],-1),
                                        index=pd.MultiIndex.from_product([steps, data_dict["sectors"]], names=['step', 'stock of']),
                                        columns=pd.MultiIndex.from_product([data_dict["regions"], data_dict["sectors"]]))
        #self.df_limiting.index = pd.MultiIndex.from_product([steps, data_dict["sectors"]], names=['step', 'stock of'])
        self.aff_regions = []
        for e in data_dict["events"]:
            self.aff_regions.append(e['aff-regions'])

        self.aff_regions = list(misc.flatten(self.aff_regions))

        self.aff_sectors = []
        for e in data_dict["events"]:
            self.aff_sectors.append(e['aff-sectors'])

        self.aff_sectors = list(misc.flatten(self.aff_sectors))
        self.indicators = {}
        self.storage = pathlib.Path(data_dict['results_storage'])/'indicators.json'

    @classmethod
    def from_model(cls, model : Simulation):
        data_dict = {}
        data_dict["n_timesteps_to_sim"] = model.n_timesteps_to_sim
        data_dict["regions"] = model.mrio.regions
        data_dict["sectors"] = model.mrio.sectors
        with (pathlib.Path(model.params["results_storage"])/".simulated_events.json").open() as f:
            events = json.load(f)

        data_dict["events"] = events
        data_dict["prod"] = model.mrio.production_evolution
        data_dict["prodmax"] = model.mrio.production_cap_evolution
        data_dict["overprod"] = model.mrio.overproduction_evolution
        data_dict["c_demand"] = model.mrio.classic_demand_evolution
        data_dict["r_demand"] = model.mrio.rebuild_demand_evolution
        data_dict["r_prod"] = model.mrio.rebuild_production_evolution
        data_dict["fd_unmet"] = model.mrio.final_demand_unmet_evolution
        data_dict["stocks"] = model.mrio.stocks_evolution
        data_dict["limiting_stocks"] = model.mrio.limiting_stocks_evolution
        return cls(data_dict)

    @classmethod
    def from_storage_path(cls, storage_path):
        data_dict = {}
        if not isinstance(storage_path, pathlib.Path):
            storage_path = pathlib.Path(storage_path)
            assert storage_path.exists(), str("Directory does not exist:"+str(storage_path))
        results_path = storage_path/"results"
        with (storage_path/"params.json").open() as f:
            simulation_params = json.load(f)
        with (storage_path/"indexes"/"indexes.json").open() as f:
            indexes = json.load(f)
        with (storage_path/"results"/"simulated_events.json").open() as f:
            events = json.load(f)
        t = simulation_params["n_timesteps"]
        if indexes['fd_cat'] is None:
            indexes['fd_cat'] = np.array(["Final demand"])

        data_dict["results_storage"] = results_path
        data_dict["n_timesteps_to_sim"] = t
        data_dict["regions"] = indexes["regions"]
        data_dict["n_regions"] = indexes["n_regions"]
        data_dict["sectors"] = indexes["sectors"]
        data_dict["n_sectors"] = indexes["n_sectors"]
        data_dict["events"] = events
        data_dict["prod"] = np.memmap(results_path/"iotable_XVA_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["prodmax"] = np.memmap(results_path/"iotable_X_max_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["overprod"] = np.memmap(results_path/"overprodvector_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["c_demand"] = np.memmap(results_path/"classic_demand_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["r_demand"] = np.memmap(results_path/"rebuild_demand_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["r_prod"] = np.memmap(results_path/"rebuild_prod_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["fd_unmet"] = np.memmap(results_path/"final_demand_unmet_record", mode='r+', dtype='float64',shape=(t,indexes['n_regions']*indexes['n_sectors']))
        data_dict["stocks"] = np.memmap(results_path/"stocks_record", mode='r+', dtype='float64',shape=(t*indexes['n_sectors'],indexes['n_industries']))
        data_dict["limiting_stocks"] = np.memmap(results_path/"limiting_stocks_record", mode='r+', dtype='bool',shape=(t*indexes['n_sectors'],indexes['n_industries']))
        return cls(data_dict)

    @classmethod
    def dict_from_storage_path(cls, storage_path):
        data_dict = {}
        if not isinstance(storage_path, pathlib.Path):
            storage_path = pathlib.Path(storage_path)
            assert storage_path.exists(), str("Directory does not exist:"+str(storage_path))
        results_path = storage_path/"results"
        with (storage_path/"params.json").open() as f:
            simulation_params = json.load(f)
        with (storage_path/"indexes"/"indexes.json").open() as f:
            indexes = json.load(f)
        with (storage_path/"results"/"simulated_events.json").open() as f:
            events = json.load(f)
        t = simulation_params["n_timesteps"]
        if indexes['fd_cat'] is None:
            indexes['fd_cat'] = np.array(["Final demand"])

        data_dict["n_timesteps_to_sim"] = t
        data_dict["regions"] = indexes["regions"]
        data_dict["n_regions"] = indexes["n_regions"]
        data_dict["sectors"] = indexes["sectors"]
        data_dict["n_sectors"] = indexes["n_sectors"]
        data_dict["events"] = events
        data_dict["prod"] = np.memmap(results_path/"iotable_XVA_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["prodmax"] = np.memmap(results_path/"iotable_X_max_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["overprod"] = np.memmap(results_path/"overprodvector_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["c_demand"] = np.memmap(results_path/"classic_demand_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["r_demand"] = np.memmap(results_path/"rebuild_demand_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["r_prod"] = np.memmap(results_path/"rebuild_prod_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
        data_dict["fd_unmet"] = np.memmap(results_path/"final_demand_unmet_record", mode='r+', dtype='float64',shape=(t,indexes['n_regions']*indexes['n_sectors']))
        data_dict["stocks"] = np.memmap(results_path/"stocks_record", mode='r+', dtype='float64',shape=(t*indexes['n_sectors'],indexes['n_industries']))
        data_dict["limiting_stocks"] = np.memmap(results_path/"limiting_stocks_record", mode='r+', dtype='bool',shape=(t*indexes['n_sectors'],indexes['n_industries']))
        return data_dict

    def calc_tot_fd_unmet(self):
        self.indicators['tot_fd_unmet'] = self.df_loss['fdloss'].sum()

    def calc_aff_fd_unmet(self):
        self.indicators['aff_fd_unmet'] = self.df_loss[self.df_loss.region.isin(self.aff_regions)]['fdloss'].sum()

    def calc_rebuild_durations(self):
        rebuilding = (self.df[self.df['variable']=='rebuild_demand'].groupby('step').sum().ne(0)).value.to_numpy()
        self.indicators['rebuild_durations'] = [ sum( 1 for _ in group ) for key, group in itertools.groupby( rebuilding ) if key ]

    def calc_recovery_duration(self):
        pass

    def calc_general_shortage(self):
        #TODO
        a = self.df_limiting.T.stack(level=0)
        a.index = a.index.rename(['region','sector', 'step'])#.sum(axis=1).groupby(['step','region','sector']).sum()/8
        b = a.sum(axis=1).groupby(['step','region','sector']).sum()/8
        c = b.groupby(['step','region']).sum()/8
        c = c.groupby('step').sum()/6
        if not c.ne(0).any():
            self.indicators['shortage_b'] = False
        else:
            self.indicators['shortage_b'] = True
            shortage_date_start = c.ne(0.0).argmax()
            self.indicators['shortage_date_start'] = shortage_date_start
            shortage_date_end = c.iloc[shortage_date_start:].eq(0).argmax()+shortage_date_start
            self.indicators['shortage_date_end'] = shortage_date_end
            self.indicators['shortage_date_max'] = c.argmax()
            self.indicators['shortage_ind_max'] = c.max()
            self.indicators['shortage_ind_mean'] = c.iloc[shortage_date_start:shortage_date_end].mean()

    def calc_tot_prod_change(self):
        df2=self.df[self.df.variable=="production"].set_index(['step','region','sector']).drop(['variable'],axis=1).unstack([1,2])
        df2.columns=df2.columns.droplevel(0)
        prod_chg = df2 - df2.iloc[1,:]
        prod_chg = prod_chg.round(6)
        self.indicators['prod_gain_tot'] = prod_chg.mul(prod_chg.gt(0)).sum().sum()
        self.indicators['prod_lost_tot'] = prod_chg.mul(~prod_chg.gt(0)).sum().sum()
        prod_chg = prod_chg.drop(self.aff_regions, axis=1)
        self.indicators['prod_gain_unaff'] = prod_chg.mul(prod_chg.gt(0)).sum().sum()
        self.indicators['prod_lost_unaff'] = prod_chg.mul(~prod_chg.gt(0)).sum().sum()

    def update_indicators(self):
        self.calc_tot_fd_unmet()
        self.calc_aff_fd_unmet()
        self.calc_rebuild_durations()
        self.calc_recovery_duration()
        self.calc_general_shortage()
        self.calc_tot_prod_change()

    def write_indicators(self):
        self.update_indicators()
        with self.storage.open('w') as f:
            json.dump(self.indicators, f)
