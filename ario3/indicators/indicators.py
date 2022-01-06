from ario3.simulation.base import Simulation
import numpy as np
import pandas as pd
import itertools

class Indicators(object):
    def __init__(self, model: Simulation) -> None:
        super().__init__()
        #if indexes['fd_cat'] is None:
        #    indexes['fd_cat'] = np.array(["Final demand"])

        steps = [i for i in range(model.n_timesteps_to_sim)]

        prod_df = pd.DataFrame(model.mrio.production_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        prodmax_df = pd.DataFrame(model.mrio.production_cap_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        overprod_df = pd.DataFrame(model.mrio.overproduction_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        c_demand_df = pd.DataFrame(model.mrio.classic_demand_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        r_demand_df = pd.DataFrame(model.mrio.rebuild_demand_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        r_prod_df = pd.DataFrame(model.mrio.rebuild_production_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        fd_unmet_df = pd.DataFrame(model.mrio.final_demand_unmet_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.fd_cat]))
        stocks_df = pd.DataFrame(model.mrio.stocks_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        stocks_df.index = pd.MultiIndex.from_product([steps, model.mrio.sectors], names=['step', 'stock of'])


        prod_df['step'] = prod_df.index
        prodmax_df['step'] = prodmax_df.index
        overprod_df['step'] = overprod_df.index
        c_demand_df['step'] = c_demand_df.index
        r_demand_df['step'] = r_demand_df.index
        r_prod_df['step'] = r_prod_df.index
        fd_unmet_df['step'] = fd_unmet_df.index

        df = prod_df.set_index('step').melt(ignore_index=False)
        df=df.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'production'})
        df['demand'] = c_demand_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'demand'})
        df['rebuild_demand'] = r_demand_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'rebuild_demand'})
        df['production_max'] = prodmax_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'production_max'})
        df['rebuild_production'] = r_prod_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'rebuild_production'})
        df['overprod'] = overprod_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'value':'overprod'})
        df = df.reset_index().melt(id_vars=['region', 'sector', 'step'])
        df_stocks = (stocks_df.unstack(level=1)).pct_change().fillna(0).add(1).cumprod().sub(1).melt(ignore_index=False).rename(columns={
            'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'}).reset_index()

        df_loss = fd_unmet_df.set_index('step').melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'fd_cat', 'value':'fdloss'}).reset_index()

        self.df_limiting = pd.DataFrame(model.mrio.limiting_stocks_evolution, columns=pd.MultiIndex.from_product([model.mrio.regions, model.mrio.sectors]))
        self.df_limiting.index = pd.MultiIndex.from_product([steps, model.mrio.sectors], names=['step', 'stock of'])

        self.aff_regions = []
        for e in model.events:
            self.aff_regions.append(e.aff_regions)

        self.aff_sectors = []
        for e in model.events:
            self.aff_sectors.append(e.aff_sectors)

        self.df=df
        self.df_stocks=df_stocks
        self.df_loss = df_loss
        self.indicators = {}

    def calc_tot_fd_unmet(self):
        self.indicators['tot_fd_unmet'] = self.df_loss['fdloss'].sum()

    def calc_aff_fd_unmet(self):
        self.indicators['aff_fd_unmet'] = self.df_loss[self.df_loss.region.isin(self.aff_regions)]['fdloss'].sum()

    def rebuild_durations(self):
        rebuilding = (self.df[self.df['variable']=='rebuild_demand'].groupby('step').sum().ne(0)).value.to_numpy()
        self.indicators['rebuild_durations'] = [ sum( 1 for _ in group ) for key, group in itertools.groupby( rebuilding ) if key ]

    def general_shortage(self):
        #TODO
        a = self.df_limiting.set_index(['step','region','sector','stock of'])
        a = a.unstack()
        b = a.sum(axis=1).groupby(['step','region','sector']).sum()/8
        c = b.groupby(['step','region']).sum()/8
        c = c.groupby('step').sum()/6

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
