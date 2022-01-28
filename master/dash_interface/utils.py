import pathlib
#print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import pandas as pd
import numpy as np
import json

from defaults import WORKING_DIR

def load_files(results_path, indexes, tot_time):
    t = tot_time
    if indexes['fd_cat'] is None:
        indexes['fd_cat'] = np.array(["Final demand"])
    prod = np.memmap(results_path/"iotable_XVA_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
    prodmax = np.memmap(results_path/"iotable_X_max_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
    overprod = np.memmap(results_path/"overprodvector_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
    c_demand = np.memmap(results_path/"classic_demand_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
    r_demand = np.memmap(results_path/"rebuild_demand_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
    r_prod = np.memmap(results_path/"rebuild_prod_record", mode='r+', dtype='float64',shape=(t,indexes['n_industries']))
    fd_unmet = np.memmap(results_path/"final_demand_unmet_record", mode='r+', dtype='float64',shape=(t,indexes['n_regions']*len(indexes['fd_cat'])))
    stocks = np.memmap(results_path/"stocks_record", mode='r+', dtype='float64',shape=(t*indexes['n_sectors'],indexes['n_industries']))
    limiting_stocks = np.memmap(results_path/"limiting_stocks_record", mode='r+', dtype='bool',shape=(t*indexes['n_sectors'],indexes['n_industries']))

    all_sectors = indexes['sectors']

    sectors = list(all_sectors)
    steps = [i for i in range(tot_time)]

    prod_df = pd.DataFrame(prod, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    prodmax_df = pd.DataFrame(prodmax, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    overprod_df = pd.DataFrame(overprod, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    c_demand_df = pd.DataFrame(c_demand, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    r_demand_df = pd.DataFrame(r_demand, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    r_prod_df = pd.DataFrame(r_prod, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    fd_unmet_df = pd.DataFrame(fd_unmet, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['fd_cat']]))
    stocks_df = pd.DataFrame(stocks, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    stocks_df.index = pd.MultiIndex.from_product([steps, sectors], names=['step', 'stock of'])
    limiting_stocks_df = pd.DataFrame(limiting_stocks, columns=pd.MultiIndex.from_product([indexes['regions'], indexes['sectors']]))
    limiting_stocks_df.index = pd.MultiIndex.from_product([steps, sectors], names=['step', 'stock of'])

    files = {
        "production":prod_df,
        "rebuild_production":r_prod_df,
        "production_max":prodmax_df,
        "demand":c_demand_df,
        "rebuild_demand":r_demand_df,
        "overprod":overprod_df,
        "fdloss":fd_unmet_df,
        "stocks":stocks_df,
        "limiting":limiting_stocks_df
    }
    #TODO: implement this
    #if os.path.isfile(results_path+scenario+"/stocks.csv"):
    #    files['stocks'] = pd.read_csv(results_path+scenario+"/stocks.csv", skiprows=2, names=pd.MultiIndex.from_product([countries, sectors], names=['regions', 'sectors']))
    #    files['stocks'].index = pd.MultiIndex.from_product([steps, sectors], names=['steps','stock of'])
    #    files['stocks'].name = "Stocks - "
    #if os.path.isfile(results_path+scenario+"/limiting_stocks.csv"):
    #    files['limiting'] = pd.read_csv(results_path+scenario+"/limiting_stocks.csv", skiprows=2, names=pd.MultiIndex.from_product([countries, sectors], names=['regions', 'sectors']))
    #    files['limiting'].index = pd.MultiIndex.from_product([steps, sectors], names=['steps','stock of'])
    #    files['limiting'].name = "Limiting stocks - "

    files["production"].name="Prod - "
    files["rebuild_production"].name="Rebuild prod - "
    files["production_max"].name="ProdMax - "
    files["demand"].name="\"Classic\" Demand - "
    files["rebuild_demand"].name="Rebuild demand - "
    files["overprod"].name="Overprod - "
    files["fdloss"].name="FD Losses - "
    return files

def load_dfs(params, index=False):
    #print("load_dfs::params: ",params)
    storage = WORKING_DIR/params["storage_dir"]
    n_time = params["n_timesteps"]
    if not (storage/"indexes/indexes.json").exists():
        print("Failure - indexes not present")
    with (storage/"indexes/indexes.json").open('r') as f:
        indexes = json.load(f)
    if index:
        storage = storage/"indexes/"
    else:
        storage = WORKING_DIR/params["results_storage"]
    #print("load_dfs::storage: ",storage)
    try:
        print(storage)
        print(n_time)
        files = load_files(storage, indexes, n_time)
    except FileNotFoundError:
        return {"loaded":"-1"}, (-1), (-1)
    files['demand']['step']=files['demand'].index
    files['rebuild_production']['step']=files['rebuild_production'].index
    files['production_max']['step']=files['production_max'].index
    files['rebuild_demand']['step']=files['rebuild_demand'].index
    files['overprod']['step']=files['overprod'].index
    #.pct_change().fillna(0).add(1).cumprod().sub(1)
    files['production']['step']=files['production'].index
    df = files['production'].set_index('step').melt(ignore_index=False)
    df=df.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'production'})
    df2 = files['demand'].set_index('step').melt(ignore_index=False)
    df2=df2.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'demand'})
    df['demand']=df2['demand']
    df2 = files['rebuild_production'].set_index('step').melt(ignore_index=False)
    df2=df2.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'rebuild_production'})
    df['rebuild_production']=df2['rebuild_production']
    df2 = files['production_max'].set_index('step').melt(ignore_index=False)
    df2=df2.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'production_max'})
    df['production_max']=df2['production_max']
    df2 = files['rebuild_demand'].set_index('step').melt(ignore_index=False)
    df2=df2.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'rebuild_demand'})
    df['rebuild_demand']=df2['rebuild_demand']
    df2 = files['overprod'].set_index('step').melt(ignore_index=False)
    df2=df2.rename(columns={'variable_0':'region','variable_1':'sector', 'value':'overprod'})
    df['overprod']=df2['overprod']
    df = df.reset_index().melt(id_vars=['region', 'sector', 'step'])
    #df_stocks = (files['stocks'].unstack(level=1)).pct_change().fillna(0).add(1).cumprod().sub(1).melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'}).reset_index()
    stocks_df = files['stocks'].copy().astype(np.float32).reset_index()
    del files['stocks']
    stocks_df['step'] = stocks_df['step'].astype("uint8")
    stocks_df['stock of'] = stocks_df['stock of'].astype("category")
    stocks_df = stocks_df.set_index(['step', 'stock of'])
    stocks_df = stocks_df.pct_change().fillna(0).add(1).cumprod().sub(1).melt(ignore_index=False).rename(columns={
       'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'}).reset_index()
    stocks_df['region'] = stocks_df['region'].astype("category")
    stocks_df['sector'] = stocks_df['sector'].astype("category")


    return {"loaded":"1"}, df, stocks_df

def load_indexes(params):
    if (pathlib.Path(params["storage_dir"])/"indexes"/"indexes.json").exists():
        with (pathlib.Path(params["storage_dir"])/"indexes"/"indexes.json").open('r') as fp:
            return json.load(fp)

def load_mrio_params(params):
    if (pathlib.Path(params["storage_dir"])/"mrio_params.json").exists():
        with (pathlib.Path(params["storage_dir"])/"mrio_params.json").open('r') as fp:
            return json.load(fp)
