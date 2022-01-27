import pandas as pd
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

    all_sectors = indexes['sectors']
    all_countries = indexes['regions']

    sectors = list(all_sectors)
    countries = list(all_countries)
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

    files = {
        "production":prod_df,
        "rebuild_production":r_prod_df,
        "production_max":prodmax_df,
        "demand":c_demand_df,
        "rebuild_demand":r_demand_df,
        "overprod":overprod_df,
        "fdloss":fd_unmet_df,
        "stocks":stocks_df
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

import datetime
import logging
logging.basicConfig(filename='log.log', level=logging.DEBUG)

class EnterExitLog():
    def __init__(self, funcName):
        self.funcName = funcName

    def __enter__(self):
        logging.debug('Started: %s' % self.funcName)
        self.init_time = datetime.datetime.now()
        return self

    def __exit__(self, type, value, tb):
        logging.debug('Finished: %s in: %s seconds' % (self.funcName, datetime.datetime.now() - self.init_time))

def func_timer_decorator(func):
    def func_wrapper(*args, **kwargs):
        with EnterExitLog(func.__name__):
            return func(*args, **kwargs)

    return func_wrapper


"""
A simple app demonstrating how to dynamically render tab content containing
dcc.Graph components to ensure graphs get sized correctly. We also show how
dcc.Store can be used to cache the results of an expensive graph generation
process so that switching tabs is fast.
"""
import time
import ast
import json
import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash import dcc, html, ALL, MATCH, ALLSMALLER
from dash.exceptions import PreventUpdate
from flask_caching import Cache
import pathlib
import pickle as pkl
from ario3.simulation import Simulation
import os

from dash_extensions.enrich import DashProxy, ServersideOutputTransform, ServersideOutput, Output, Input, State, Trigger

app = DashProxy(external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
                transforms=[
    ServersideOutputTransform(),  # enable use of ServersideOutput objects
                    ])

#cache = Cache(app.server, config={
#    'CACHE_TYPE': 'filesystem',
#    'CACHE_DIR': 'cache-directory'
#})

TIMEOUT = 3600

cwd = pathlib.Path.cwd()

default_params = { "storage_dir": "dash-testing/",
            "bool_run_detailled": True,
            "psi_param": 0.8,
            "model_time_step": 1,
            "timestep_dividing_factor": 365,
            "inventory_restoration_time": 40,
            "alpha_base": 1.0,
            "alpha_max": 1.25,
            "alpha_tau": 365,
            "rebuild_tau": 30,
            "n_timesteps": 365,
            "min_duration": (365 // 100) * 25,
            "impacted_region_base_production_toward_rebuilding": 0.01,
            "row_base_production_toward_rebuilding": 0.0,
            "exio_params_file":str(cwd/"dash-testing"/"mrio_params.json")
}

if (cwd/default_params["storage_dir"]/"indexes"/"indexes.json").exists():
    with (cwd/default_params["storage_dir"]/"indexes"/"indexes.json").open('r') as fp:
        indexes = json.load(fp)

if (cwd/default_params["storage_dir"]/"params.json").exists():
    with (cwd/default_params["storage_dir"]/"params.json").open('r') as fp:
        default_params = json.load(fp)


#@cache.memoize(timeout=TIMEOUT)
@func_timer_decorator
def load_dfs(params, index=False):
    #print("load_dfs::params: ",params)
    storage = cwd/params["storage_dir"]
    n_time = params["n_timesteps"]
    if not (storage/"indexes/indexes.json").exists():
        print("Failure - indexes not present")
    with (storage/"indexes/indexes.json").open('r') as f:
        indexes = json.load(f)
    if index:
        storage = storage/"indexes/"
    else:
        storage = cwd/params["results_storage"]
    #print("load_dfs::storage: ",storage)
    files = load_files(storage, indexes, n_time)
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
    df_stocks = (files['stocks'].unstack(level=1)).pct_change().fillna(0).add(1).cumprod().sub(1).melt(ignore_index=False).rename(columns={'variable_0':'region','variable_1':'sector', 'variable_2':'stock of'}).reset_index()
    return df, df_stocks

#df, df_stocks = load_dfs(default_params, index=True)
@app.callback(ServersideOutput("df-store", "data"), ServersideOutput("df-stocks-store", "data"), Input("load-data-button", "n_clicks"), State("params-store","data"), memoize=True)
def load_df_server(_, params):
    return load_dfs(params, index=False)

#server = app.server

@func_timer_decorator
def render_events_tab(params, events={}):
    res = html.Div(children = [
        dbc.Button(
                    "Add another event",
                    id="add-event",
                    n_clicks=0,
                    style={"display": "inline-block"},
                ),
        html.Br(),
        html.Div(id="event-container", children=[]),
        html.Br(),
        dbc.Button(
                    "Register events",
                    id="register-events",
                    n_clicks=0,
                    style={"display": "inline-block"},
                    className="d-grid gap-2 d-md-flex justify-content-md-end"
                ),
        html.H2(children=["Launch simulation : ",
                          dbc.Button(
                              "Yes !",
                              color="primary",
                              id="launch-sim-button"
                          ),
                          html.Div(id="sim-badge")
                         ], className="d-grid gap-2 d-md-flex justify-content-md-end"),
    ])
    return res

@func_timer_decorator
def event_controls(uid, params, indexes):
    res = dbc.Card(children=[
        html.Div(children=[
            html.Label("Day of occurence:"),
            dcc.Slider(
                min=0.0,
                max=params['n_timesteps'],
                value=0,
                tooltip={"placement": "bottom", "always_visible": True},
                id={
                    'event-id': str(uid[0]),
                    'attribute': 'occur'
                },
                included=False,
                persistence=True,
                persistence_type='local',
            ),
        ]),
        html.Br(),
        html.Div(children=[
            dbc.InputGroup([
                dbc.InputGroupText("Total direct damages:"),
                dbc.InputGroupText("$"),
                dbc.Input(
                    id={
                        'event-id': str(uid[0]),
                        'attribute': 'q_dmg'
                    },
                    type="number", placeholder="Amount",
                    persistence=True,
                    persistence_type='local',
                ),
            ], className="mb-3" ),
        ]),
        html.Div(children=[
            dbc.InputGroup([
                dbc.InputGroupText("Duration:"),
                dbc.InputGroupText("days"),
                dbc.Input(
                    id={
                        'event-id': str(uid[0]),
                        'attribute': 'duration'
                    },
                    type="number",
                    placeholder="Amount",
                    persistence=True,
                    persistence_type='local',
                ),
            ], className="mb-3" ),
        ]),
        html.Div(style={'width':'auto'}, children=[
            dbc.InputGroup([
                dbc.InputGroupText("Regions affected"),
                dcc.Dropdown(
                    id={
                    'event-id': str(uid[0]),
                    'attribute': 'aff-regions'
                    },
                    value=[indexes['regions'][0]],
                    options=[
                        {'label': c, 'value': c}
                        for c in indexes['regions']
                    ],
                    multi=True,
                    persistence=True,
                    persistence_type='local',
                ),], className="mb-3",),

            dbc.InputGroup([
                dbc.InputGroupText("Affected sectors"),
                dcc.Dropdown(
                    id={
                    'event-id': str(uid[0]),
                    'attribute': 'aff-sectors'
                    },
                    value=[indexes['sectors'][0]],
                    options=[
                        {'label': c, 'value': c}
                        for c in indexes['sectors']
                    ],
                    multi=True,
                    persistence=True,
                    persistence_type='local',
                ),
            ],
            className="mb-3",),
        ]),
        html.Div([
            dbc.Label("Damage distribution across regions"),
            dbc.RadioItems(
                options=[
                    {"label": "Equally shared", "value": "shared"},
                    {"label": "Custom", "value": "custom"}
                ],
                value="shared",
                id={"event-id":str(uid[0]), "attribute":"dmg-distrib-regions-select"},
                persistence=True,
                persistence_type='local',
            ),
        ]),
        html.Div(id={"event-id":str(uid[0]), "attribute":'dmg-distrib-regions-custom'}),
        html.Div([
            dbc.Label("Damage distribution across sectors"),
            dbc.RadioItems(
                options=[
                    {"label": "GDP based", "value": "gdp"},
                    {"label": "Custom", "value": "custom"}
                ],
                value="gdp",
                id={"event-id":str(uid[0]), "attribute":"dmg-distrib-sectors-select"},
                persistence=True,
                persistence_type='local',
            ),
        ]),
        html.Div(id={"event-id":str(uid[0]), "attribute":'dmg-distrib-sectors-custom'}),
        html.Div([
            dbc.InputGroup([
                    dbc.InputGroupText("Rebuilding sectors"),
                    dcc.Dropdown(
                        id={
                            'event-id': str(uid[0]),
                            'attribute': 'rebuilding-sectors'
                        },
                        value=[[indexes['sectors'][0]]],
                        options=[
                            {'label': c, 'value': c}
                            for c in indexes['sectors']
                        ],
                        multi=True,
                        persistence=True,
                        persistence_type='local',
                    ),
                ],
                className="mb-3",),
        ]),
        html.Div(id={"event-id":str(uid[0]), "attribute":'rebuilding-distrib-sectors-custom'}),
    ], body=True)
    return res

#event_template =

@func_timer_decorator
def params_controls(params=default_params):
    res = html.Div(children=[
        dbc.Card(children=[
        html.Div(children=[
                dbc.Label("Working dir ?"),
                dbc.Input(id="local_storage",
                          type="text",
                          value=params['storage_dir'],
                          persistence=True,
                          persistence_type='local',
                         ),
            ]
        ),
        html.Div(children=[
                dbc.Label("Simulation duration (days)"),
                dbc.Input(id="n_timesteps", type="number", value=params['n_timesteps'],
                persistence=True,
                persistence_type='local',),
            ]
        ),
        html.Div(children=[
                dbc.Label("Inventory restoration tau"),
                dbc.Input(id="inventory_restoration_time", type="number", value=params['inventory_restoration_time'],
                persistence=True,
                persistence_type='local',),
            ]
        ),
        html.Div(children=[
                dbc.Label("Psi parameter"),
                dcc.Slider(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=params['psi_param'],
                    marks={str(mark / 10): str(mark / 10) for mark in range(11)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='psi_param',
                persistence=True,
                persistence_type='local',
                ),
            ]
        ),
        html.Div(children=[
                dbc.Label("alpha_max parameter"),
                dcc.Slider(
                    min=1.0,
                    max=1.30,
                    value=params['alpha_max'],
                    step=0.01,
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='alpha_max',
                persistence=True,
                persistence_type='local',
                ),
            ]
        ),
        html.Div(children=[
                dbc.Label("alpha_tau parameter"),
                dbc.Input(id="alpha_tau", type="number", value=params['alpha_tau'],
                persistence=True,
                persistence_type='local',),
            ]
        ),
        html.Div(children=[
                dbc.Label("Base share of production used for rebuilding in impacted region"),
                dcc.Slider(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=params['impacted_region_base_production_toward_rebuilding'],
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='impacted_region_base_production_toward_rebuilding',
                persistence=True,
                persistence_type='local',
                ),
            ]
        ),
        html.Div(children=[
                dbc.Label("Base share of production exported for rebuilding in non-impacted region"),
                dcc.Slider(
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=params['row_base_production_toward_rebuilding'],
                    tooltip={"placement": "bottom", "always_visible": True},
                    id='unimpacted_region_base_production_toward_rebuilding',
                persistence=True,
                persistence_type='local',
                ),
            ]
        ),
    ],
    body=True,
    ),
    html.Hr(),
    html.H2(children=["Validate current parameters: ",
                    dbc.Button(
                        "Yes !",
                        color="primary",
                        id="set-params-button"
                        ),
        ], className="d-grid gap-2 d-md-flex justify-content-md-end"),
    ])
    return res

app.layout = dbc.Container(style={'max-width':'inherit'}, children=[
        #dcc.Store(id='main-df', storage_type='session'),
        #dcc.Store(id='stock-df', storage_type='session'),
        dcc.Store(id='params-store', storage_type='session'),
        dcc.Store(id='events-store', storage_type='session'),
        dcc.Store(id='df-store'),
        dcc.Store(id='df-stocks-store'),
        html.H1("ARIO"),
        html.Hr(),
        dbc.Tabs(children=[
                dbc.Tab(label="Parameters", tab_id="params"),
                dbc.Tab(label="Events", tab_id="events"),
                dbc.Tab(label="Graphs", tab_id="graphs"),
            ],
            id="tabs",
            active_tab="params",
        ),
        html.Div(id="tab-content", className="p-4"),
    ]
)

@func_timer_decorator
def graph_core(params, indexes):
    graph_core = html.Div(id='graph_core', className="p-4", children = [
        dbc.Button("(Re)load data", color="primary", id="load-data-button"),
        html.Div(style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'grid-column-gap' : '20px'}, children=[
                    html.Div(children=[
                        html.H1("Main variables evolution"),
                        html.Div(style={'width':'auto'}, children=[
                            html.Label("Regions selected"),
                            dcc.Dropdown(
                                id='region-dropdown',
                                value=[indexes['regions'][0]],
                                options=[
                                    {'label': c, 'value': c}
                                    for c in indexes['regions']
                                ],
                                multi=True,
                            ),
                            html.Label("Sectors selected"),
                            dcc.Dropdown(
                                id='sector-dropdown',
                                value=[indexes['sectors'][0]],
                                options=[
                                    {'label': c, 'value': c}
                                    for c in indexes['sectors']
                                ],
                                multi=True
                            ),
                            html.Label("Variables selected"),
                            dcc.Checklist(
                                id='variable-checklist',
                                options=[
                                    {'label': 'Production', 'value': 'production'},
                                    {'label': 'Rebuild production', 'value': 'rebuild_production'},
                                    {'label': 'Production Cap', 'value': 'production_max'},
                                    {'label': 'Demand', 'value': 'demand'},
                                    {'label': 'Rebuild demand', 'value': 'rebuild_demand'},
                                    {'label': 'Overproduction', 'value': 'overprod'}
                                ],
                                value=['production']
                            )
                        ]),
                    ]),
                    html.Div(
                        children=[
                            html.H1("Stocks evolution"),
                            html.Div(style={'width':'auto'}, children=[
                                html.Label("Sector selected"),
                                dcc.Dropdown(
                                    id='stock-sector-dropdown',
                                    value=indexes['sectors'][0],
                                    options=[
                                        {'label': c, 'value': c}
                                        for c in indexes['sectors']
                                    ],
                                    multi=False
                                ),
                                html.Div(style={'width':'auto'}),
                                html.Label("Stocks of :"),
                                dcc.Dropdown(
                                    id='stock-dropdown',
                                    value=[indexes['sectors'][0]],
                                    options=[
                                        {'label': c, 'value': c}
                                        for c in indexes['sectors']
                                    ],
                                    multi=True
                                )]),

                        ]),# Define callback to update graph
                    html.Div(children=[dcc.Graph(id='graph-1')]),
                    html.Div(children=[dcc.Graph(id='graph-2')]),
                ]),
                html.Label("Steps"),
                dcc.RangeSlider(
                    min=0,
                    max=params['n_timesteps'],
                    value=[0,params['n_timesteps']],
                    marks={str(step): str(step) for step in range(params['n_timesteps']) if step % 10 == 0},
                    tooltip={"placement": "bottom", "always_visible": True},
                    allowCross=False,
                    id='year-selection'
                ),
                html.Div(children=[
                    html.H1("Usefull second graph"),
                    html.Div(style={'width':'auto'}, children=[
                        html.Label("Regions selected"),
                        dcc.Dropdown(
                            id='region-dropdown-2',
                            value=[indexes['regions'][0]], options=[
                                {'label': c, 'value': c}
                                for c in indexes['regions']
                            ],
                            multi=True
                        ),
                        html.Label("Sectors selected"),
                        dcc.Dropdown(
                            id='sector-dropdown-2',
                            value=[indexes['sectors'][0]], options=[
                                {'label': c, 'value': c}
                                for c in indexes['sectors']
                            ],
                            multi=True
                        ),
                        html.Label("Variables selected"),
                        dcc.Checklist(
                            id='variable-checklist-2',
                            options=[
                                {'label': 'Production', 'value': 'production'},
                                {'label': 'Rebuild production', 'value': 'rebuild_production'},
                                {'label': 'Production Cap', 'value': 'production_max'},
                                {'label': 'Demand', 'value': 'demand'},
                                {'label': 'Rebuild demand', 'value': 'rebuild_demand'},
                                {'label': 'Overproduction', 'value': 'overprod'}
                            ],
                            value=['production']
                        )
                    ]),
                    html.Div([dcc.Graph(id='graph-3')])
                ])
    ])
    return graph_core

events_tab = html.Div()

@app.callback(
    Output("params-store", "data"),
    Input("set-params-button","n_clicks"),
    State("params-store", "data"),
    State("local_storage", "value"),
    State("n_timesteps", "value"),
    State("inventory_restoration_time", "value"),
    State("psi_param", "value"),
    State("alpha_max", "value"),
    State("alpha_tau", "value"),
    State("impacted_region_base_production_toward_rebuilding", "value"),
    State("unimpacted_region_base_production_toward_rebuilding", "value"),
)
def update_params(n, params, storage, n_time, inv_tau, psi, alpha_m, alpha_tau, imp_base_prod_rebuild, unimp_base_prod_rebuild):
    if not n and not params:
        return default_params
    else:
        #params = params or default_params
        params["storage_dir"]=storage
        params['results_storage']=params['storage_dir']+"results/"
        params["exio_params_file"] = storage+"mrio_params.json"
        params["bool_run_detailled"]=True
        params["psi_param"]=psi
        params["inventory_restoration_time"]= inv_tau
        params["alpha_max"]= alpha_m
        params["alpha_tau"]=alpha_tau
        params["n_timesteps"]= n_time
        params["min_duration"]= (n_time // 100) * 25
        params["impacted_region_base_production_toward_rebuilding"]= imp_base_prod_rebuild
        params["row_base_production_toward_rebuilding"]= unimp_base_prod_rebuild
        with (pathlib.Path(storage)/"params.json").open('w') as f:
            f.write(json.dumps(params, indent=4, sort_keys=True))
        return params

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    State("params-store", "data")
)
def render_tab_content(active_tab, params):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if params is None:
        params=default_params
    if active_tab is not None:
        if active_tab == "params":
            return params_controls(params)
        elif active_tab == "events":
            return render_events_tab(params, indexes)
        elif active_tab == "graphs":
            try:
                params['results_storage']=params['storage_dir']+"results/"
                #df, _ = load_dfs(params)
            except Exception as e:
                print("Exception: ",e.args)
                return dbc.Alert("Couldn't load the results, have you launched the simulation ?", color="danger")
            return graph_core(params, indexes)
        else:
            return "No tab selected"

@app.callback(Output("sim-badge", "children"),
              Input("launch-sim-button", "n_clicks"),
              State("params-store", "data"),
              State("events-store", "data")
              )
def launch_sim(n, params, events):
    """
    This callback
    """
    if not n:
        return dbc.Badge("Pending", color="secondary", className="me-1")

    print(events)
    print(params)
    wd = cwd
    if not wd.exists() or not wd.is_dir():
        return dbc.Badge("Failure - invalid wd", color="danger", className="me-1")

    mrio_path = pathlib.Path(params['storage_dir'])/"mrio.pkl"
    if not mrio_path.exists() or not mrio_path.is_file():
        print(mrio_path)
        print(mrio_path.exists())
        print(mrio_path.is_file())
        return dbc.Badge("Failure - invalid mrio file", color="danger", className="me-1")
    model = Simulation(mrio_path, params)
    model.read_events_from_list(events)
    model.loop()
    return dbc.Badge("Success", color="success", className="me-1")

@app.callback(
    Output('graph-1', 'figure'),
    State('params-store','data'),
    State('df-store','data'),
    Input('region-dropdown', 'value'),
    Input('sector-dropdown', 'value'),
    Input('variable-checklist', 'value'),
    Input('year-selection', 'value')
)
def update_graph1(params, df, region_selection, sector_selection, variable_selection, year_selection):
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    if region_selection is None or sector_selection is None or variable_selection is None or year_selection is None:
        return px.line()
    #df, _ = load_dfs(params)
    dff = df[df.step.between(year_selection[0],year_selection[1])]
    dff = dff[dff.region.isin(region_selection)]
    dff = dff[dff.sector.isin(sector_selection)]
    dff = dff[dff.variable.isin(variable_selection)]
    #print(dff.step, dff.value, dff.region, dff.sector, dff.variable)
    fig = px.line(x=dff.step, y=dff.value, symbol = dff.region, color = dff.sector, line_dash=dff.variable)
    #fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig

@app.callback(
    Output('graph-3', 'figure'),
    State('params-store','data'),
    State('df-store','data'),
    Input('region-dropdown-2', 'value'),
    Input('sector-dropdown-2', 'value'),
    Input('variable-checklist-2', 'value'),
    Input('year-selection', 'value')
)
def update_graph3(params, df, region_selection, sector_selection, variable_selection, year_selection):
    if region_selection is None or sector_selection is None or variable_selection is None or year_selection is None:
        return px.line()
    #df, _ = load_dfs(params)
    dff = df[df.step.between(year_selection[0],year_selection[1])]
    dff = dff[dff.region.isin(region_selection)]
    dff = dff[dff.sector.isin(sector_selection)]
    dff = dff[dff.variable.isin(variable_selection)]
    #print(dff.step, dff.value, dff.region, dff.sector, dff.variable)
    fig = px.line(x=dff.step, y=dff.value, symbol = dff.region, color = dff.sector, line_dash=dff.variable)
       #fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig

@app.callback(
    Output('graph-2', 'figure'),
    State('params-store','data'),
    State('df-stocks-store','data'),
    Input('region-dropdown', 'value'),
    Input('stock-sector-dropdown', 'value'),
    Input('year-selection', 'value'),
    Input('stock-dropdown', 'value')
)
def update_graph2(params, df_stocks, region_selection, sector, year_selection, stocks_selection):
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    if region_selection is None or sector is None or year_selection is None or stocks_selection is None:
        return px.line()
    #_, df_stocks = load_dfs(params)
    dff = df_stocks
    dff = dff[dff.sector==sector]
    dff = dff[dff.step.between(year_selection[0],year_selection[1])]
    dff = dff[dff.region.isin(region_selection)]
    dff = dff[dff["stock of"].isin(stocks_selection)]
    #print(dff.step, dff.value, dff.region, dff.region)
    fig = px.line(x=dff.step, y=dff.value, symbol = dff.region, color = dff['stock of'])
    #fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    return fig

@app.callback(
    Output("event-container", "children"),
    [
        Input("add-event", "n_clicks"),
        Input({"type": "dynamic-delete", "index": ALL}, "n_clicks"),
    ],
    [State("event-container", "children")],
    State("params-store", "data"),
)
def events_list_callback(n_clicks, _, children, params):
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0] #wtf is this ?
    if "index" in input_id:
        delete_chart = json.loads(input_id)["index"]
        children = [
            chart for chart in children if "'index': " + str(delete_chart) not in str(chart)
        ]
    else:
        uid = n_clicks,
        new_element = html.Div(
            style={
                "width": "60%",
                "display": "inline-block",
                "outline": "thin lightgrey solid",
                "padding": 10,
                "margin" : 5,
            },
            children=[
                html.Button(
                    "X",
                    id={"type": "dynamic-delete", "index": n_clicks},
                    n_clicks=0,
                    style={"display": "block"},
                ),
                event_controls(uid, params, indexes)
            ],
        )
        children.append(new_element)
    return children

@app.callback(
    Output("events-store", "data"),
    Input("register-events", "n_clicks"),
    [State({"event-id":ALL, "attribute":ALL}, "value"),
    State({"event-id":ALL, "attribute":ALL}, "id")],
)
def register_events(n_clicks, events, events_id_dict):
    if not n_clicks:
        return []
    else:
        events_ids = set([id['event-id'] for id in events_id_dict])
        #print('events_id_dict: ',events_id_dict)
        mega_dict = {(id['event-id'],id['attribute']):ev for id,ev in zip(events_id_dict,events)}
        #print('mega-dict: ', mega_dict)
        res = []
        for id in events_ids:
            res.append({
                'name':id,
                'occur':mega_dict[(id, 'occur')],
                'duration':mega_dict[(id, 'duration')],
                'q_dmg':mega_dict[(id, 'q_dmg')],
                'aff-regions':mega_dict[(id, 'aff-regions')],
                'dmg-distrib-regions':mega_dict[(id, 'dmg-distrib-regions')],
                'aff-sectors':mega_dict[(id, 'aff-sectors')],
                'dmg-distrib-sectors-custom':mega_dict[(id, 'dmg-distrib-sectors-custom')],
                'dmg-distrib-sectors':[v/(n_v+1) for v,n_v in zip(mega_dict[(id, 'dmg-distrib-sectors')], range(len(mega_dict[(id, 'dmg-distrib-sectors')])))], #TODO: v,k
                'rebuilding-sectors': {k:v/(n_v+1) for k,v,n_v in zip(mega_dict[(id, 'rebuilding-sectors')], mega_dict[(id, 'rebuilding-distrib-sectors')], range(len(mega_dict[(id, 'rebuilding-distrib-sectors')])))} #TODO: v,k
                      })
        #print(res)
        return res
    #print(dash.callback_context.triggered)
    #print(dash.callback_context.states_list)

def dmg_distrib_slider_regions(e_uid,values,marks, n_regions,disabled):
    #print(n_regions)
    #print('eyh: ',sum([v/n for v,n in zip(values[:-1],range(1,n_regions))]))
    res = html.Div([
        html.Label("Share"),
        dcc.RangeSlider(
            min=-0.0004,
            max=n_regions*(1-sum([v/n for v,n in zip(values[:-1],range(1,n_regions))]))+0.0004,
            value=values,
            included=True,
            marks=marks,
            step=0.01,
            #tooltip={"placement": "top", "always_visible":True},
            id={'event-id':e_uid, 'attribute':'dmg-distrib-regions'},
            disabled = disabled
                ),
    ])
    return res

#@app.callback(Output({"event-id":ALL, "attribute":"aff-regions"}, "value"),


@app.callback(
    Output({"event-id":ALL, "attribute":"dmg-distrib-regions-custom"}, "children"),
    Input({"event-id": ALL, "attribute":"dmg-distrib-regions-select"}, "value"),
    Input({'event-id': ALL, 'attribute':'dmg-distrib-regions'}, "value"),
    Input({'event-id': ALL, 'attribute': 'aff-regions'}, "value"),
    Input({'event-id': ALL, 'attribute': 'aff-regions'}, "id")
)
def dmg_distrib_region_custom(radio, distrib, aff_regions, id):
    res = []
    #print('r::id: ', id)
    for event_n in range(len(id)):
        r=radio[event_n]
        aff_r = aff_regions[event_n]
        #print('r::distrib: ',distrib)
        if len(distrib) <= event_n:
            dis = None
        else:
            dis = distrib[event_n]
        uid = id[event_n]['event-id']
        if aff_regions is None:
            res.append(html.Div(children=dmg_distrib_slider_regions(-1,[0],{},1,True)))
        elif r == "shared":
            values = [(n+1) * (1/len(aff_r)) for n in range(len(aff_r))]
            marks = { v+0.00001:r+': '+str(v/float(r_n+1)) for v,r,r_n in zip(values,aff_r,range(len(aff_r)))}
            res.append(dmg_distrib_slider_regions(uid,values,marks,len(aff_r),True))
        else:
            ctx = dash.callback_context.triggered
            #print('r::ctx: ',ctx)
            #[{'prop_id': '{"attribute":"dmg-distrib","event-id":"0"}.value', 'value': [0.47, 1]}]
            if ast.literal_eval(ctx[0]['prop_id'].split('.')[0])['attribute'] == "dmg-distrib-regions":
                values = dis
            else:
                values = [(n+1) * (1/len(aff_r)) for n in range(len(aff_r))]
            #print('r::values: ',values)
            #print('r::aff_r: ',aff_r)
            marks = { v+0.00001:r+': '+str(v/float(r_n+1)) for v,r,r_n in zip(values,aff_r,range(len(aff_r)))}
            res.append(dmg_distrib_slider_regions(uid,values,marks,len(aff_r),False))
    #print('r::res: ', res)
    return res

def dmg_distrib_slider_sectors(e_uid,values,marks, n_sectors,disabled):
    #print(n_regions)
    #print('eyh: ',sum([v/n for v,n in zip(values[:-1],range(1,n_regions))]))
    res = html.Div([
        html.Label("Share"),
        dcc.RangeSlider(
            min=-0.0004,
            max=n_sectors*(1-sum([v/n for v,n in zip(values[:-1],range(1,n_sectors))]))+0.0004,
            value=values,
            included=True,
            marks=marks,
            step=0.01,
            #tooltip={"placement": "top", "always_visible":True},
            id={'event-id':e_uid, 'attribute':'dmg-distrib-sectors'},
            disabled = disabled
                ),
    ])
    return res

#@app.callback(Output({"event-id":ALL, "attribute":"aff-regions"}, "value"),


@app.callback(
    Output({"event-id":ALL, "attribute":"dmg-distrib-sectors-custom"}, "children"),
    Input({"event-id": ALL, "attribute":"dmg-distrib-sectors-select"}, "value"),
    Input({'event-id': ALL, 'attribute':'dmg-distrib-sectors'}, "value"),
    Input({'event-id': ALL, 'attribute': 'aff-sectors'}, "value"),
    Input({'event-id': ALL, 'attribute': 'aff-sectors'}, "id")
)
def dmg_distrib_sector_custom(radio, distrib, reb_sectors, id):
    res = []
    #print('s::id: ', id)
    for event_n in range(len(id)):
        r=radio[event_n]
        aff_s = reb_sectors[event_n]
        #print('s::radio: ',radio)
        #print('s::aff_s: ',reb_sectors)
        #print('s::distrib: ',distrib)
        if len(distrib) <= event_n:
            dis = None
        else:
            dis = distrib[event_n]
        uid = id[event_n]['event-id']
        if reb_sectors is None:
            res.append(html.Div(children=dmg_distrib_slider_sectors(-1,[0],{},1,True)))
        elif r == "gdp":
            values = [(n+1) * (1/len(aff_s)) for n in range(len(aff_s))]
            marks = { v+0.00001:s+': '+str(v/float(s_n+1)) for v,s,s_n in zip(values,aff_s,range(len(aff_s)))}
            res.append(dmg_distrib_slider_sectors(uid,values,marks,len(aff_s),True))
        else:
            ctx = dash.callback_context.triggered
            #print('s::ctx: ',ctx)
            #[{'prop_id': '{"attribute":"dmg-distrib","event-id":"0"}.value', 'value': [0.47, 1]}]
            if ast.literal_eval(ctx[0]['prop_id'].split('.')[0])['attribute'] == "dmg-distrib-sectors":
                values = dis
            else:
                values = [(n+1) * (1/len(aff_s)) for n in range(len(aff_s))]
            #print('s::values: ',values)
            #print('s::aff_s: ',aff_s)
            marks = { v+0.00001:s+': '+str(v/float(s_n+1)) for v,s,s_n in zip(values,aff_s,range(len(aff_s)))}
            res.append(dmg_distrib_slider_sectors(uid,values,marks,len(aff_s),False))
    #print('s::res: ', res)
    return res

def rebuilding_distrib_slider_sectors(e_uid,values,marks, n_sectors,disabled):
    #print(n_regions)
    #print('eyh: ',sum([v/n for v,n in zip(values[:-1],range(1,n_regions))]))
    res = html.Div([
        html.Label("Share"),
        dcc.RangeSlider(
            min=-0.0004,
            max=n_sectors*(1-sum([v/n for v,n in zip(values[:-1],range(1,n_sectors))]))+0.0004,
            value=values,
            included=True,
            marks=marks,
            step=0.01,
            #tooltip={"placement": "top", "always_visible":True},
            id={'event-id':e_uid, 'attribute':'rebuilding-distrib-sectors'},
            disabled = disabled
                ),
    ])
    return res

#@app.callback(Output({"event-id":ALL, "attribute":"aff-regions"}, "value"),


@app.callback(
    Output({"event-id":ALL, "attribute":"rebuilding-distrib-sectors-custom"}, "children"),
    Input({'event-id': ALL, 'attribute':'rebuilding-distrib-sectors'}, "value"),
    Input({'event-id': ALL, 'attribute':'rebuilding-sectors'}, "value"),
    Input({'event-id': ALL, 'attribute':'rebuilding-sectors'}, "id")
)
def rebuild_distrib_sector_custom(distrib, reb_sectors, id):
    res = []
    print('rs::id: ', id)
    ctx = dash.callback_context.triggered
    print('rs::ctx: ',ctx)
    for event_n in range(len(id)):
        #r=radio[event_n]
        reb_s = reb_sectors[event_n]
        print('rs::reb_sectors: ',reb_sectors)
        print('rs::reb_s: ', reb_s)
        print('rs::distrib: ',distrib)
        #print(not reb_s)
        if len(distrib) <= event_n:
            dis = None
        else:
            dis = distrib[event_n]
        uid = id[event_n]['event-id']
        if not reb_s:
            res.append(html.Div(children=rebuilding_distrib_slider_sectors(uid,[0],{},1,True)))
        else:
            #[{'prop_id': '{"attribute":"dmg-distrib","event-id":"0"}.value', 'value': [0.47, 1]}]
            if ctx[0]['prop_id'] == '.':
                values = [1]
            elif ast.literal_eval(ctx[0]['prop_id'].split('.')[0])['attribute'] == "rebuilding-distrib-sectors":
                values = dis
            else:
                values = [(n+1) * (1/len(reb_s)) for n in range(len(reb_s))]
            print('rs::values: ',values)
            print('rs::reb_s: ',reb_s)
            for v,s,s_n in zip(values,reb_s,range(len(reb_s))):
                print(v,s,s_n)
            marks = { v+0.00001:s+': '+str(v/float(s_n+1)) for v,s,s_n in zip(values,reb_s,range(len(reb_s)))}
            res.append(rebuilding_distrib_slider_sectors(uid,values,marks,len(reb_s),False))
    #print('rs::res: ', res)
    return res

if __name__ == "__main__":
    app.run_server(debug=False, port=8885)
