#from pathlib import Path
#print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import dash_bootstrap_components as dbc
from dash import dcc, html

layout = dbc.Container(style={'max-width':'inherit'}, children=[
    #dcc.Store(id='main-df', storage_type='session'),
    #dcc.Store(id='stock-df', storage_type='session'),
    dcc.Store(id='params-store', storage_type='session'),
    dcc.Store(id='indexes-store', storage_type='session'),
    dcc.Store(id='mrio-params-store', storage_type='session'),
    dcc.Store(id='events-store', storage_type='session'),
    dcc.Store(id='df-store'),
    dcc.Store(id='df-stocks-store'),
    dcc.Store(id='data-loaded-trigger'),
    #dcc.Store(id='params-loaded-trigger'),
    html.H1("ARIO"),
    html.Div(id="Header", children=[
        dbc.Row(children=[
            dbc.Col(children=[
                "Parameter last load :",
                html.Div(children=[], id = "last-params-timestamp"),
            ], width="auto"),
            dbc.Col(children=[
                "Result data last load :",
                html.Div(children=[], id = "last-load-timestamp"),
            ], width="auto"),
            dbc.Col(dbc.Button("(Re)Fetch data", color="primary", id="load-data-button"),
                    width="auto"),
            dbc.Col(children=[
                "Simulation status :",
                dbc.Row(children=[], id = "sim-status"),
                dbc.Row(children=[], id = "sim-status-badge"),
            ], width="2"),
        ]),
    ]),
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
])
