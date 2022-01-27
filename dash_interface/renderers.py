#from pathlib import Path
#print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

"""
Html renderers of the different elements
"""
import dash_bootstrap_components as dbc
from dash import dcc, html #, ALL, MATCH, ALLSMALLER

def events_tab():
    res = html.Div(children = [
        dbc.Row(children=[
            dbc.Col(
                dbc.Button(
                    "Add another event",
                    id="add-event",
                    n_clicks=0,
                    style={"display": "inline-block"},
                ),
                width="auto",
            ),
            dbc.Col(
                dbc.Button("Register displayed events",
                           id="register-events",
                           n_clicks=0,
                           style={"display": "inline-block"},
                           className="d-grid gap-2 d-md-flex justify-content-md-end"
                           ),
                width="auto",
            ),
        ]),
        html.Div(id="event-container", children=[]),
        html.Br(),
        html.H2(children=["Launch simulation : ",
                          dbc.Button(
                              "Yes !",
                              color="primary",
                              id="launch-sim-button"
                          ),
                          ], className="d-grid gap-2 d-md-flex"),
    ])
    return res

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
        html.Div(id={"event-id":str(uid[0]), "attribute":'rebuilding-distrib-sectors-custom'}),
    ], body=True)
    return res

def params_controls(params):
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
                          ], className="d-grid gap-2 d-md-flex"),
    ])
    return res


def graph_core(params, indexes):
    graph_core = html.Div(id='graph-core')
    return graph_core
