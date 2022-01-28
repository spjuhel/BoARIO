"""
Dash app for ARIO3
"""
from datetime import datetime
from defaults import WORKING_DIR, load_defaults
####################
# Starting the app #
####################
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, ServersideOutputTransform

app = DashProxy(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True,
                transforms=[
                    ServersideOutputTransform(),  # enable use of ServersideOutput objects
                ])
server = app.server

DEFAULTS_STORAGE = "defaults-storage"
current_params, default_mrio_params, default_indexes = load_defaults(DEFAULTS_STORAGE)

params = current_params

from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from layouts import layout

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/ario3':
         return layout
    else:
        return '404'


#######################
# Make ario3 loadable #
#######################
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')

#############
# Callbacks #
#############
import ast
import json
import dash
from dash import dcc, html, ALL #, MATCH, ALLSMALLER
from dash.exceptions import PreventUpdate
import pathlib
import ario3.simulation.base as sim
#from ario3.indicators.indicators import Indicators
import dash_bootstrap_components as dbc
from dash_extensions.enrich import ServersideOutput, Output, Input, State
from dash_interface import utils
from dash_interface import renderers as render
import plotly.express as px

@app.callback(
    Output("params-store", "data"),
    Output("indexes-store", "data"),
    Output("mrio-params-store", "data"),
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
        return [current_params, default_indexes, default_mrio_params]
    else:
        #params = params or current_params
        params["storage_dir"]=storage
        params['results_storage']=params['storage_dir']+"/results/"
        params["mrio_params_file"] = storage+"/mrio_params.json"
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
        indexes = utils.load_indexes(params)
        mrio_params = utils.load_mrio_params(params)
        return [params, indexes, mrio_params]

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
    State("params-store", "data"),
    State("indexes-store", "data")
)
def render_tab_content(active_tab, params, indexes):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if params is None:
        params=current_params
    if active_tab is not None:
        if active_tab == "params":
            return render.params_controls(params)
        elif active_tab == "events":
            return render.events_tab()
        elif active_tab == "graphs":
            try:
                params['results_storage']=params['storage_dir']+"results/"
                #df, _ = load_dfs(params)
            except Exception as e:
                print("Exception: ",e.args)
                return dbc.Alert("Couldn't load the results, have you launched the simulation ?", color="danger")
            return render.graph_core(params, indexes)
        else:
            return "No tab selected"

@app.callback(Output("sim-status-badge", "children"),
              Input("launch-sim-button", "n_clicks"),
              State("params-store", "data"),
              State("events-store", "data")
              )
def launch_sim(n, params, events):
    """
    This callback
    """
    if not n:
        return dbc.Badge("Simulation pending", color="secondary", className="me-1")

    print(events)
    print(params)
    wd = WORKING_DIR
    if not wd.exists() or not wd.is_dir():
        return dbc.Badge("Failure - invalid wd", color="danger", className="me-1")

    mrio_path = pathlib.Path(params['storage_dir'])/"mrio.pkl"
    if not mrio_path.exists() or not mrio_path.is_file():
        print(mrio_path)
        print(mrio_path.exists())
        print(mrio_path.is_file())
        return dbc.Badge("Failure - invalid mrio file", color="danger", className="me-1")
    #if params['']:
    model = sim.Simulation(mrio_path, params)
    model.read_events_from_list(events)
    with (pathlib.Path(params['storage_dir'])/"last_events.json").open('w') as f:
        json.dump(events, f, indent=4, sort_keys=True)
    model.loop()
    return dbc.Badge("Simulation success !", color="success", className="me-1")

@app.callback(
    Output('graph-1', 'figure'),
    Input('region-dropdown', 'value'),
    Input('sector-dropdown', 'value'),
    Input('variable-checklist', 'value'),
    Input('year-selection', 'value'),
    State('df-store','data'),
)
def update_graph1(region_selection, sector_selection, variable_selection, year_selection, df):
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
    if region_selection is None or sector_selection is None or variable_selection is None or year_selection is None:
        return px.line()
    #df, _ = load_dfs(params)
    #['reg1'] ['construction'] ['production', 'rebuild_production'] [0, 365] {'alpha_base': 1, 'alpha_max': 1.25, 'alpha_tau': 365, 'bool_run_detailled': True, 'mrio_params_file': 'dash-testing-save/mrio_params.json', 'impacted_region_base_production_toward_rebuilding': 0, 'inventory_restoration_time': 64, 'min_duration': 75, 'model_time_step': 1, 'n_timesteps': 365, 'psi_param': 0.75, 'rebuild_tau': 30, 'results_storage': 'dash-testing-save/results/', 'row_base_production_toward_rebuilding': 0.01, 'storage_dir': 'dash-testing-save/', 'timestep_dividing_factor': 365}        region        sector  step    variable         value
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
    Input('region-dropdown-2', 'value'),
    Input('sector-dropdown-2', 'value'),
    Input('variable-checklist-2', 'value'),
    Input('year-selection', 'value'),
    State('params-store','data'),
    State('df-store','data'),
)
def update_graph3(region_selection, sector_selection, variable_selection, year_selection, params, df):
    if region_selection is None or sector_selection is None or variable_selection is None or year_selection is None:
        return px.line()
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
    Input('region-dropdown', 'value'),
    Input('stock-sector-dropdown', 'value'),
    Input('year-selection', 'value'),
    Input('stock-dropdown', 'value'),
    State('params-store','data'),
    State('df-stocks-store','data'),
)
def update_graph2(region_selection, sector, year_selection, stocks_selection, params, df_stocks):
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
    State("indexes-store", "data")
)
def events_list_callback(n_clicks, _, children, params, indexes):
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
                render.event_controls(uid, params, indexes)
            ],
        )
        children.append(new_element)
    return children

@app.callback(
    Output("events-store", "data"),
    Input("register-events", "n_clicks"),
    [State({"event-id":ALL, "attribute":ALL}, "value"),
     State({"event-id":ALL, "attribute":ALL}, "id"),
     State("params-store", "data")],
)
def register_events(n_clicks, events, events_id_dict, params):
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
                'dmg-distrib-sectors-type':mega_dict[(id, 'dmg-distrib-sectors-select')],
                'dmg-distrib-sectors':[v/(n_v+1) for v,n_v in zip(mega_dict[(id, 'dmg-distrib-sectors')], range(len(mega_dict[(id, 'dmg-distrib-sectors')])))], #TODO: v,k
                'rebuilding-sectors': {k:v/(n_v+1) for k,v,n_v in zip(mega_dict[(id, 'rebuilding-sectors')], mega_dict[(id, 'rebuilding-distrib-sectors')], range(len(mega_dict[(id, 'rebuilding-distrib-sectors')])))} #TODO: v,k
            })
            print(res)
        with (pathlib.Path(params['storage_dir'])/"last_events.json").open('w') as f:
            json.dump(res, f, indent=4, sort_keys=True)
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
        uid = id[event_n]['event-id']
        if len(distrib) <= event_n:
            dis = None
        else:
            dis = distrib[event_n]
        if aff_regions is None:
            res.append(html.Div(children=dmg_distrib_slider_regions(-1,[1],{},1,True)))
        elif r == "shared":
            values = [(n+1) * (1/len(aff_r)) for n in range(len(aff_r))]
            marks = { v+0.00001:r+': '+str(v/float(r_n+1)) for v,r,r_n in zip(values,aff_r,range(len(aff_r)))}
            res.append(dmg_distrib_slider_regions(uid,values,marks,len(aff_r),True))
        else:
            ctx = dash.callback_context.triggered
            #print('r::ctx: ',ctx)
            if ctx[0]['prop_id'] == '.':
                values = [1]
            elif ast.literal_eval(ctx[0]['prop_id'].split('.')[0])['attribute'] == "dmg-distrib-regions":
                values = dis
            else:
                values = [(n+1) * (1/len(aff_r)) for n in range(len(aff_r))]
                #print('r::values: ',values)
                #print('r::aff_r: ',aff_r)
            marks = { v+0.00001:r+': '+str(v/float(r_n+1)) for v,r,r_n in zip(values,aff_r,range(len(aff_r)))} #type: ignore
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
def dmg_distrib_sector_custom(radio, distrib, aff_sectors, id):
    res = []
    #print('s::id: ', id)
    for event_n in range(len(id)):
        r=radio[event_n]
        aff_s = aff_sectors[event_n]
        #print('s::radio: ',radio)
        #print('s::aff_s: ',aff_sectors)
        #print('s::distrib: ',distrib)
        uid = id[event_n]['event-id']
        if len(distrib) <= event_n:
            dis = None
        else:
            dis = distrib[event_n]
            #print('s::distrib: ',distrib)
        if aff_sectors is None:
            res.append(html.Div(children=dmg_distrib_slider_sectors(-1,[1],{},1,True)))
        elif r == "gdp":
            values = [(n+1) * (1/len(aff_s)) for n in range(len(aff_s))]
            marks = { v+0.00001:s+': '+str(v/float(s_n+1)) for v,s,s_n in zip(values,aff_s,range(len(aff_s)))}
            res.append(dmg_distrib_slider_sectors(uid,values,marks,len(aff_s),True))
        else:
            ctx = dash.callback_context.triggered
            #print('s::ctx: ',ctx)
            if ctx[0]['prop_id'] == '.':
                values = [1]
                #[{'prop_id': '{"attribute":"dmg-distrib","event-id":"0"}.value', 'value': [0.47, 1]}]
            elif ast.literal_eval(ctx[0]['prop_id'].split('.')[0])['attribute'] == "dmg-distrib-sectors":
                values = dis
            else:
                values = [(n+1) * (1/len(aff_s)) for n in range(len(aff_s))]
                #print('s::values: ',values)
                #print('s::aff_s: ',aff_s)
            marks = { v+0.00001:s+': '+str(v/float(s_n+1)) for v,s,s_n in zip(values,aff_s,range(len(aff_s)))} #type: ignore
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
    #print('rs::id: ', id)
    ctx = dash.callback_context.triggered
    #print('rs::ctx: ',ctx)
    for event_n in range(len(id)):
        #r=radio[event_n]
        reb_s = reb_sectors[event_n]
        #print('rs::reb_sectors: ',reb_sectors)
        #print('rs::reb_s: ', reb_s)
        #print('rs::distrib: ',distrib)
        #print(not reb_s)
        uid = id[event_n]['event-id']
        if len(distrib) <= event_n:
            dis = [1]
        else:
            dis = distrib[event_n]
        if not reb_s:
            res.append(html.Div(children=rebuilding_distrib_slider_sectors(uid,[1],{},1,True)))
        else:
            #[{'prop_id': '{"attribute":"dmg-distrib","event-id":"0"}.value', 'value': [0.47, 1]}]
            if ctx[0]['prop_id'] == '.':
                values = [1]
            elif ast.literal_eval(ctx[0]['prop_id'].split('.')[0])['attribute'] == "rebuilding-distrib-sectors":
                values = dis
            else:
                values = [(n+1) * (1/len(reb_s)) for n in range(len(reb_s))]
                #print('rs::values: ',values)
                #print('rs::reb_s: ',reb_s)
                #for v,s,s_n in zip(values,reb_s,range(len(reb_s))):
                #    print(v,s,s_n)
            marks = { v+0.00001:s+': '+str(v/float(s_n+1)) for v,s,s_n in zip(values,reb_s,range(len(reb_s)))}
            res.append(rebuilding_distrib_slider_sectors(uid,values,marks,len(reb_s),False))
            ##print('rs::res: ', res)
    return res

#df, df_stocks = load_dfs(current_params, index=True)
@app.callback(
    ServersideOutput("data-loaded-trigger", "data"),
    ServersideOutput("df-store", "data"),
    ServersideOutput("df-stocks-store", "data"),
    Input("load-data-button", "n_clicks"),
    State("params-store","data"),
    memoize=True, prevent_initial_call=True)
def load_df_server(trigger, params):
    ctx = dash.callback_context.triggered
    print("load_df_server::ctx: ",ctx)
    return utils.load_dfs(params)

@app.callback(
    Output("last-load-timestamp", "children"),
    Input("data-loaded-trigger", "data")
)
def load_data_timestamp(trigger):
    if trigger is None:
        return dbc.Badge("Never")
    elif trigger['loaded'] == "-1":
        return dbc.Badge("Files not found")
    elif trigger['loaded'] == "1":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        return dbc.Badge(current_time)
    else:
        return dbc.Badge("Other problem")

@app.callback(
    Output("last-params-timestamp", "children"),
    Input("set-params-button","n_clicks"),
)
def load_params_timestamp(trigger):
    if trigger is None:
        return dbc.Badge("Default params")
    else:
        return dbc.Badge("Current params")

@app.callback(
    Output("sim-status", "children"),
    Input("last-params-timestamp","children"),
    State("params-store", "data"),
    State("events-store", "data"),
)
def simulation_status(trigger, params, events):
    # params check
    if params is None:
        params_status = dbc.Badge("Params None - WIP")
    else:
        params_status = dbc.Badge("Params ok - WIP")
    # event check
    events_status = dbc.Badge("Events ok - WIP")
    # mrio file check
    mrio_path = pathlib.Path(params['storage_dir'])/"mrio.pkl"
    if not mrio_path.exists() or not mrio_path.is_file():
        mrio_status = dbc.Badge("Failure - invalid MRIO file", color="danger")
    else:
        mrio_status = dbc.Badge("MRIO file found", color="success")
    return [params_status, events_status, mrio_status]

@app.callback(
    Output("graph-core", "children"),
    Input("data-loaded-trigger", "data"),
    State("params-store", "data"),
    State("indexes-store", "data"),
)
def graph_zone(trigger_data, params, indexes):
    if trigger_data is None:
        return dbc.Col(dbc.Alert("Please fetch data !", color="primary"))
    else:
        core = html.Div(children=[
            dbc.Row(children=[
                dbc.Col(width={"size": 6}, children=[
                    # Graph 1 Col
                    html.Div(children=[
                        html.H3("Main variables evolution"),
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
                    html.Div(children=[dcc.Graph(id='graph-1')]),
                ]),
                dbc.Col(width={"size": 6}, children=[
                    # Graph 2 (stocks) Col
                    html.Div(children=[
                        html.H3("Stocks evolution"),
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
                                options=[{'label': c, 'value': c} for c in indexes['sectors']],
                                multi=True
                            )]),
                    ]),
                    html.Div(children=[dcc.Graph(id='graph-2')]),
                ]),
            ]),
            dbc.Row(children=[
                # Steps select Row
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
            ]),
            dbc.Row(children=[
                # Graph 3 Row
                html.Div(children=[
                    html.H3("Usefull second graph"),
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
                ]),
            ]),
        ])
        return core

if __name__ == '__main__':
    app.run_server(debug=True)
