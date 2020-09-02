import dash_html_components as html
import dash_bootstrap_components as dbc

from assets.navbar import Navbar
from dataTransformations.dict_path import dict_path_data

import dash_table
import pandas as pd

######################################################################################################################
######################################################################################################################
######################################################################################################################

#### source paths for test due to visualization of worth attention companies ####
path_1 = dict_path_data['buy_signal']
path_2 = dict_path_data['sell_signal']
df_buy_test = pd.read_csv(path_1, sep=',', index_col = 0)
df_sell_test = pd.read_csv(path_2, sep=',', index_col = 0)

mwig40 = dict_path_data['mwig40']
swig80 = dict_path_data['swig80']
wig = dict_path_data['wig']
wig20 = dict_path_data['wig20']
df_mwig40 = pd.read_csv(mwig40, delimiter=',',
                     index_col=[0])
df_swig80 = pd.read_csv(swig80, delimiter=',',
                     index_col=[0])
df_wig = pd.read_csv(wig, delimiter=',',
                     index_col=[0])
df_wig20 = pd.read_csv(wig20, delimiter=',',
                     index_col=[0])

df_mwig40['change'] = (df_mwig40.iloc[-1]['<CLOSE>'] - df_mwig40.iloc[-2]['<CLOSE>'])/df_mwig40.iloc[-2]['<CLOSE>'] * 100
df_swig80['change'] = (df_swig80.iloc[-1]['<CLOSE>'] - df_swig80.iloc[-2]['<CLOSE>'])/df_swig80.iloc[-2]['<CLOSE>'] * 100
df_wig['change'] = (df_wig.iloc[-1]['<CLOSE>'] - df_wig.iloc[-2]['<CLOSE>'])/df_wig.iloc[-2]['<CLOSE>'] * 100
df_wig20['change'] = (df_wig20.iloc[-1]['<CLOSE>'] - df_wig20.iloc[-2]['<CLOSE>'])/df_wig20.iloc[-2]['<CLOSE>'] * 100


import os.path, time

import os

HOME_DIR = os.getcwd()


# https://intellipaat.com/community/3770/how-to-get-file-creation-modification-date-times-in-python
today = time.ctime(os.path.getmtime(f'{HOME_DIR}/wseStocks/data/daily/pl/wse stocks/11b.txt'))
proc = "%"
#### Layout ####
navbar = Navbar()

layout = html.Div([navbar,
                   dbc.Container([

#### Visualization of worth attention companies ####
dbc.Row(dbc.Col(html.Br())),
dbc.Row([dbc.Col(html.P("Date of analysis: "),
                 width={"size": 2, "offset": 4}),
         dbc.Col(html.P(f"{today}"), style={'color': 'blue'},
                 width={"size": 3, "offset": 0})]),
dbc.Row([dbc.Col(html.P("WIG CLOSE: ")),
         dbc.Col(html.P(f"{df_wig.iloc[-1]['<CLOSE>']}",
                       style={'color': 'blue'})),
         dbc.Col(html.P("CHANGE(%): ")),
         dbc.Col(html.P(round(float(f"{df_wig.iloc[-1]['change']}"), 2),
                        style={'color': 'blue'}))]),
dbc.Row([dbc.Col(html.P("WIG20 CLOSE: ")),
         dbc.Col(html.P(f"{df_wig20.iloc[-1]['<CLOSE>']}",
                       style={'color': 'blue'})),
         dbc.Col(html.P("CHANGE(%): ")),
         dbc.Col(html.P(round(float(f"{df_wig20.iloc[-1]['change']}"), 2), "%",
                        style={'color': 'blue'}))]),
dbc.Row([dbc.Col(html.P("mWIG40 CLOSE: ")),
         dbc.Col(html.P(f"{df_mwig40.iloc[-1]['<CLOSE>']}",
                       style={'color': 'blue'})),
         dbc.Col(html.P("CHANGE(%): ")),
         dbc.Col(html.P(round(float(f"{df_mwig40.iloc[-1]['change']}"), 2), "%",
                        style={'color': 'blue'}))]),
dbc.Row([dbc.Col(html.P("sWIG80 CLOSE: ")),
         dbc.Col(html.P(f"{df_swig80.iloc[-1]['<CLOSE>']}",
                       style={'color': 'blue'})),
         dbc.Col(html.P("CHANGE(%): ")),
         dbc.Col(html.P(round(float(f"{df_swig80.iloc[-1]['change']}"), 2), "%",
                        style={'color': 'blue'}))]),
######################################################################################################################
######################################################################################################################
######################################################################################################################
dbc.Row(dbc.Col(dbc.Alert("Buy signal", color="success", style={"textAlign":"center"}))),
dbc.Row(dbc.Col(html.Br())),
dbc.Row(dbc.Col(dash_table.DataTable(data=df_buy_test.to_dict('records'),
                                     columns=[{'id': c, 'name': c} for c in df_buy_test.columns],
                                     style_as_list_view=True,
                                     style_cell={'font-size': '9px','padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'},
                                     style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                                     style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_buy_test.columns]))),
dbc.Row(dbc.Col([html.Br(), html.Br(), dbc.Alert("Sell signal", color="danger", style={"textAlign":"center"})])),
dbc.Row(dbc.Col(dash_table.DataTable(data=df_sell_test.to_dict('records'),
                                     columns=[{'id': c, 'name': c} for c in df_sell_test.columns],
                                     style_as_list_view=True,
                                     style_cell={'font-size': '9px','padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'},
                                     style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                                     style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in df_sell_test.columns]))),
dbc.Row(dbc.Col(html.Br())),
dbc.Row(dbc.Col(html.Br()))],
                       className="mt-4")])
