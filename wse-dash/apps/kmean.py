import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
import os

from assets.navbar import Navbar

HOME_DIR = os.getcwd()
KMEAN_CLUSTERING_REPORT = pd.read_csv(f'{HOME_DIR}/databases/kmean_report.csv', sep=',', index_col = 0)

navbar = Navbar()

textareas = html.Div([dbc.Textarea(className="mb-3",
                                   placeholder="Donec id elit non mi porta gravida at eget metus.")])

layout = html.Div([navbar,
                   dbc.Container([dbc.Row([dbc.Col([html.H1("Grouping of related economy sectors using the K-mean clustering algorythm"),
                                                    html.P(),
                                                    html.H2("Description"),
                                                    html.P("""\
                                                        Donec id elit non mi porta gravida at eget metus.Fusce dapibus, tellus ac cursus
                                                        commodo, tortor mauris condimentumnibh, ut fermentum massa justo sit amet risus
                                                        . Etiam porta semmalesuada magna mollis euismod. Donec sed odio dui. Donec id
                                                        elit nonmi porta gravida at eget metus. Fusce dapibus, tellus ac cursuscommodo,
                                                        tortor mauris condimentum nibh, ut fermentum massa justo sitamet risus. Etiam
                                                        porta sem malesuada magna mollis euismod. Donec sedodio dui.
                                                            """)])]),
                                  dbc.Row([html.Div([html.Button(id='submit-button',
                                                                 children='Run clusters searching'),
                                                     dbc.Collapse(dbc.Card(dbc.CardBody([html.Iframe(id='datatable',
                                                                                                     sandbox='',
                                                                                                     style={'height': '67vh', 'width': 720})
                                                                                         ])),id="collapse-clusters"),
                               ]),
                             ], justify="center", align="center", className="h-50"),
                                  dbc.Row(dbc.Col(dash_table.DataTable(
                                      data=KMEAN_CLUSTERING_REPORT.to_dict('records'),
                                      columns=[{"name": c, "id": c} for c in KMEAN_CLUSTERING_REPORT.columns],
                                      style_as_list_view=True,
                                      style_cell={'font-size': '9px','padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'},
                                      style_header={'backgroundColor': 'white','fontWeight': 'bold'},
                                      style_cell_conditional=[{'if': {'column_id': c},'textAlign': 'center'} for c in KMEAN_CLUSTERING_REPORT.columns])))
                       ],
                       className="mt-4")])

# to do:
# - dodac zapisywanie starych raportow do archiwum z suffixem po dacie
# - dodac wykres silhu
