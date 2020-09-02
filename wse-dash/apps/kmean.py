import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import dash_table
import os

from assets.navbar import Navbar

HOME_DIR = os.getcwd()

navbar = Navbar()

textareas = html.Div([dbc.Textarea(className="mb-3",
                                   placeholder="Donec id elit non mi porta gravida at eget metus.")])

layout = html.Div([navbar,
                   dbc.Container([dbc.Row([dbc.Col([html.H1("Grouping of related economy sectors using the K-mean clustering algorythm"),
                                                    html.P(),
                                                    html.H2("Description"),
                                                    html.P("""\
                                                        K-means is an efficient clustering method used to group similar 
                                                        data into divisions based on initial centroids of clusters. 
                                                        The clustering was performed on the basis of stock price 
                                                        movement data, which were logarithm and then normalized in 
                                                        a range between -1 and 1. The K value was specified with the
                                                        Silhouette method.
                                                            """)])]),
                                  dbc.Row([html.Div([html.Button(id='submit-button',
                                                                 children='Run clusters searching'),
                                                     dbc.Collapse(dbc.Card(dbc.CardBody([html.Iframe(id='datatable',
                                                                                                     sandbox='',
                                                                                                     style={'height': '67vh', 'width': 720})
                                                                                         ])),id="collapse-clusters"),
                               ]),
                             ], justify="center", align="center", className="h-50"),
                       ],
                       className="mt-4")])