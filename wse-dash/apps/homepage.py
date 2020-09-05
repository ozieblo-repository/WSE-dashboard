import dash_html_components as html
import dash_bootstrap_components as dbc

from assets.navbar import Navbar
from dataTransformations.dict_path import dict_path_data

import base64

png_sgh = dict_path_data['SGHlogotypEN']
base64_sgh = base64.b64encode(open(png_sgh, 'rb').read()).decode('ascii')

png_stooq = dict_path_data['stooqlogo']
base64_stooq = base64.b64encode(open(png_stooq, 'rb').read()).decode('ascii')

png_plotly = dict_path_data['Plotly_Dash_logo']
base64_plotly = base64.b64encode(open(png_plotly, 'rb').read()).decode('ascii')

navbar = Navbar()

layout = html.Div([navbar,
                   dbc.Container([dbc.Row(html.H1("THE DASHBOARD TO CREATE, ANALYZE AND OPTIMIZE INVESTMENT \
                   PORTFOLIO ON THE WARSAW STOCK EXCHANGE MARKET")),
                                  dbc.Row([dbc.Col([html.P(),
                                                    html.P(
                                                        """\
                                                        The final project for postgraduate studies Big Data Engineering at \
                                                        the Warsaw School of Economics.
                                                        """, style={'color': 'red'}),
                                                    html.H3("Overview"),
                                                    html.P(
                                                        """\
                                                        The tool for stock traders eager to maintain a view of \
                                                        the WSE market in order to recognize, analyze, and \
                                                        respond to market changes using technical analysis \
                                                        indicators and basic unsupervised clustering algorythm.
                                                        """),
                                                    html.H3("Main goal"),
                                                    html.P(
                                                        """\
                                                        Identify trends in prices movement of equity securities  \
                                                        using RSI, Stochastic Oscillator, MACD and Bollinger \
                                                        Bands indicators and also an unsupervised K-mean \
                                                        clustering to market sectors to create opportunities \
                                                        for capital gains.
                                                        """),
                                                    html.H3("How to use it?"),
                                                    html.Ol( [html.Li("Click the below button in order to download \
                                                              recent stock data from Stooq.com website and to run the \
                                                              technical analysis for the given day, presented on the \
                                                              Worth Attention Companies tab. Please kindly wait until \
                                                              the date of the analysis will not change."),
                                                              html.Li("Check in details visualizations related with \
                                                              each market indicator from the navigation bar menu above. \
                                                              Please click any active button to run the visualization!"),
                                                              html.Li("Check related companies based on price movements \
                                                              via K-mean clustering from the navigation bar menu above. \
                                                              To run it again, you need to refresh the webpage.")
                                                              ])])]),
                                  dbc.Row([dbc.Col([
                                      html.Br(),
                                      dbc.Button("Download last stock prices and select companies worth attention",
                                                 id="run-scrapper-and-find-best-comp",
                                                 block=True,
                                                 size="lg",
                                                 color="secondary",
                                                 href="/wac"),
                                      html.Div(id='hidden-div',
                                               style={'display':'none'})],
                                      width={"size": 8,
                                             "offset": 2})]),
                                  dbc.Row(dbc.Col(html.Br())),
                                  dbc.Row(dbc.Col(html.Br())),
                                  dbc.Row([dbc.Col(html.Div([html.Img(src='data:image/png;base64,{}'.format(base64_sgh))]),
                                                   width={"size": 2,
                                                          "offset": 0}),
                                           dbc.Col(html.Div([html.Img(src='data:image/png;base64,{}'.format(base64_stooq))]),
                                                   width={"size": 2,
                                                          "offset": 1}),
                                           dbc.Col([html.Br(),
                                                    html.Br(),
                                                    html.Div([html.Img(src='data:image/png;base64,{}'.format(base64_plotly))])],
                                                   width={"size": 2,
                                                          "offset": 3})])],
                                 className="mt-4")])
