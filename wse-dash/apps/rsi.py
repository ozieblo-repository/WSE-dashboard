import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from assets.navbar import Navbar

from dataTransformations.wsedfIntoDict import KmeanOptions

navbar = Navbar()

layout = html.Div([
    navbar,
    dbc.Container(
        [dbc.Row(dbc.Col(html.Div(),
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-0"),
         dbc.Row(dbc.Col(html.Div(html.H1("RELATIVE STRENGTH INDEX")),
                         width={"size": 6, "offset": 3}),
                 id="so-row-1"),
         dbc.Row(dbc.Col([html.Div("The RSI is a momentum oscillator intended \
         to graph the actual and historical strength or weakness of analyzed \
         securities or the market based on closing prices during a recent \
         trading period."),
                          html.Div("The nearer indicators value is to 0, the weaker the momentum is for price movements. \
                          Oppositely an RSI closer to 100 shows a signal of a stronger momentum period. As stated \
                          by author - John Welles Wilder Jr., any number above threshold equal 70 should be \
                          assessed as overbought and below 30 as oversold. An relative strength index between 30 \
                          and 70 should be considered as neutral and around 50 like no trend."),
                          html.P("")])),
         dbc.Row(dbc.Col(html.Div("Select the company to analyze:"),
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-2"),
         dbc.Row(dbc.Col(html.Div([dcc.Dropdown(id='dropdown-so',
                                                options=KmeanOptions().wse_options_for_indicators(),
                                                value='11B')]),
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-3"),
         dbc.Row([dbc.Col([html.Br(),
                           html.Span("Set M parameter:",
                                     style={"padding-right":"13px"}),
                           dcc.Input(id="input_rsi_n",
                                     type="number",
                                     placeholder="input with range",
                                     min=1,
                                     max=30,
                                     step=1,
                                     value=14),
                           html.Br(),
                           html.Br()],
                          width={"size": 3,
                                 "offset": 3}),
                  dbc.Col([html.Br(),
                           html.Div("RSI is calculated using an n-period smoothed moving average (SMA) or Exponential \
                           Moving Averages (EMA). 14 days is likely the most common period, but it have been known to use a wide \
                           variety of values of this variable by traders",
                                    style={'color':'blue',
                                           'fontSize':11})],
                          width={"size": 3,
                                 "offset": 0})]
                 #,id="so-row-n"
        ),
         dbc.Row([dbc.Col([html.Br(),
                  html.Br(),
                  html.Span("Set High threshold:",
                            style={"padding-right":"15px"}),
                  dcc.Input(id="input_rsi_high_threshold",
                            type="number",
                            placeholder="input with range",
                            min=50,
                            max=99,
                            step=1,
                            value=70),
                  html.P(),
                  html.Span("Set Low threshold:",
                            style={"padding-right":"19px"}),
                  dcc.Input(id="input_rsi_low_threshold",
                            type="number",
                            placeholder="input with range",
                            min=1,
                            max=49,
                            step=1,
                            value=30)],
                 width={"size": 3,
                        "offset": 3}),
         dbc.Col([html.Br(),
                  html.Ul([html.Li("Around 100 – high probability of reverse of the current trend downward"),
                           html.Li("70 and more – buy signal"),
                           html.Li("More than 30 and less than 70 – neutral trend"),
                           html.Li("Around 50 – no trend"),
                           html.Li("30 and less – sell signal"),
                           html.Li("Around 0 - high probability of reverse of the current trend upward")],
                          style={'color':'blue',
                                 'fontSize':11}),
                  html.Div("Number of traders believes that Wilder’s thresholds are too wide and modify them for example \
                  into 80 and 20 for overbought and oversold, respectively.",
                           style={'color':'blue',
                                  'fontSize':11})],
                 width={"size": 3,
                        "offset": 0})]),
         dbc.Row(dbc.Col([html.Div([dcc.Graph(id='candle-plot'),
                                    dcc.Graph(id='tableRSI1'),
                                    dcc.Graph(id='tableRSI2'),
                                    dcc.Graph(id='tableRSI3'),
                                    dcc.Graph(id='tableRSI4')])],
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-6"),
         dbc.Row(dbc.Col([html.Div(id="update-table")],
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-7")])])
