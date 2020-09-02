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
                         width={"size": 6, "offset": 3}),
                 id="so-row-0"),
         dbc.Row(dbc.Col(html.Div(html.H1("STOCHASTIC OSCILLATOR")),
                         width={"size": 6, "offset": 3}),
                 id="so-row-1"),
         dbc.Row(dbc.Col([html.Br(),
         html.Div(
         "The stochastic oscillator is a popular momentum indicator developed in the late 1950s \
         by George Lane. It collates a particular closing price of a security to its prices over \
         a specific period of time. It is used to herald reversals as the indicator reveals bullish \
         or bearish divergences. The indicator chart consists two lines: one reflecting the actual \
         value of the oscillator for each session, and second reflecting its simple moving average."
         ),
         html.Br()])),
        dbc.Row(dbc.Col(html.Div("Select the company to analyze:"),
                                 width={"size": 6, "offset": 3}),
                         id="so-row-2"),
         dbc.Row(dbc.Col(html.Div([
                            dcc.Dropdown(id='dropdown-stoch_osc',
                                         options=KmeanOptions().wse_options_for_indicators(),
                                         value='11B')
                                  ]),
             width={"size": 6, "offset": 3}),
             id="so-row-3"),

        dbc.Row([dbc.Col(


[html.Br(),

html.Span("Set N parameter:", style={"padding-right":"15px"}),
dcc.Input(
            id="input_n", type="number", placeholder="input with range",
            min=1, max=30, step=1, value=14
        ),
        html.P(),
html.Span("Set M parameter:", style={"padding-right":"13px"}),
dcc.Input(
            id="input_m", type="number", placeholder="input with range",
            min=1, max=10, step=1, value=3
        ),
        html.P(),
html.Span("Set O parameter:", style={"padding-right":"15px"}),
dcc.Input(
            id="input_o", type="number", placeholder="input with range",
            min=1, max=10, step=1, value=3
        ),



        html.Br()],
        width={"size": 3, "offset": 3}
        ),
        dbc.Col(
        [html.Br(),
        html.Div(
        "The sensitivity of the oscillator to market movements is reducible by \
        fitting N, M and O values. Low setting will make the indicator hypersensitive \
        to a market noise. It will offer a lot of signals, but many of them will be \
        a false positive. Popular settings (N.M.O) are: 5.3.3, 8.3.3 and 14.3.3, \
        always related to the specific asset. " ,
        style={'color':'blue', 'fontSize':11}
        )],
        width={"size": 3, "offset": 0})],
        id="so-row-nmo"),


         dbc.Row([dbc.Col([
html.Br(),
html.Br(),
html.Span("Set High threshold:", style={"padding-right":"15px"}),
dcc.Input(
            id="input_so_high_threshold", type="number", placeholder="input with range",
            min=50, max=99, step=1, value=80
        ),
        html.P(),
html.Span("Set Low threshold:", style={"padding-right":"19px"}),
dcc.Input(
            id="input_so_low_threshold", type="number", placeholder="input with range",
            min=1, max=49, step=1, value=20
        )
         ],
        width={"size": 3, "offset": 3}),
         dbc.Col([
html.Br(),
        html.Div(
        "A buy signal occurs at the time when the stochastic line enters below low threshold level (default = 20), \
        into the “oversold” area and then crosses above that threshold. Otherwise, a sell signal takes place when \
        the line moves above high threshold level (default = 80), into the “overbought” area and then crosses \
        below that threshold.",
        style={'color':'blue', 'fontSize':11}
        )],
        width={"size": 3, "offset": 0}
         )]),


         dbc.Row(dbc.Col([html.Div([dcc.Graph(id='candle-plot_stoch_osc'),
                                    dcc.Graph(id='tableSO1'),
                                    dcc.Graph(id='tableSO2'),
                                    dcc.Graph(id='tableSO3'),
                                    dcc.Graph(id='tableSO4')
                                    ])],
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-6"),
         dbc.Row(dbc.Col([html.Div(id="update-table")],
                         width={"size": 6,
                                "offset": 3}),
                 id="so-row-7")
         ])])
