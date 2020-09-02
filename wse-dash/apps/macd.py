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
                 id="macd-row-0"),
         dbc.Row(dbc.Col(html.Div(html.H1("MOVING AVERAGE CONVERGENCE / DIVERGENCE (MACD)")),
                         width={"size": 12, "offset": 0}),
                 id="macd-row-1"),
         dbc.Row(dbc.Col([html.Div(
         "Take into account the differences in values and the short-term exponential \
         class, connections between them by examining the convergence and divergence \
         of large moving. MACD is presented in the form of two lines: MACD and the \
         so-called signal. You can proceed to the recognition of divergences between \
         the connector and the price chart."
         ),
         html.P("")])),
        dbc.Row(dbc.Col(html.Div("Select the company to analyze:"),
                                 width={"size": 6, "offset": 3}),
                         id="bb-row-2"),
         dbc.Row(dbc.Col(html.Div([
                            dcc.Dropdown(id='dropdown-so',
                                         options=KmeanOptions().wse_options_for_indicators(),
                                         value='11B')
                                  ]),
             width={"size": 6, "offset": 3}),
             id="bb-row-3"),

        dbc.Row([dbc.Col(
            [html.Br(),
            html.Span("Set Short EMA:", style={"padding-right":"13px"}),
            dcc.Input(id="input_macd_n", type="number", placeholder="input with range",min=1, max=30, step=1, value=12),
             html.Br(),
             html.Br()],
            width={"size": 3, "offset": 3}),

            dbc.Col([html.Br(),html.Div("The MACD line is calculated by subtracting the 26-day exponential moving average (EMA) \
                                        from the 12-day EMA.",
        style={'color':'blue', 'fontSize':11})], width={"size": 3, "offset": 0})]
        ),

        dbc.Row([dbc.Col(
            [html.Br(),
            html.Span("Set Long EMA:", style={"padding-right":"13px"}),
            dcc.Input(id="input_macd_m", type="number", placeholder="input with range",min=1, max=30, step=1, value=26),
             html.Br(),
             html.Br()],
            width={"size": 3, "offset": 3}),

            dbc.Col([html.Br(),html.Div("The signal line is the average of the above-formed MACD line, usually an exponential \
                                        mean of period 9 is used.",
        style={'color':'blue', 'fontSize':11})], width={"size": 3, "offset": 0})]
        ),

         dbc.Row([dbc.Col(
             [html.Br(),
              html.Span("Set signal EMA:", style={"padding-right": "13px"}),
              dcc.Input(id="input_macd_s", type="number", placeholder="input with range", min=1, max=30, step=1, value=9),
              html.Br(),
              html.Br()],
             width={"size": 3, "offset": 3}),

             dbc.Col([html.Br(), html.Div("The signal line is the average of the above-formed MACD line, usually an exponential \
                                          mean of period 9 is used.",
                                          style={'color': 'blue', 'fontSize': 11})], width={"size": 3, "offset": 0})]
         ),

         dbc.Row(dbc.Col([html.Div([dcc.Graph(id='macd-plot'),
                                    dcc.Graph(id='tableMACD1'),
                                    dcc.Graph(id='tableMACD2')
                                    ])]),
                 id="bb-row-4"),
         dbc.Row(dbc.Col([html.Div(id="update-table")],
                         width={"size": 6,
                                "offset": 3}),
                 id="bb-row-5")
         ])])