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
                 id="bb-row-0"),
         dbc.Row(dbc.Col(html.Div(html.H1("BOLLINGER BANDS")),
                         width={"size": 6, "offset": 4}),
                 id="bb-row-1"),
         dbc.Row(dbc.Col([html.Div(
         "The set consists of three elements. The first (and also the middle, if \
         you look at the resulting whole plot) is the moving average (SMA) with \
         a default value of 20. Two other components to the control lines, which \
         are drawn below and superior. Lower band to SMA minus two standard deviations. \
         Combined top band for SMA plus two standard deviations. There are opinions \
         that the standard value deviation is 2.5 thanks to which 99% of the price \
         action is between two ribbons, which means that the external injection \
         becomes a very significant signal."
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
            html.Span("Set Period:", style={"padding-right":"13px"}),
            dcc.Input(id="input_bb_n", type="number", placeholder="input with range",min=1, max=100, step=1, value=20),
             html.Br(),
             html.Br()],
            width={"size": 3, "offset": 3}),

            dbc.Col([html.Br(),html.Div("The 20-day calculation period is taken as a starting point. As the periods lengthen, \
                                        the number of standard deviations used should be increased.",
        style={'color':'blue', 'fontSize':11})], width={"size": 3, "offset": 0})]
        ),

        dbc.Row([dbc.Col(
            [html.Br(),
            html.Span("Set multiple standard deviation:", style={"padding-right":"13px"}),
            dcc.Input(id="input_bb_m", type="number", placeholder="input with range",min=0.5, max=5, step=0.5, value=2),
             html.Br(),
             html.Br()],
            width={"size": 3, "offset": 3}),

            dbc.Col([html.Br(),html.Div("2 standard deviations are considered standard. As the periods shorten or lengthen, \
                                        the number of standard deviations used should be reduced or increased. At 50 periods, \
                                        2.5 standard deviation is a good choice, while at 10 periods, 1.5 does quite well.",
        style={'color':'blue', 'fontSize':11})], width={"size": 3, "offset": 0})]
        ),

         dbc.Row(dbc.Col([html.Div([dcc.Graph(id='bb-plot'),
                                    dcc.Graph(id='tableBB1'),
                                    dcc.Graph(id='tableBB2')
                                    ])]),
                 id="bb-row-4"),
         dbc.Row(dbc.Col([html.Div(id="update-table")],
                         width={"size": 6,
                                "offset": 3}),
                 id="bb-row-5")
         ])])