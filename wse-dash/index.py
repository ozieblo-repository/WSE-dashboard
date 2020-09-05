#### Import libraries and modules ####

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import (Input,
                               Output,
                               State)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import os
import numpy as np
import pandas as pd
from datetime import datetime

from app import app
from apps import (homepage,
                  attention_tab,
                  so,
                  rsi,
                  authors,
                  kmean,
                  macd,
                  bb)

from dataTransformations.dict_path import dict_path_data

HOME_DIR = os.getcwd()

app.layout = html.Div([dcc.Location(id='url', refresh=False), html.Div(id='page-content')])

################################################## NAVIGATION BAR #####################################################
@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/homepage': return homepage.layout
    elif pathname == '/authors': return authors.layout
    elif pathname == '/wac': return attention_tab.layout
    elif pathname == '/so': return so.layout
    elif pathname == '/rsi': return rsi.layout
    elif pathname == '/macd': return macd.layout
    elif pathname == '/bb': return bb.layout
    elif pathname == '/kmean': return kmean.layout
    elif pathname == '/nn': return app2.layout
    elif pathname is None: return '404'
    else: return homepage.layout

############################################### KMEAN CLUSTERING ######################################################
# kmean.py - Run searching of clusters (table creation), to be restructured into DataTable way and marge for multiple output with the next callback below
@app.callback(Output('datatable','srcDoc'), [Input('submit-button','n_clicks')], [State('datatable','value')]
, prevent_initial_call=True)
def update_datatable(n_clicks, csv_file):

    # to do: nie moze tego wywolywac, wtedy przepala caly clustering andzien dobry. błąd > niech funkcja tworzy csv i ten plik .py odpalic z wywolania w callbacku
    from dataTransformations.kmeansclustering import KMeansClustering

    if n_clicks is None: raise dash.exceptions.PreventUpdate
    if n_clicks is not None:
        df = KMeansClustering()
        x = df.kMeansClustering()
        return x.to_html()

# kmean.py - Run searching of clusters
@app.callback(Output("collapse-clusters", "is_open"), [Input("submit-button", "n_clicks")], [State("collapse-clusters", "is_open")]
, prevent_initial_call=True)
def toggle_collapse(n, is_open):
    if n is None: raise dash.exceptions.PreventUpdate
    if n: return not is_open
    return is_open

#################################################### SCRAPPER #########################################################
# call to run the scrapper
@app.callback(Output('hidden-div', 'children'), [Input('run-scrapper-and-find-best-comp', 'n_clicks')]
, prevent_initial_call=True)
def run_script_onClick(n_clicks):
    if n_clicks is None: raise dash.exceptions.PreventUpdate
    if n_clicks is not None:
        print('test')
        # hashed to not raise the monitor alarm on the server during the dashboard development :)
        os.system('python3 /Users/michalozieblo/Desktop/WSE-demo/WSE-demo/wse-dash/scrapper.py')
        # os.system(f'python3 {HOME_DIR}/dataTransformations/attention_companies.py')
        return

##################################################### RSI #############################################################
# RSI visualization
@app.callback([Output('candle-plot', 'figure'),
               Output('tableRSI1', 'figure'),
               Output('tableRSI2', 'figure'),
               Output('tableRSI3', 'figure'),
               Output('tableRSI4', 'figure')],
              [Input('dropdown-so', 'value'),
               Input("input_rsi_n", "value"),
               Input("input_rsi_high_threshold", "value"),
               Input("input_rsi_low_threshold", "value")])
def multi_output(value,n,high_threshold,low_threshold):
    if value is None: raise PreventUpdate
    path = dict_path_data['wse_stocks']
    df = pd.read_csv(os.path.join(path, r'%s.txt' % value),
                     delimiter=',',
                     index_col=[0])
    date_index = []
    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)
    fig = make_subplots(rows=4, cols=1,
                        subplot_titles=('RSI EWMA', 'RSI SMA', 'Candlestick chart'),
                        shared_xaxes=True,
                        vertical_spacing=0.05)
    window_length = n # Window length for moving average
    close = df['<CLOSE>'] # Get just the adjusted close
    delta = close.diff() # Get the difference in price from previous step

    # to edit: Get rid of the first row, which is NaN since it did not have a previous
    # row to calculate the differences - nope, we need it to compare with common x axis
    delta[1] = 0

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up1 = up.ewm(span=window_length).mean()
    roll_down1 = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))

    # Calculate the SMA
    roll_up2 = up.rolling(window_length).mean()
    roll_down2 = down.abs().rolling(window_length).mean()

    # Calculate the RSI based on SMA
    RS2 = roll_up2 / roll_down2
    RSI2 = 100.0 - (100.0 / (1.0 + RS2))

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=RSI1[-90:-1],
                             mode='lines',
                             name='RSI EWMA'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=RSI2[-90:-1],
                             mode='lines',
                             name='RSI SMA'),
                  row=2, col=1)

    fig.add_trace(go.Candlestick(x=date_index[-90:-1],
                                 open=df[-90:-1]['<OPEN>'],
                                 high=df[-90:-1]['<HIGH>'],
                                 low=df[-90:-1]['<LOW>'],
                                 close=df[-90:-1]['<CLOSE>'],
                                 name='candlesticks'),
                  row=3, col=1)

    fig.update_layout(height=800,
                      title_text="{} RSI (EWMA and SMA) - last 90 trading days:".format(value),
                      shapes=[dict(type="line", xref="x1", yref="y1", x0=date_index[-90], y0=high_threshold,
                                   x1=date_index[-2], y1=high_threshold, line_width=1),
                              dict(type="line", xref="x1", yref="y1", x0=date_index[-90], y0=low_threshold,
                                   x1=date_index[-2], y1=low_threshold, line_width=1),
                              dict(type="line", xref="x2", yref="y2", x0=date_index[-90], y0=low_threshold,
                                   x1=date_index[-2], y1=low_threshold, line_width=1),
                              dict(type="line", xref="x2", yref="y2", x0=date_index[-90], y0=high_threshold,
                                   x1=date_index[-2], y1=high_threshold, line_width=1),
                              dict(type="rect", xref="x2", yref="y2", x0=date_index[-90], y0=high_threshold,
                                   x1=date_index[-2], y1=100, line_width=0, fillcolor="LightPink", opacity=0.3),
                              dict(type="rect", xref="x2", yref="y2", x0=date_index[-90], y0=low_threshold,
                                   x1=date_index[-2], y1=0, line_width=0, fillcolor="PaleTurquoise", opacity=0.3),
                              dict(type="rect", xref="x1", yref="y1", x0=date_index[-90], y0=high_threshold,
                                   x1=date_index[-2], y1=100, line_width=0, fillcolor="LightPink", opacity=0.3),
                              dict(type="rect", xref="x1", yref="y1", x0=date_index[-90], y0=low_threshold,
                                   x1=date_index[-2], y1=0, line_width=0, fillcolor="PaleTurquoise", opacity=0.3)])

    overboughtEWMA = []
    for i,j in enumerate(RSI1[-90:-1]):
        if j > high_threshold: overboughtEWMA.append(date_index[i-90])

    oversoldEWMA = []
    for i,j in enumerate(RSI1[-90:-1]):
        if j < low_threshold: oversoldEWMA.append(date_index[i-90])

    overboughtSMA = []
    for i,j in enumerate(RSI2[-90:-1]):
        if j > high_threshold: overboughtSMA.append(date_index[i-90])

    oversoldSMA = []
    for i,j in enumerate(RSI2[-90:-1]):
        if j < low_threshold: oversoldSMA.append(date_index[i-90])

    fig2 = go.Figure(data=[go.Table(header=dict(values=["Overbought momentum since last 90 days via EWMA"]),
                           cells=dict(values=[pd.DataFrame(overboughtEWMA)]))])

    fig3 = go.Figure(data=[go.Table(header=dict(values=["Oversold momentum since last 90 days via EWMA"]),
                           cells=dict(values=[pd.DataFrame(oversoldEWMA)]))])

    fig4 = go.Figure(data=[go.Table(header=dict(values=["Overbought momentum since last 90 days via SMA"]),
                           cells=dict(values=[pd.DataFrame(overboughtSMA)]))])

    fig5 = go.Figure(data=[go.Table(header=dict(values=["Oversold momentum since last 90 days via SMA"]),
                           cells=dict(values=[pd.DataFrame(oversoldSMA)]))])

    fig.update_layout(margin=dict(l=0, r=0, t=80, b=0))
    fig2.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig3.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig4.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig5.update_layout(margin=dict(l=0, r=20, t=0, b=30),
                       height=170)

    return fig, fig2, fig3, fig4, fig5

#### Stochastic Oscillator ####
@app.callback(
    [Output('candle-plot_stoch_osc', 'figure'),
     Output('tableSO1', 'figure'),
     Output('tableSO2', 'figure'),
     Output('tableSO3', 'figure'),
     Output('tableSO4', 'figure')],
    [Input('dropdown-stoch_osc', 'value'),
     Input("input_n", "value"), Input("input_m", "value"), Input("input_o", "value"),
     Input("input_so_high_threshold", "value"), Input("input_so_low_threshold", "value")])
def SO_output(value,n,m,o,high_threshold,low_threshold):
    if value is None:
        raise PreventUpdate

    path = dict_path_data['wse_stocks']

    df = pd.read_csv(os.path.join(path, r'%s.txt' % value),
                     delimiter=',',
                     index_col=[0])

    date_index = []

    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)

    fig = make_subplots(rows=4,
                        cols=1,
                        subplot_titles=('FAST', 'SLOW', 'Candlestick chart'),
                        shared_xaxes=True,
                        vertical_spacing=0.05)

    array_close = np.array(df['<CLOSE>'])
    array_high = np.array(df['<HIGH>'])
    array_low = np.array(df['<LOW>'])

    # Finding Highest Values within k Periods
    y=0
    # kperiods are 14 array start from 0 index
    kperiods=n-1
    array_highest=[]
    for x in range(0,array_high.size-kperiods):
        z=array_high[y]
        for j in range(0,kperiods):
            if(z<array_high[y+1]):
                z=array_high[y+1]
            y=y+1
        array_highest.append(z) # creating list highest of k periods
        y=y-(kperiods-1) # skip one from starting after each iteration

    #Finding Lowest Values within k Periods
    y=0
    array_lowest=[]
    for x in range(0,array_low.size-kperiods):
        z=array_low[y]
        for j in range(0,kperiods):
            if(z>array_low[y+1]):
                z=array_low[y+1]
            y=y+1
        # creating list lowest of k periods
        array_lowest.append(z)

        y=y-(kperiods-1) # skip one from starting after each iteration

    # Finding %K Line Values
    Kvalue=[]
    for x in range(kperiods,array_close.size):
       k = ((array_close[x]-array_lowest[x-kperiods])*100/(array_highest[x-kperiods]-array_lowest[x-kperiods]))
       Kvalue.append(k)

    # Finding %D Line Values
    y=0
    # dperiods for calculate d values
    dperiods=m
    Dvalue=[None,None]
    for x in range(0,len(Kvalue)-dperiods+1):
        sum=0
        for j in range(0,dperiods):
            sum=Kvalue[y]+sum
            y=y+1
        mean=sum/dperiods

        Dvalue.append(mean) # d values for %d line adding in the list Dvalue

        y=y-(dperiods-1) # skip one from starting after each iteration

    # Finding SMA %D Line Values
    y=0
    # dperiods for calculate d values
    sma_dperiods=o
    sma_Dvalue=[None,None]
    for x in range(0,len(Dvalue)-sma_dperiods+1):
        sum=0
        for j in range(0,sma_dperiods):
            if Dvalue[y] is None:
                Dvalue[y] = 0
            sum=Dvalue[y]+sum
            y=y+1
        mean=sum/sma_dperiods
        sma_Dvalue.append(mean)
        y=y-(sma_dperiods-1)

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=Kvalue[-90:-1],
                             mode='lines',
                             name='Kvalue'),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=Dvalue[-90:-1],
                             mode='lines',
                             name='Dvalue'),
                  row=1, col=1)


    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                                 y=Dvalue[-90:-1],
                                 mode='lines',
                                 name='Dvalue'),
                                 row=2, col=1)

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                                 y=sma_Dvalue[-90:-1],
                                 mode='lines',
                                 name='sma_Dvalue'),
                                 row=2, col=1)

    fig.add_trace(go.Candlestick(x=date_index[-90:-1],
                                 open=df[-90:-1]['<OPEN>'],
                                 high=df[-90:-1]['<HIGH>'],
                                 low=df[-90:-1]['<LOW>'],
                                 close=df[-90:-1]['<CLOSE>'],
                                 name='candlesticks'),
                                 row=3, col=1)

    fig.update_layout(height=800,
                      title_text="{} stochastic oscillator - last 90 trading days:".format(value),
                      shapes=[
                          dict(type="line",
                               xref="x1",
                               yref="y1",
                               x0=date_index[-90],
                               y0=high_threshold,
                               x1=date_index[-2],
                               y1=high_threshold,
                               line_width=1),
                          dict(type="line",
                               xref="x1",
                               yref="y1",
                               x0=date_index[-90],
                               y0=low_threshold,
                               x1=date_index[-2],
                               y1=low_threshold,
                               line_width=1),
                          dict(type="line",
                               xref="x2",
                               yref="y2",
                               x0=date_index[-90],
                               y0=low_threshold,
                               x1=date_index[-2],
                               y1=low_threshold,
                               line_width=1),
                          dict(type="line",
                               xref="x2",
                               yref="y2",
                               x0=date_index[-90],
                               y0=high_threshold,
                               x1=date_index[-2],
                               y1=high_threshold,
                               line_width=1),
                          dict(type="rect",
                               xref="x2",
                               yref="y2",
                               x0=date_index[-90],
                               y0=high_threshold,
                               x1=date_index[-2],
                               y1=100,
                               line_width=0,
                               fillcolor="LightPink",
                               opacity=0.3),
                          dict(type="rect",
                               xref="x2",
                               yref="y2",
                               x0=date_index[-90],
                               y0=low_threshold,
                               x1=date_index[-2],
                               y1=0,
                               line_width=0,
                               fillcolor="PaleTurquoise",
                               opacity=0.3),
                          dict(type="rect",
                               xref="x1",
                               yref="y1",
                               x0=date_index[-90],
                               y0=high_threshold,
                               x1=date_index[-2],
                               y1=100,
                               line_width=0,
                               fillcolor="LightPink",
                               opacity=0.3),
                          dict(type="rect",
                               xref="x1",
                               yref="y1",
                               x0=date_index[-90],
                               y0=low_threshold,
                               x1=date_index[-2],
                               y1=0,
                               line_width=0,
                               fillcolor="PaleTurquoise",
                               opacity=0.3)
                      ])

    overbought_so_fast = []
    for i,j in enumerate(Dvalue[-90:-1]):
        if j > high_threshold:
            overbought_so_fast.append(date_index[i-90])

    oversold_so_fast = []
    for i,j in enumerate(Dvalue[-90:-1]):
        if j < low_threshold:
            oversold_so_fast.append(date_index[i-90])

    fig2 = go.Figure(data=[go.Table(header=dict(values=["Overbought momentum since last 90 days - FAST"]),
                           cells=dict(values=[pd.DataFrame(overbought_so_fast)]))])

    fig3 = go.Figure(data=[go.Table(header=dict(values=["Oversold momentum since last 90 days - FAST"]),
                           cells=dict(values=[pd.DataFrame(oversold_so_fast)]))])

    overbought_so_slow = []
    for i,j in enumerate(sma_Dvalue[-90:-1]):
        if j > high_threshold:
            overbought_so_slow.append(date_index[i-90])

    oversold_so_slow = []
    for i,j in enumerate(sma_Dvalue[-90:-1]):
        if j < low_threshold:
            oversold_so_slow.append(date_index[i-90])

    fig4 = go.Figure(data=[go.Table(header=dict(values=["Overbought momentum since last 90 days - SLOW"]),
                           cells=dict(values=[pd.DataFrame(overbought_so_slow)]))])

    fig5 = go.Figure(data=[go.Table(header=dict(values=["Oversold momentum since last 90 days - SLOW"]),
                           cells=dict(values=[pd.DataFrame(oversold_so_slow)]))])

    fig.update_layout(margin=dict(l=0, r=0, t=80, b=0))
    fig2.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig3.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig4.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig5.update_layout(margin=dict(l=0, r=20, t=0, b=30),
                       height=170)

    return fig, fig2, fig3, fig4, fig5

##################################################### Bollinger Bands #####################################################

@app.callback(
    [Output('bb-plot', 'figure'),
     Output('tableBB1', 'figure'),
     Output('tableBB2', 'figure')],
    [Input('dropdown-so', 'value'),
     Input("input_bb_n", "value"),
     Input("input_bb_m", "value")])
def update_output(value, n, m):

    path = dict_path_data['wse_stocks']

    df = pd.read_csv(os.path.join(path, r'%s.txt' % value),
                    delimiter=',',
                    index_col=[0])

    date_index = []

    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)

    fig = make_subplots(vertical_spacing=0, rows=1, cols=1)

    window_length_bb = n
    standard_deviation = m

    df['MA20'] = df['<CLOSE>'].rolling(window=window_length_bb).mean()
    df['20dSTD'] = df['<CLOSE>'].rolling(window=window_length_bb).std()
    df['Upper_2'] = df['MA20'] + (df['20dSTD'] * standard_deviation)
    df['Upper_1'] = df['MA20'] + df['20dSTD']
    df['Lower_2'] = df['MA20'] - (df['20dSTD'] * standard_deviation)
    df['Lower_1'] = df['MA20'] - df['20dSTD']

    # Create a function to signal when to buy and sell an asset
    def buy_sell(signal):
        Buy = []
        Sell = []
        flag = -1

        for i in range(0, len(signal)):
            if signal['<CLOSE>'][i] > signal['Upper_2'][i]:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal['<CLOSE>'][i])
                    flag = 1
                else:
                    Buy.append(np.nan)
            elif signal['<CLOSE>'][i] < signal['Lower_2'][i]:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal['<CLOSE>'][i])
                    flag = 0
                else:
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)

        return (Buy, Sell)

    # Create buy and sell column
    a = buy_sell(df)
    df['Buy_Signal_price'] = a[0]
    df['Sell_Signal_price'] = a[1]

    fig.add_trace(go.Candlestick(x=date_index[-90:-1],
                                 open=df[-90:-1]['<OPEN>'],
                                 high=df[-90:-1]['<HIGH>'],
                                 low=df[-90:-1]['<LOW>'],
                                 close=df[-90:-1]['<CLOSE>'],
                                 name='candlesticks'))

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['MA20'],
                             name='MA20 Line'))

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['Upper_2'],
                             name='Upper Line_2'))

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['Lower_2'],
                             name="Lower Line_2"))

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['Upper_1'],
                             name='Upper Line'))

    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['Lower_1'],
                             name="Lower Line"))

    fig.update_layout(height=800,
                      width=1200,
                      title_text="{} Bollinger Bands for the last 90 trading days:".format(value),
                      showlegend=True,
    )
    # Create function to extract the day where the sell signal emerged
    sellSignal = []
    for i, j in enumerate(df[-90:-1]['Sell_Signal_price']):
        if pd.isnull(j) == False:
            sellSignal.append(date_index[i - 90])

    # Create function to extract the day where the boy signal emerged
    buySignal = []
    for i, j in enumerate(df[-90:-1]['Buy_Signal_price']):
        if pd.isnull(j) == False:
            buySignal.append(date_index[i - 90])

    # Create table with day where the sell signal emerged
    fig2 = go.Figure(data=[go.Table(header=dict(values=["Sell signals within last 90 days via BB"]),
                                    cells=dict(values=[pd.DataFrame(sellSignal)]))])

    # Create table with day where the boy signal emerged
    fig3 = go.Figure(data=[go.Table(header=dict(values=["Buy signals within last 90 days via BB"]),
                                    cells=dict(values=[pd.DataFrame(buySignal)]))])

    fig2.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)

    fig3.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    return fig, fig2, fig3

# supports MACD vizualizations

@app.callback(
    [Output('macd-plot', 'figure'),
     Output('tableMACD1', 'figure'),
     Output('tableMACD2', 'figure')],
    [Input('dropdown-so', 'value'),
     Input("input_macd_n", "value"),
     Input("input_macd_m", "value"),
     Input("input_macd_s", "value")])

def MACD_output(value, n, m, s):

    path = dict_path_data['wse_stocks']
    df = pd.read_csv(os.path.join(path, r'%s.txt' % value), delimiter=',', index_col=[0])
    date_index = []

    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)

    fig = make_subplots(rows=2,
                        cols=1,
                        subplot_titles=('Visually the MACD and Signal Line',
                                        'Visually the stock buy and sell signal'),
                        shared_xaxes=True,
                        vertical_spacing=0.1)

    # Windows length for moving average
    short_window_length = n
    long_window_length = m
    signal_window_length = s

    # Calculate the MACD and signal line indicators

    # Calculate the short term exponential moving average (EMA)
    ShortEMA = df['<CLOSE>'].ewm(span=short_window_length, adjust=False).mean()

    # Calcualte the long term exponential moving average (EMA)
    LongEMA = df['<CLOSE>'].ewm(span=long_window_length, adjust=False).mean()

    # Calculate the MACD line
    MACD = ShortEMA - LongEMA

    # Calculate the signal line
    signal = MACD.ewm(span=signal_window_length, adjust=False).mean()

    # Create new columns for the data
    df['MACD'] = MACD
    df['Signal Line'] = signal

    # Create a function to signal when to buy and sell an asset
    def buy_sell(signal):
        Buy = []
        Sell = []
        flag = -1

        for i in range(0, len(signal)):
            if signal['MACD'][i] > signal['Signal Line'][i]:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal['<CLOSE>'][i])
                    flag = 1
                else: Buy.append(np.nan)
            elif signal['MACD'][i] < signal['Signal Line'][i]:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal['<CLOSE>'][i])
                    flag = 0
                else: Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)

        return Buy, Sell

    # Create buy and sell column
    a = buy_sell(df)
    df['Buy_Signal_price'] = a[0]
    df['Sell_Signal_price'] = a[1]

    # Create plot Sell and Boy signal

    # Add MACD line to plot
    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['MACD'],
                             name="MACD Line"),
                  row=1,
                  col=1)

    # Add signal line to plot
    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['Signal Line'],
                             name="Signal Line"),
                  row=1,
                  col=1)

    # Create candle plot
    fig.add_trace(go.Candlestick(x=date_index[-90:-1],
                                 open=df[-90:-1]['<OPEN>'],
                                 high=df[-90:-1]['<HIGH>'],
                                 low=df[-90:-1]['<LOW>'],
                                 close=df[-90:-1]['<CLOSE>'],
                                 name='candlesticks'),
                  row=2,
                  col=1)

    # Add to candle plot close price line
    fig.add_trace(go.Scatter(x=date_index[-90:-1],
                             y=df[-90:-1]['<CLOSE>'],
                             name="Close Price"),
                  row=2,
                  col=1)

    # Add to candle plot markers with Boy signal
    fig.add_trace(go.Scatter(mode='markers',
                             marker_symbol='triangle-up',
                             x=date_index[-90:-1],
                             y=df[-90:-1]['Buy_Signal_price'],
                             marker=dict(color='rgba(85, 185, 39, 1)',
                                         size=20
                                         ),
                             name='Buy Signal Price'),
                  row=2,
                  col=1)

    # Add to candle plot markers with sell signal
    fig.add_trace(go.Scatter(mode='markers',
                             marker_symbol='triangle-down',
                             x=date_index[-90:-1],
                             y=df[-90:-1]['Sell_Signal_price'],
                             marker=dict(color='rgba(185, 39, 39, 1)',
                                         size=20
                                         ),
                             name='Sell Signal Price'),
                  row=2,
                  col=1)

    # Add to title for plots and horizontal line to macd plot
    fig.update_layout(title_text=" {} MACD for the last 90 trading days:".format(value),
                      shapes=[
                          dict(type="line", xref="x1", yref="y1",
                               x0=date_index[-90],
                               y0=0,
                               x1=date_index[-2],
                               y1=0,
                               line_width=2)],
                      width=1200,
                      height=1400)

    # Create function to extract the day where the sell signal emerged
    sellSignal = []
    for i, j in enumerate(df[-90:-1]['Sell_Signal_price']):
        if pd.isnull(j) == False: sellSignal.append(date_index[i - 90])

    # Create function to extract the day where the boy signal emerged
    buySignal = []
    for i, j in enumerate(df[-90:-1]['Buy_Signal_price']):
        if pd.isnull(j) == False: buySignal.append(date_index[i - 90])

    # Create table with day where the sell signal emerged
    fig2 = go.Figure(data=[go.Table(header=dict(values=["Sell signals within last 90 days via MACD"]),
                                    cells=dict(values=[pd.DataFrame(sellSignal)]))])

    # Create table with day where the boy signal emerged
    fig3 = go.Figure(data=[go.Table(header=dict(values=["Buy signals within last 90 days via MACD"]),
                                    cells=dict(values=[pd.DataFrame(buySignal)]))])

    fig2.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)
    fig3.update_layout(margin=dict(l=0, r=20, t=0, b=5),
                       height=170)

    return fig, fig2, fig3

if __name__ == '__main__': app.run_server(debug=True)
