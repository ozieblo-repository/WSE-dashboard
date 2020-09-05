import pandas as pd
import os
from datetime import datetime
import numpy as np
import sys

from wsedfIntoDict import KmeanOptions
from dict_path import dict_path_data

HOME_DIR = os.chdir('/Users/michalozieblo/Desktop/wse-dash-3')
HOME_DIR = os.getcwd()

END_REPORT_BUY = f'{HOME_DIR}/databases/buy_signal.csv'
END_REPORT_SELL = f'{HOME_DIR}/databases/sell_signal.csv'

sys.path.append(f'{HOME_DIR}/dataTransformations/')

############################################### Stochastic Oscillator ##################################################

def so_output(value):

    '''
    Take data for the last given day and check if there is an active stochastic oscillator signal to buy or sell stocks
    of the company. Used by signal_reports_homepage() function.

    @author: Michal Ozieblo

    :param value: takes string type abbreviation of the company name for which calculation is processed
    :return: dictionary with data type object if SO type signal is active and a few selected values used
             for visualization
    '''

    # if value is None: raise PreventUpdate # line to be deleted due to be outside of a callback decorator

    path = dict_path_data['wse_stocks'] # path to folder with .csv files with daily stock prices downloaded from Stooq
    df = pd.read_csv(os.path.join(path, r'%s.txt' % value), delimiter=',', index_col=[0]) # open csv database with stock
    # prices specified for given company by value argument

    date_index = [] # create empty list used to create date index in the loop below

    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)

    # define needed arrays to calcualate the indicator value
    array_close = np.array(df['<CLOSE>'])
    array_high = np.array(df['<HIGH>'])
    array_low = np.array(df['<LOW>'])
    array_vol = np.array(df['<VOL>'])

    # Finding Highest Values within k Periods

    y = 0 # define null temporary variable used for the loops below
    kperiods=13 # kperiods = 14; it starts from 0; MAY BE IMPROVED TO IMPLEMENT IT AS INPUT VARIABLE -> TO BE CONSIDERED
    array_highest = [] # create empty list used to create an array with highest prices during given 14 day period in the
    # loop below

    # range scope of the below loop is from the beginning of an array till end index value minus kperiods
    for x in range(0, array_high.size-kperiods):
        z = array_high[y] # z variable takes first highest price, then during each next iteration takes the next value
        # till end of an array minus kperiod index
        for j in range(0, kperiods):
            # if the highest price is lower than the day after, check the next one in order during the next iteration
            if(z < array_high[y+1]): z = array_high[y+1]
            y=y+1 # increment value of y variable to take the day+2 highest price during a new loop above
        array_highest.append(z) # creating list highest of k periods
        y = y-(kperiods-1) # skip one from starting after each iteration

    #Finding Lowest Values within k Periods

    y = 0 # define again null temporary variable used for the loops below
    array_lowest = [] # create empty list used to create an array with lowest prices during given 14 day period in the
    # loop below

    # logic below is equal to the above used for highest values
    for x in range(0, array_low.size-kperiods):
        z = array_low[y]
        for j in range(0, kperiods):
            if(z > array_low[y+1]): z = array_low[y+1]
            y = y+1
        array_lowest.append(z) # creating list lowest of k periods
        y = y-(kperiods-1) # skip one from starting after each iteration

    # Finding %K Line Values

    Kvalue = [] # create an empty list for the loop below

    # for index from 13 to an end of the array length do the below loop, it starts from 13 due to 2 week period scope
    # of calculation as it needs take the value with the index for 13 days before

    for x in range(kperiods, array_close.size):

        # k variable is equal to the close price on the day number x minus the lowest on day with the index number x-13
        # multiplied by 100, then divided by highest price on the day x-13 minus lowest price on the day x-13
        k = ((array_close[x]-array_lowest[x-kperiods])*100/(array_highest[x-kperiods]-array_lowest[x-kperiods]))
        Kvalue.append(k)

    # Finding %D Line Values

    y = 0 # define again null temporary variable used for the loops below
    dperiods = 3 # dperiods for calculate d values
#HOTFIX 22Aug2020
    # Dvalue = [None, None] # two None objects included due to for loop indexation below, based on length of that list
    Dvalue = [0, 0]

    for x in range(0, len(Kvalue)-dperiods+1):
        sum = 0
        for j in range(0, dperiods):
            sum = Kvalue[y]+sum
            y = y+1
        mean = sum/dperiods
        Dvalue.append(mean) # d values for %d line adding in the list Dvalue
        y = y-(dperiods-1) # skip one from starting after each iteration

    # Finding SMA %D Line Values

    y = 0
    sma_dperiods = 3 # dperiods for calculate d values
    sma_Dvalue = [None, None] # two None objects included due to for loop indexation below, based on length of that list

    for x in range(0, len(Dvalue)-sma_dperiods+1):
        sum = 0
        for j in range(0, sma_dperiods):
            #HOTFIX 22Aug2020               if Dvalue[y] is None: Dvalue[y] = 0 # give zero value for two first items in the row
            sum = Dvalue[y]+sum
            y = y+1
        mean = sum/sma_dperiods
        sma_Dvalue.append(mean)
        y = y-(sma_dperiods-1)

    index_loop = -1 # take the last day

    overbought_so_fast = []
    oversold_so_fast = []
    overbought_so_slow = []
    oversold_so_slow = []

    if Dvalue[index_loop] > 80: overbought_so_fast.append(date_index[index_loop])
    if Dvalue[index_loop] < 20: oversold_so_fast.append(date_index[index_loop])
    if sma_Dvalue[index_loop] > 80: overbought_so_slow.append(date_index[index_loop])
    if sma_Dvalue[index_loop] < 20: oversold_so_slow.append(date_index[index_loop])

    so_results = {'ticker':value,
                  'lastClose':array_close[index_loop],
                  'volume':array_vol[index_loop],
                  'date':date_index[index_loop],
                  'overbought_so_fast':overbought_so_fast,
                  'oversold_so_fast':oversold_so_fast,
                  'overbought_so_slow':overbought_so_slow,
                  'oversold_so_slow':oversold_so_slow}

    return so_results

######################################################### RSI ##########################################################

def rsi_output(value):

    '''
    Take data for the last given day and check if there is an active relative strength index signal to buy or sell
    stocks of the company. Used by signal_reports_homepage() function.

    @author: Michal Ozieblo

    :param value: takes string type abbreviation of the company name for which calculation is processed
    :return: dictionary with data type object if RSI type signal is active
    '''

    # if value is None: raise PreventUpdate # line to be deleted due to be outside of a callback decorator

    path = dict_path_data['wse_stocks']
    df = pd.read_csv(os.path.join(path, r'%s.txt' % value), delimiter=',', index_col=[0])
    date_index = []

    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)

    window_length = 14 # window length for moving average
    close = df['<CLOSE>'] # get just the adjusted close
    delta = close.diff() # get the difference in price from previous step


    delta[1] = 0 # change NaN since it did not have a previous row to calculate the difference, needed to compare with common x axis

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

    index_loop = -1

    overboughtEWMA = []
    oversoldEWMA = []
    overboughtSMA = []
    oversoldSMA = []

    if RSI1.iloc[index_loop] > 70: overboughtEWMA.append(date_index[index_loop])
    if RSI1.iloc[index_loop] < 30: oversoldEWMA.append(date_index[index_loop])
    if RSI2.iloc[index_loop] > 70: overboughtSMA.append(date_index[index_loop])
    if RSI2.iloc[index_loop] < 30: oversoldSMA.append(date_index[index_loop])

    rsi_results = {'ticker':value,
                   'lastClose':close.iloc[index_loop],
                   'overboughtEWMA':overboughtEWMA,
                   'oversoldEWMA':oversoldEWMA,
                   'overboughtSMA':overboughtSMA,
                   'oversoldSMA':oversoldSMA}

    return rsi_results

##################################################### MACD ############################################################

def macd_output(value):

    '''
    Take data for the last given day and check if there is an active Moving Average Convergence Divergence signal to buy
    or sell stocks of the company. Used by signal_reports_homepage() function.

    @author: Mateusz Jeczarek

    :param value: takes string type abbreviation of the company name for which calculation is processed
    :return: dictionary with data type object if MACD type signal is active
    '''

    path = dict_path_data['wse_stocks']

    df = pd.read_csv(os.path.join(path, r'%s.txt' % value),
                    delimiter=',',
                    index_col=[0])

    date_index = []

    for i in df['<DATE>']:
        date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
        date_index.append(date)

    # Calculate the MACD and signal line indicators
    # Calculate the short term exponential moving average (EMA)
    ShortEMA = df['<CLOSE>'].ewm(span=12, adjust=False).mean()


    LongEMA = df['<CLOSE>'].ewm(span=26, adjust=False).mean() # calcualte the long term exponential moving average (EMA)
    MACD = ShortEMA - LongEMA # calculate the MACD line
    signal = MACD.ewm(span=9, adjust=False).mean() # calculate the signal line

    # Create new columns for the data
    df['MACD'] = MACD
    df['Signal Line'] = signal

    # Create a function to signal when to buy and sell an asset
    def buy_sell(signal):

        Buy = []
        Sell = []
        flag = -1

        for i in range(0, len(signal)):
            if df.iloc[i]['MACD'] > df.iloc[i]['Signal Line']:
                Sell.append(np.nan)
                if flag != 1:
                    Buy.append(signal.iloc[i]['<CLOSE>'])
                    flag = 1
                else:
                    Buy.append(np.nan)
            elif df.iloc[i]['MACD'] < df.iloc[i]['Signal Line']:
                Buy.append(np.nan)
                if flag != 0:
                    Sell.append(signal.iloc[i]['<CLOSE>'])
                    flag = 0
                else:
                    Sell.append(np.nan)
            else:
                Buy.append(np.nan)
                Sell.append(np.nan)

        return (Buy, Sell)

    buy_sell_df = buy_sell(df) # create buy and sell column

    df['Buy_Signal_price'] = buy_sell_df[0]
    df['Sell_Signal_price'] = buy_sell_df[1]

    # extract the day where the sell or buy signal emerged
    sellSignal = []
    buySignal = []
    if pd.isnull(df.iloc[-1]['Sell_Signal_price']) == False: sellSignal.append(date_index[-1])
    if pd.isnull(df.iloc[-1]['Buy_Signal_price']) == False: buySignal.append(date_index[-1])

    # create the dictionary with results
    macd_results = {'sellSignal':sellSignal,
                    'buySignal':buySignal}

    return macd_results

def signal_reports_homepage():

    '''
    Main function.
    Generates two reports (buy and sell) based on output from indicators analysis of stock prices time series.
    The reports are used later for creating visualisations.

    @author: Michal Ozieblo
    '''

    # create vars with paths to lists of WIG group
    path_wig_20 = dict_path_data['wse_wig20']
    path_wse_mwig40 = dict_path_data['wse_mwig40']
    path_wse_swig80 = dict_path_data['wse_swig80']

    # open each database with list of companies in each WIG group
    df_wig20 = pd.read_csv(path_wig_20, delimiter=';')
    df_mwig40 = pd.read_csv(path_wse_mwig40, delimiter=';')
    df_swig80 = pd.read_csv(path_wse_swig80, delimiter=';')

    # take abbreviations of companies to run analysis for each of them in abbrev_list list
    x = KmeanOptions()
    x = x.wse_options_for_indicators()
    abbrev_list = []
    for i,j in enumerate(x):
        abbrev_list.append(j['value'])

    # define structures for dataframes
    d = {'TICKER':[], 'SO FAST':[], 'SO SLOW':[], 'RSI EWMA':[], 'RSI SMA':[], 'MACD_buy':[], 'MACD_sell':[]}
    d2 = {'COMPANY':[], 'TICKER': [], 'GROUP': [], "LAST CLOSE": [], "VOLUME": [], "MARKET SECTOR": [], 'MACD': []}

    df_buy_to_save = pd.DataFrame(data=d2)
    df_sell_to_save = pd.DataFrame(data=d2)

    # for every given company from abbrev_list calculate signals
    for i,j in enumerate(abbrev_list):

        print("in progress: ", j)

        df = pd.DataFrame(data=d) # empty dataframe at the beginning of the loop to collect signal +/- signs as the result

        # empty dataframes at the beginning of the loop to collect dates if signal is active and other needed data to visualization
        df_buy = pd.DataFrame(data=d2)
        df_sell = pd.DataFrame(data=d2)

        # add the ticker
        df_buy.loc[0,'TICKER'] = [j]
        df_sell.loc[0, 'TICKER'] = [j]

        # lower() used due to default abbreviations given in capital letters
        if so_output(j.lower())['overbought_so_fast'] != []: df.loc[0,'SO FAST'] = "+"
        else: df.loc[0,'SO FAST'] = "-"

        if so_output(j.lower())['overbought_so_slow'] != []: df.loc[0,'SO SLOW'] = "+"
        else: df.loc[0,'SO SLOW'] = "-"

        if rsi_output(j.lower())['overboughtEWMA'] != []: df.loc[0,'RSI EWMA'] = "+"
        else: df.loc[0,'RSI EWMA'] = "-"

        if macd_output(j.lower())['buySignal'] != []: df.loc[0, 'MACD_buy'] = "Signal Confirmed"
        else: df.loc[0,'MACD_buy'] = "No Signal"

        if macd_output(j.lower())['sellSignal'] != []: df.loc[0, 'MACD_sell'] = "Signal Confirmed"
        else: df.loc[0,'MACD_sell'] = "No Signal"

        # open file as dataframe with data about market sectors (manually created by taking data from infosfera.com site)
        tmp_df_market_sector = pd.read_csv(f'{HOME_DIR}/databases/wseDataframe.csv', sep=';')
        print('wseDataframe.csv file created')

        # select only the needed row for given company
        tmp_df_market_sector = tmp_df_market_sector.loc[tmp_df_market_sector['Ticker'] == j]

        # if each version of SO gives buy signal, append needed values to df_buy
        if df.loc[0,'SO FAST'] == "+" and df.loc[0,'SO SLOW'] == "+" and df.loc[0,'RSI EWMA'] == "+":
            df_buy.loc[0, 'LAST CLOSE'] = so_output(j.lower())['lastClose']
            df_buy.loc[0, 'MARKET SECTOR'] = tmp_df_market_sector['Branża'].values
            df_buy.loc[0, 'COMPANY'] = tmp_df_market_sector['Emitent'].values
            df_buy.loc[0, 'VOLUME'] = so_output(j.lower())['volume']
            df_buy.loc[0, 'MACD'] = df.loc[0,'MACD_buy'] # add info if MACD confirms the signal

            # append proper WIG group
            if j in set(df_wig20['Ticker']): df_buy.loc[0,'GROUP'] = "WIG20"
            if j in set(df_mwig40['Ticker']): df_buy.loc[0,'GROUP'] = "mWIG40"
            if j in set(df_swig80['Ticker']): df_buy.loc[0,'GROUP'] = "sWIG80"
            if j not in set(df_swig80['Ticker']) and j not in set(df_wig20['Ticker']) and j not in set(df_mwig40['Ticker']): df_buy.loc[0,'GROUP'] = "other"

            # append df_buy to the final report (df_buy_to_save)
            df_buy_to_save = df_buy_to_save.append(df_buy, ignore_index = True)

        # if each version of SO gives sell signal, append needed values to df_sell
        if df.loc[0,'SO FAST'] == "-" and df.loc[0,'SO SLOW'] == "-" and df.loc[0,'RSI EWMA'] == "-":
            df_sell.loc[0, 'LAST CLOSE'] = so_output(j.lower())['lastClose']
            df_sell.loc[0, 'MARKET SECTOR'] = tmp_df_market_sector['Branża'].values
            df_sell.loc[0, 'COMPANY'] = tmp_df_market_sector['Emitent'].values
            df_sell.loc[0, 'VOLUME'] = so_output(j.lower())['volume']
            df_sell.loc[0, 'MACD'] = df.loc[0,'MACD_sell'] # add info if MACD confirms the signal

            # append proper WIG group
            if j in set(df_wig20['Ticker']): df_sell.loc[0,'GROUP'] = "WIG20"
            if j in set(df_mwig40['Ticker']): df_sell.loc[0,'GROUP'] = "mWIG40"
            if j in set(df_swig80['Ticker']): df_sell.loc[0,'GROUP'] = "sWIG80"
            if j not in set(df_swig80['Ticker']) and j not in set(df_wig20['Ticker']) and j not in set(df_mwig40['Ticker']): df_sell.loc[0,'GROUP'] = "other"

            # append df_sell to the final report (df_sell_to_save)
            df_sell_to_save = df_sell_to_save.append(df_sell, ignore_index = True)

        # save the results
        df_buy_to_save.to_csv(END_REPORT_BUY, encoding='utf-8')
        df_sell_to_save.to_csv(END_REPORT_SELL, encoding='utf-8')
        print('End: report files created. Please refresh the tab if web browser.')

signal_reports_homepage() # run the main function
