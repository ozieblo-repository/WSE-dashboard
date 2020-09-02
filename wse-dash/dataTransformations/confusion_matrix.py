# file is not a part of the dashboard
# @author: Michal Ozieblo

# test package included in confusion_matrix_test folder

# import standard libraries
import pandas as pd
import os
from datetime import datetime
import numpy as np
import sys


# import internal packages
from wsedfIntoDict import KmeanOptions
from dict_path import dict_path_data

# set paths
#HOME_DIR = os.chdir('/Users/michalozieblo/Desktop/wse-dash-3')
HOME_DIR = os.getcwd()
CM_REPORT_BUY = f'{HOME_DIR}/databases/confusion_matrix_buy_signal.csv'
CM_REPORT_SELL = f'{HOME_DIR}/databases/confusion_matrix_sell_signal.csv'
sys.path.append(f'{HOME_DIR}/dataTransformations/')

# assign constants
LONG_TERM_SPREAD = 1.078
NUMBER_OF_DAYS_AHEAD = 20

class Confusion_matrix():

    def so_output(value):

        '''
        Modified function from abbreviation_companies.py used to collect statistics based on the confusion matrix
        :param value: string type abbreviation of the company name for which calculation is processed (capital letters)
        :return: list of dictionaries with dates of the record, close prices, and dates if the signal for given company
                 was active
        '''

        path = dict_path_data['wse_stocks'] # path to folder with files including daily stock prices from Stooq.com
        df = pd.read_csv(os.path.join(path, r'%s.txt' % value),
                         delimiter=',', index_col=[0]) # open csv database with stock prices specified for given company
                                                       # by value argument

        date_index = [] # create empty list used to create date index in the loop below

        for i in df['<DATE>']:
            date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
            date_index.append(date)

        # define needed arrays to calcualate the indicator value
        array_close = np.array(df['<CLOSE>'])
        array_high = np.array(df['<HIGH>'])
        array_low = np.array(df['<LOW>'])

        # Finding Highest Values within k Periods

        y = 0 # define null temporary variable used for the loops below
        kperiods=13 # kperiods = 14; it starts from 0;
        array_highest = [] # create empty list used to create an array with highest prices during given 14 day period
                           # in the loop below

        # range scope of the below loop is from the beginning of an array till end index value minus kperiods
        for x in range(0, array_high.size-kperiods):
            z = array_high[y] # z variable takes first highest price, then during each next iteration takes the next
                              # value till end of an array minus kperiod index
            for j in range(0, kperiods):
                # if the highest price is lower than the day after, check the next one in order during the next iteration
                if(z < array_high[y+1]): z = array_high[y+1]
                y=y+1 # increment value of y variable to take the day+2 highest price during a new loop above
            array_highest.append(z) # creating list highest of k periods
            y = y-(kperiods-1) # skip one from starting after each iteration

        #Finding Lowest Values within k Periods

        y = 0 # define again null temporary variable used for the loops below
        array_lowest = [] # create empty list used to create an array with lowest prices during given 14 day period
                          # in the loop below

        # logic below is equal to the above used for highest values
        for x in range(0, array_low.size-kperiods):
            z = array_low[y]
            for j in range(0, kperiods):
                if(z > array_low[y+1]): z = array_low[y+1]
                y = y+1
            array_lowest.append(z) # creating list lowest of k periods
            y = y-(kperiods-1) # skip one from starting after each iteration

        # Finding %K Line Values

        Kvalue = []
        for x in range(kperiods,array_close.size):
           k = ((array_close[x]-array_lowest[x-kperiods])*100/(array_highest[x-kperiods]-array_lowest[x-kperiods]))
           Kvalue.append(k)

        # Finding %D Line Values

        y = 0 # define again null temporary variable used for the loops below
        dperiods = 3 # dperiods for calculate d values
        Dvalue = [None, None]

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
        sma_Dvalue = [None, None]

        for x in range(0, len(Dvalue)-sma_dperiods+1):
            sum = 0
            for j in range(0, sma_dperiods):
                if Dvalue[y] is None: Dvalue[y] = 0
                sum = Dvalue[y]+sum
                y = y+1
            mean = sum/sma_dperiods
            sma_Dvalue.append(mean)
            y = y-(sma_dperiods-1)

        index_loop = -1

        confusion_matrix_so_results_list = []

        for i in range(111):

            overbought_so_fast = []
            if Dvalue[index_loop] > 80: overbought_so_fast.append(date_index[index_loop])

            oversold_so_fast = []
            if Dvalue[index_loop] < 20: oversold_so_fast.append(date_index[index_loop])

            overbought_so_slow = []
            if sma_Dvalue[index_loop] > 80: overbought_so_slow.append(date_index[index_loop])

            oversold_so_slow = []
            if sma_Dvalue[index_loop] < 20: oversold_so_slow.append(date_index[index_loop])

            confusion_matrix_report_so_results = {'date':date_index[index_loop],
                                                  'close':array_close[index_loop],
                                                  'overbought_so_fast':overbought_so_fast,
                                                  'overbought_so_slow': overbought_so_slow,
                                                  'oversold_so_fast':oversold_so_fast,
                                                  'oversold_so_slow':oversold_so_slow}

            index_loop = index_loop - 1

            confusion_matrix_so_results_list.append(confusion_matrix_report_so_results)

        confusion_matrix_so_results_list.reverse()

        return confusion_matrix_so_results_list

    def rsi_output(value):

        '''
        Modified function from abbreviation_companies.py used to collect statistics based on the confusion matrix.

        :param value: string type abbreviation of the company name for which calculation is processed (capital letters)
        :return: a list with dictionaries including dates if the overbought signal was active
        '''

        path = dict_path_data['wse_stocks']
        df = pd.read_csv(os.path.join(path, r'%s.txt' % value), delimiter=',', index_col=[0])
        date_index = []

        for i in df['<DATE>']:
            date = datetime.strptime(str(i), '%Y%m%d').strftime('%m/%d/%Y')
            date_index.append(date)

        window_length = 14 # Window length for moving average
        close = df['<CLOSE>'] # Get just the adjusted close
        delta = close.diff() # Get the difference in price from previous step

        # Get rid of the first row, which is NaN since it did not have a previous
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

        index_loop = -1

        confusion_matrix_rsi_results_list = []

        for i in range(111):

            overboughtEWMA = []
            oversoldEWMA = []
            # overboughtSMA = []
            # oversoldSMA = []

            if RSI1.iloc[index_loop] > 70: overboughtEWMA.append(date_index[index_loop])
            if RSI1.iloc[index_loop] < 30: oversoldEWMA.append(date_index[index_loop])

            confusion_matrix_rsi_results = {'date': date_index[index_loop],
                                            'overboughtEWMA': overboughtEWMA,
                                            'oversoldEWMA': oversoldEWMA}

            index_loop = index_loop - 1

            confusion_matrix_rsi_results_list.append(confusion_matrix_rsi_results)

        confusion_matrix_rsi_results_list.reverse()

        return confusion_matrix_rsi_results_list

    def macd_output(value):

        '''
        FO FILL
        :param value:
        :return:
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

        # Calcualte the long term exponential moving average (EMA)
        LongEMA = df['<CLOSE>'].ewm(span=26, adjust=False).mean()

        # Calculate the MACD line
        MACD = ShortEMA - LongEMA

        # Calculate the signal line
        signal = MACD.ewm(span=9, adjust=False).mean()

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
                    else: Buy.append(np.nan)
                elif df.iloc[i]['MACD'] < df.iloc[i]['Signal Line']:
                    Buy.append(np.nan)
                    if flag != 0:
                        Sell.append(signal.iloc[i]['<CLOSE>'])
                        flag = 0
                    else: Sell.append(np.nan)
                else:
                    Buy.append(np.nan)
                    Sell.append(np.nan)

            return (Buy, Sell)

        # Create buy and sell column
        a = buy_sell(df)

        df['Buy_Signal_price'] = a[0]
        df['Sell_Signal_price'] = a[1]

        sellSignal = []
        buySignal = []

        index_loop = -1

        for i in range(111):

            # Create function to extract the day where the sell signal emerged
            if pd.isnull(df.iloc[index_loop]['Sell_Signal_price']) == False: sellSignal.append(date_index[index_loop])

            # Create function to extract the day where the buy signal emerged
            if pd.isnull(df.iloc[index_loop]['Buy_Signal_price']) == False: buySignal.append(date_index[index_loop])

            index_loop = index_loop - 1

        macd_results = {'sellSignal':sellSignal, 'buySignal':buySignal}

        return macd_results

    def confusion_matrix():

        '''
        Function to collect statistics based on the confusion matrix calculated depending on market indicators
        (stochastic oscillator, RSI and MACD) including the spread or not.

        :return: reports with results regarding the confusion matrix
        '''

        # take abbreviations of companies to run analysis for each of them
        x = KmeanOptions()
        x = x.wse_options_for_indicators()
        abbrev_list = []
        for i,j in enumerate(x): abbrev_list.append(j['value'])

        # assign initial structure for the temporary dataframe used to below calculations
        d = {'TICKER':[], 'DATE':[], 'SO FAST':[], 'SO SLOW':[], 'RSI EWMA':[], 'MACD_buy':[], 'MACD_sell':[]}

        # True Positive excluding/including MACD indicator signal if future price is above the spread
        TP_ABOVE_SPREAD_all_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_ABOVE_SPREAD_all = [0] * NUMBER_OF_DAYS_AHEAD

        # True Positive excluding/including MACD indicator signal if the price is below spread, but positive
        TP_BELOW_SPREAD_all_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_all = [0] * NUMBER_OF_DAYS_AHEAD

        # True Negative excluding/including MACD indicator signal
        TN_all_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TN_all = [0] * NUMBER_OF_DAYS_AHEAD

        # False Positive excluding/including MACD indicator signal
        FP_all_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FP_all = [0] * NUMBER_OF_DAYS_AHEAD

        # False Negative excluding/including MACD indicator signal
        FN_all_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FN_all = [0] * NUMBER_OF_DAYS_AHEAD

        TP_ABOVE_SPREAD_SOfast_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_ABOVE_SPREAD_SOfast = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_SOfast_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_SOfast = [0] * NUMBER_OF_DAYS_AHEAD
        TN_SOfast_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TN_SOfast = [0] * NUMBER_OF_DAYS_AHEAD
        FP_SOfast_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FP_SOfast = [0] * NUMBER_OF_DAYS_AHEAD
        FN_SOfast_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FN_SOfast = [0] * NUMBER_OF_DAYS_AHEAD

        TP_ABOVE_SPREAD_SOslow_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_ABOVE_SPREAD_SOslow = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_SOslow_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_SOslow = [0] * NUMBER_OF_DAYS_AHEAD
        TN_SOslow_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TN_SOslow = [0] * NUMBER_OF_DAYS_AHEAD
        FP_SOslow_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FP_SOslow = [0] * NUMBER_OF_DAYS_AHEAD
        FN_SOslow_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FN_SOslow = [0] * NUMBER_OF_DAYS_AHEAD

        TP_ABOVE_SPREAD_RSI_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_ABOVE_SPREAD_RSI = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_RSI_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_RSI = [0] * NUMBER_OF_DAYS_AHEAD
        TN_RSI_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TN_RSI = [0] * NUMBER_OF_DAYS_AHEAD
        FP_RSI_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FP_RSI = [0] * NUMBER_OF_DAYS_AHEAD
        FN_RSI_excl_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FN_RSI = [0] * NUMBER_OF_DAYS_AHEAD

        TP_ABOVE_SPREAD_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TP_BELOW_SPREAD_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        TN_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FP_MACD = [0] * NUMBER_OF_DAYS_AHEAD
        FN_MACD = [0] * NUMBER_OF_DAYS_AHEAD

        for j in abbrev_list:

            df = pd.DataFrame(data=d) # reset initial dataframe
            so_results = Confusion_matrix.so_output(j.lower()) # output abbreviations have capital letters,
                                                               # that's why .lower() is used
            macd_results = Confusion_matrix.macd_output(j.lower())
            rsi_rsults = Confusion_matrix.rsi_output(j.lower())

            # general rule: if signal for the indicator is positive, set +, if negative, set -
            for i in range(90):

                df.loc[i,'TICKER'] = j
                df.loc[i, 'DATE'] = so_results[i]['date']

                if so_results[i]['overbought_so_fast'] != []: df.loc[i,'SO FAST'] = "+"
                if so_results[i]['overbought_so_slow'] != []: df.loc[i,'SO SLOW'] = "+"
                if so_results[i]['oversold_so_fast'] != []: df.loc[i,'SO FAST'] = "-"
                if so_results[i]['oversold_so_slow'] != []: df.loc[i,'SO SLOW'] = "-"
                if rsi_rsults[i]['overboughtEWMA'] != []: df.loc[i,'RSI EWMA'] = "+"
                if rsi_rsults[i]['oversoldEWMA'] != []: df.loc[i,'RSI EWMA'] = "-"
                if so_results[i]['date'] in macd_results.get('buySignal'): df.loc[i, 'MACD_buy'] = "Signal Confirmed"
                if so_results[i]['date'] in macd_results.get('sellSignal'): df.loc[i, 'MACD_sell'] = "Signal Confirmed"

                if (df.loc[i,'SO FAST'] == "+" and df.loc[i,'SO SLOW'] == "+" and df.loc[i,'RSI EWMA'] == "+"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i+1+day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_all_excl_MACD[day_number] = TP_ABOVE_SPREAD_all_excl_MACD[day_number]+1
                        elif ((so_results[i]['close'] / so_results[i + 1+day_number]['close'] <= LONG_TERM_SPREAD
                              and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_all_excl_MACD[day_number] = TP_BELOW_SPREAD_all_excl_MACD[day_number]+1
                        else:
                            FP_all_excl_MACD[day_number] = FP_all_excl_MACD[day_number]+1

                if (df.loc[i, 'SO FAST'] == "+" and df.loc[i, 'SO SLOW'] == "+" and df.loc[i, 'RSI EWMA'] == "+"
                        and df.loc[i, 'MACD_buy'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_all[day_number] = TP_ABOVE_SPREAD_all[day_number]+1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                              and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_all[day_number] = TP_BELOW_SPREAD_all[day_number]+1
                        else:
                            FP_all[day_number] = FP_all[day_number]+1

                if (df.loc[i, 'SO FAST'] == "-" and df.loc[i, 'SO SLOW'] == "-" and df.loc[i, 'RSI EWMA'] == "-"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_all_excl_MACD[day_number] = TN_all_excl_MACD[day_number]+1
                        else:
                            FN_all_excl_MACD[day_number] = FN_all_excl_MACD[day_number]+1

                if (df.loc[i, 'SO FAST'] == "-" and df.loc[i, 'SO SLOW'] == "-" and df.loc[i, 'RSI EWMA'] == "-"
                        and df.loc[i, 'MACD_sell'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_all[day_number] = TN_all[day_number]+1
                        else:
                            FN_all[day_number] = FN_all[day_number]+1

                if (df.loc[i, 'SO FAST'] == "+"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_SOfast_excl_MACD[day_number]=TP_ABOVE_SPREAD_SOfast_excl_MACD[day_number]+1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_SOfast_excl_MACD[day_number]=TP_BELOW_SPREAD_SOfast_excl_MACD[day_number]+1
                        else:
                            FP_SOfast_excl_MACD[day_number] = FP_SOfast_excl_MACD[day_number] + 1

                if (df.loc[i, 'SO FAST'] == "+" and df.loc[i, 'MACD_buy'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_SOfast[day_number] = TP_ABOVE_SPREAD_SOfast[day_number] + 1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_SOfast[day_number] = TP_BELOW_SPREAD_SOfast[day_number] + 1
                        else:
                            FP_SOfast[day_number] = FP_SOfast[day_number] + 1

                if (df.loc[i, 'SO FAST'] == "-"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_SOfast_excl_MACD[day_number] = TN_SOfast_excl_MACD[day_number] + 1
                        else:
                            FN_SOfast_excl_MACD[day_number] = FN_SOfast_excl_MACD[day_number] + 1

                if (df.loc[i, 'SO FAST'] == "-" and df.loc[i, 'MACD_sell'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_SOfast[day_number] = TN_SOfast[day_number] + 1
                        else:
                            FN_SOfast[day_number] = FN_SOfast[day_number] + 1

                if (df.loc[i, 'SO SLOW'] == "+"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_SOslow_excl_MACD[day_number]=TP_ABOVE_SPREAD_SOslow_excl_MACD[day_number]+1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_SOslow_excl_MACD[day_number]=TP_BELOW_SPREAD_SOslow_excl_MACD[day_number]+1
                        else:
                            FP_SOslow_excl_MACD[day_number] = FP_SOslow_excl_MACD[day_number] + 1

                if (df.loc[i, 'SO SLOW'] == "+" and df.loc[i, 'MACD_buy'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_SOslow[day_number] = TP_ABOVE_SPREAD_SOslow[day_number] + 1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_SOslow[day_number] = TP_BELOW_SPREAD_SOslow[day_number] + 1
                        else:
                            FP_SOslow[day_number] = FP_SOslow[day_number] + 1

                if (df.loc[i, 'SO SLOW'] == "-"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_SOslow_excl_MACD[day_number] = TN_SOslow_excl_MACD[day_number] + 1
                        else:
                            FN_SOslow_excl_MACD[day_number] = FN_SOslow_excl_MACD[day_number] + 1

                if (df.loc[i, 'SO SLOW'] == "-" and df.loc[i, 'MACD_sell'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_SOslow[day_number] = TN_SOslow[day_number] + 1
                        else:
                            FN_SOslow[day_number] = FN_SOslow[day_number] + 1

                if (df.loc[i, 'RSI EWMA'] == "+"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_RSI_excl_MACD[day_number] = TP_ABOVE_SPREAD_RSI_excl_MACD[day_number] + 1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_RSI_excl_MACD[day_number] = TP_BELOW_SPREAD_RSI_excl_MACD[day_number] + 1
                        else:
                            FP_RSI_excl_MACD[day_number] = FP_RSI_excl_MACD[day_number] + 1

                if (df.loc[i, 'RSI EWMA'] == "+" and df.loc[i, 'MACD_buy'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_RSI[day_number] = TP_ABOVE_SPREAD_RSI[day_number] + 1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_RSI[day_number] = TP_BELOW_SPREAD_RSI[day_number] + 1
                        else:
                            FP_RSI[day_number] = FP_RSI[day_number] + 1

                if (df.loc[i, 'RSI EWMA'] == "-"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_RSI_excl_MACD[day_number] = TN_RSI_excl_MACD[day_number] + 1
                        else:
                            FN_RSI_excl_MACD[day_number] = FN_RSI_excl_MACD[day_number] + 1

                if (df.loc[i, 'RSI EWMA'] == "-" and df.loc[i, 'MACD_sell'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_RSI[day_number] = TN_RSI[day_number] + 1
                        else:
                            FN_RSI[day_number] = FN_RSI[day_number] + 1

                if (df.loc[i, 'MACD_buy'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > LONG_TERM_SPREAD:
                            TP_ABOVE_SPREAD_MACD[day_number] = TP_ABOVE_SPREAD_MACD[day_number] + 1
                        elif ((so_results[i]['close'] / so_results[i + 1 + day_number]['close'] <= LONG_TERM_SPREAD
                               and so_results[i]['close'] / so_results[i + 1 + day_number]['close']) > 1):
                            TP_BELOW_SPREAD_MACD[day_number] = TP_BELOW_SPREAD_MACD[day_number] + 1
                        else:
                            FP_MACD[day_number] = FP_MACD[day_number] + 1


                if (df.loc[i, 'MACD_sell'] == "Signal Confirmed"):
                    for day_number in range(NUMBER_OF_DAYS_AHEAD):
                        if (so_results[i]['close'] / so_results[i + 1 + day_number]['close']) < 1:
                            TN_MACD[day_number] = TN_MACD[day_number] + 1
                        else:
                            FN_MACD[day_number] = FN_MACD[day_number] + 1

            print("Processed company (abbreviation): ", j) # print present company abbreviation to follow processing

        def confusion_matrix(title,
                             TP_all, TP_BELOW_SPREAD_all, TN_all, FP_all, FN_all):

            # create empty lists to collect results
            # _spread suffix means that the output includes rule to calculate TP only if a future price is above the spread
            sensitivity_spread = []  # sensitivity, recall, hit rate, or true positive rate (TPR)
            sensitivity = []
            specificity = []  # specificity, selectivity or true negative rate (TNR)
            precision_spread = []  # precision or positive predictive value (PPV)
            precision = []
            npv = []  # negative predictive value (NPV)
            fnr_spread = []
            fnr = []  # miss rate or false negative rate (FNR)
            fpr = []  # miss rate or false positive rate (FPR)
            acc_spread = []  # accuracy (ACC)
            acc = []
            n_spread = []  # number of samples
            n = []
            f1 = []
            f1_spread = []

            # calculate the results for each day including MACD rule

            for i in range(20):

                try:
                    sensitivity_spread.append((TP_all[i] / (TP_all[i] + FN_all[i])) * 100)
                except ZeroDivisionError:
                    sensitivity_spread.append(0)
                try:
                    sensitivity.append(((TP_BELOW_SPREAD_all[i]
                                         + TP_all[i]) / (TP_BELOW_SPREAD_all[i] + TP_all[i] + FN_all[i])) * 100)
                except ZeroDivisionError:
                    sensitivity.append(0)

                try:
                    specificity.append((TN_all[i] / (TN_all[i] + FP_all[i])) * 100)
                except ZeroDivisionError:
                    specificity.append(0)

                try:
                    precision_spread.append((TP_all[i] / (TP_all[i] + FP_all[i])) * 100)
                except ZeroDivisionError:
                    precision_spread.append(0)

                try:
                    precision.append(((TP_BELOW_SPREAD_all[i] + TP_all[i])
                                      / (TP_BELOW_SPREAD_all[i] + TP_all[i] + FP_all[i])) * 100)
                except ZeroDivisionError:
                    precision.append(0)

                try:
                    npv.append((TN_all[i] / (TN_all[i] + FN_all[i])) * 100)
                except ZeroDivisionError:
                    npv.append(0)

                try:
                    fnr_spread.append((FN_all[i] / (FN_all[i] + TP_all[i])) * 100)
                except ZeroDivisionError:
                    fnr_spread.append(0)

                try:
                    fnr.append((FN_all[i] / (FN_all[i] + TP_BELOW_SPREAD_all[i] + TP_all[i])) * 100)
                except ZeroDivisionError:
                    fnr.append(0)

                try:
                    fpr.append((FP_all[i] / (FP_all[i] + TN_all[i])) * 100)
                except ZeroDivisionError:
                    fpr.append(0)

                try:
                    acc_spread.append(((TP_all[i] + TN_all[i]) / (TP_all[i] + TN_all[i] + FP_all[i] + FN_all[i])) * 100)
                except ZeroDivisionError:
                    acc_spread.append(0)

                try:
                    acc.append(((TP_BELOW_SPREAD_all[i] + TP_all[i] + TN_all[i])
                                / (TP_BELOW_SPREAD_all[i] + TP_all[i] + TN_all[i] + FP_all[i] + FN_all[
                        i])) * 100)
                except ZeroDivisionError:
                    acc.append(0)

                tmp = TP_all[i] + TN_all[i] + FP_all[i] + FN_all[i]
                n_spread.append(tmp)

                tmp = TP_BELOW_SPREAD_all[i] + TP_all[i] + TN_all[i] + FP_all[i] + FN_all[i]
                n.append(tmp)

                try:
                    f1.append(2 * (precision[i] * 0.01 * sensitivity[i] * 0.01) / (
                                precision[i] * 0.01 + sensitivity[i] * 0.01))
                except ZeroDivisionError:
                    f1.append(0)

                try:
                    f1_spread.append(2 * (precision_spread[i] * 0.01 * sensitivity_spread[i] * 0.01) / (
                                precision_spread[i] * 0.01 + sensitivity_spread[i] * 0.01))
                except ZeroDivisionError:
                    f1_spread.append(0)

            # set the meaningful index for the dataframe
            index = ['D+1', 'D+2', 'D+3', 'D+4', 'D+5', 'D+6', 'D+7', 'D+8', 'D+9', 'D+10',
                     'D+11', 'D+12', 'D+13', 'D+14', 'D+15', 'D+16', 'D+17', 'D+18', 'D+19', 'D+20']

            # create the dataframe to keep the results
            df_macd = pd.DataFrame({'SENSITIVITY': sensitivity, 'SPECIFICITY': specificity, 'PRECISION': precision,
                                    'NPV': npv, 'FNR': fnr, 'FPR': fpr, 'ACCURACY': acc, 'f1': f1, 'n': n,
                                    'SENSITIVITY_spread': sensitivity_spread, 'PRECISION_spread': precision_spread,
                                    'FNR_spread': fnr_spread, 'ACCURACY_spread': acc_spread, 'f1_spread': f1_spread,
                                    'n_spread': n_spread},
                                   index=index)
            df_macd = df_macd.round(1)
            df_macd = df_macd.transpose()

            df_macd.to_csv(f'/Users/michalozieblo/Desktop/WSE-dashboard-01092020/wse-dash/databases/confusion_matrix_{title}.csv',
                           encoding='utf-8')

        confusion_matrix('ALL_EXCL_MACD', TP_ABOVE_SPREAD_all_excl_MACD, TP_BELOW_SPREAD_all_excl_MACD,
                         TN_all_excl_MACD, FP_all_excl_MACD, FN_all_excl_MACD)
        confusion_matrix('ALL', TP_ABOVE_SPREAD_all, TP_BELOW_SPREAD_all, TN_all, FP_all, FN_all)
        confusion_matrix('SO_FAST', TP_ABOVE_SPREAD_SOfast, TP_BELOW_SPREAD_SOfast, TN_SOfast, FP_SOfast, FN_SOfast)
        confusion_matrix('SO_FAST_AND_MACD', TP_ABOVE_SPREAD_SOfast_excl_MACD, TP_BELOW_SPREAD_SOfast_excl_MACD,
                         TN_SOfast_excl_MACD,
                         FP_SOfast_excl_MACD, FN_SOfast_excl_MACD)
        confusion_matrix('SO_SLOW', TP_ABOVE_SPREAD_SOslow, TP_BELOW_SPREAD_SOslow, TN_SOslow, FP_SOslow, FN_SOslow)
        confusion_matrix('SO_SLOW_AND_MACD', TP_ABOVE_SPREAD_SOslow_excl_MACD, TP_BELOW_SPREAD_SOslow_excl_MACD,
                         TN_SOslow_excl_MACD, FP_SOslow_excl_MACD, FN_SOslow_excl_MACD)
        confusion_matrix('RSI', TP_ABOVE_SPREAD_RSI_excl_MACD, TP_BELOW_SPREAD_RSI_excl_MACD, TN_RSI_excl_MACD,
                         FP_RSI_excl_MACD, FN_RSI_excl_MACD)
        confusion_matrix('RSI_AND_MACD', TP_ABOVE_SPREAD_RSI, TP_BELOW_SPREAD_RSI, TN_RSI, FP_RSI, FN_RSI)
        confusion_matrix('MACD', TP_ABOVE_SPREAD_MACD, TP_BELOW_SPREAD_MACD, TN_MACD, FP_MACD, FN_MACD)

        print("Reports for Confusion Matrix are created.")

if __name__ == '__main__':
    Confusion_matrix.confusion_matrix()
