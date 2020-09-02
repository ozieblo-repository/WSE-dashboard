import os

#HOME_DIR = os.chdir('/Users/michalozieblo/Downloads/WSE-dashboard/wse-dash')

HOME_DIR = os.getcwd()

dict_path_data = {
    'wse_stocks' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse stocks',
    'wse_stocks_indicators' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse stocks indicators',
    'wse_indices' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse indices',
    'csv_files' :
        f'{HOME_DIR}/databases/csv-files',
    'wse_wig20' :
        f'{HOME_DIR}/databases/wse_wig20.csv',
    'wse_mwig40' :
        f'{HOME_DIR}/databases/wse_mwig40.csv',
    'wse_swig80' :
        f'{HOME_DIR}/databases/wse_swig80.csv',
    'wseDataframe' :
        f'{HOME_DIR}/databases/wseDataframe.csv',
    'wseStocks' :
        f'{HOME_DIR}/wseStocks',
    'SGHlogotypEN' :
        f'{HOME_DIR}/assets/SGHlogotypEN.png',
    'stooqlogo' :
        f'{HOME_DIR}/assets/stooqlogo.png',
    'Plotly_Dash_logo' :
        f'{HOME_DIR}/assets/Plotly_Dash_logo.png',
    'buy_signal' :
        f'{HOME_DIR}/databases/buy_signal.csv',
    'sell_signal' :
        f'{HOME_DIR}/databases/sell_signal.csv',
    'kmean' :
        f'{HOME_DIR}/databases/kmean_report.csv',
    'databases' :
        f'{HOME_DIR}/databases',
    'mwig40' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse indices/mwig40.txt',
    'swig80' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse indices/swig80.txt',
    'wig' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse indices/wig.txt',
    'wig20' :
        f'{HOME_DIR}/wseStocks/data/daily/pl/wse indices/wig20.txt'
}
