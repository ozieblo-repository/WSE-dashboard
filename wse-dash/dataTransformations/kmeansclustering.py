from parameters import Parameters
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
import os

from wsedfIntoDict import KmeanOptions

HOME_DIR = os.getcwd()

KMEAN_CLUSTERING_REPORT = f'{HOME_DIR}/databases/kmean_report.csv'

class KMeansClustering:

    def kMeansClustering(self):

        tmp_abbreviations_of_companies = KmeanOptions()
        abbreviations_of_companies = tmp_abbreviations_of_companies.wse_options_for_indicators()

        abbrev_list = []
        for i,j in enumerate(abbreviations_of_companies):
            abbrev_list.append(j['value'])

        movements = Parameters(abbrev_list)

        daily_movement_object = movements.daily_movement()

        # Replace NaN with 0's:
        daily_movement_object.fillna(0)

        # a dataframe transformation into an array and the transpose of a matrix
        norm_movements = daily_movement_object.to_numpy()

        # Replace NaN with 0's:
        norm_movements[np.isnan(norm_movements)] = 0

        # Normalize samples individually to unit norm:
        norm_movements = normalize(norm_movements, axis=0)

        # data transpose due to the KMean algorythm specific construction (calculation on columns, not rows):
        norm_movements = norm_movements.transpose()

        # print of control information about data shape:
        print('daily_movement_object shape: ',daily_movement_object.shape)
        print('norm_movements shape: ',norm_movements.shape)

        # Test element-wise for Not a Number (NaN), return result as a bool array, change for 0 if True
        # (Impact on the result, but not significant on a large scale. Zero means no change in the price of the item \
        # on a given day):
        norm_movements[np.isnan(norm_movements)] = 0

        kmeans = KMeans(n_clusters = 24,
                        max_iter = 1000)

        condition = False

        HOME_DIR = os.getcwd()

        tmp_df_market_sector = pd.read_csv(f'{HOME_DIR}/databases/wseDataframe.csv', sep=';')

        kmeans.fit(norm_movements)
        labels = kmeans.predict(norm_movements)

        y = []

        for i,j in enumerate(abbrev_list):
            x = (tmp_df_market_sector['Ticker']==j)
            y.append(tmp_df_market_sector[tmp_df_market_sector['Branża'].where(x).notna()]['Branża'].values)

        df = pd.DataFrame({'Labels':labels,
                               'Companies':daily_movement_object.columns,
                               'Economic sector':y})

        df = df.astype({"Labels": int, "Companies": str, "Economic sector": str})

        df.sort_values(by=['Labels','Economic sector'], axis = 0, inplace=True)

        pd.set_option("display.max_rows", None, "display.max_columns", None)

        df.to_csv(KMEAN_CLUSTERING_REPORT, encoding='utf-8')

        return df

