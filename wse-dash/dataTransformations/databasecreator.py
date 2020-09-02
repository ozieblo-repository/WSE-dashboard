import pandas as pd
import os

# source to path to .csv files with historical values for each company
from parameters import path
from wsedfIntoDict import KmeanOptions
from dataTransformations.dict_path import dict_path_data

class DatabaseCreator:

    def __init__(self):
        pass

    def database_creator(self):

        '''
        The first and main process of the whole project. Creates the main database created by concatenation historical \
        data of mWIG40 companies in December 2019 with index based on date. Missed values related with a time before \
        company listing or suspended quotes included.

        abbreviation_values - List of full names of companies from the dictionary.

        pre_column_names - List of target prefix for each column merged with the company name shortcut, \
        translation due to polish names of each column title in source files.

        df_to_merge - Temporary dataframe used in a loop with loaded source data for each single company.

        df_to_merge_list - List of single dataframes with values for each single company.

        column_names_with_suffix - Temporary list used in a loop to create company specific column names.

        database - Dataframe with the main database.

        :return: `database.csv` - The main database.
        '''

        tmp_abbreviations_of_companies = KmeanOptions()
        abbreviations_of_companies = tmp_abbreviations_of_companies.wse_options_for_indicators()

        abbrev_list = []
        for i,j in enumerate(abbreviations_of_companies):
            abbrev_list.append(j['value'])

        pre_column_names = ['<TICKER>','<PER>','<TIME>','<OPEN>','<HIGH>','<LOW>','<CLOSE>','<VOL>','<OPENINT>']
        df_to_merge_list = []

        for j in abbrev_list:

            column_names_with_suffix = [(column_headline + '_%s' % j) for column_headline in pre_column_names]

            df_to_merge = pd.read_csv(os.path.join(path,r'%s.txt' % j),
                                      delimiter=',',
                                      index_col='<DATE>') # index set based on `Data` column is important to remember

            df_to_merge.index = pd.to_datetime(df_to_merge.index, format='%Y%m%d')

            df_to_merge.columns = column_names_with_suffix
            df_to_merge.index.names = ['<DATE>']

            df_to_merge_list.append(df_to_merge)

        database = pd.concat(df_to_merge_list,
                                 axis=1,
                                 sort=True) # sorting is important to successfull creation of the database

        for j in abbrev_list:
            del database[f'<PER>_{j}']
            del database[f'<TIME>_{j}']
            del database[f'<OPENINT>_{j}']

        print('Success! Main database created by concatenation historical data of mWIG40 companies in December 2019 ',
              'with index based on date.\n',
              'Warning! Missed values related with a time before company listing or suspended quotes included!')
        return database.to_csv(os.path.join(dict_path_data['databases'],r'database.csv'))

    if __init__ == "__main__":
        print("ADMIN INFO: DatabaseCreator run directly")
    else:
        print("ADMIN INFO: DatabaseCreator imported into another module")

#instance of the class and main function induction to local run:
dummy_instance_of_DatabaseCreator = DatabaseCreator()
dummy_instance_of_DatabaseCreator.database_creator()
