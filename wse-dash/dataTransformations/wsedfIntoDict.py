import pandas as pd
#from .dict_path import dict_path_data # spolki z infosfera

import os

#HOME_DIR = os.chdir('/Users/michalozieblo/Desktop/WSE-demo/WSE-demo/wse-dash')
HOME_DIR = os.getcwd()

import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append(f'{HOME_DIR}/dataTransformations/')


from dict_path import dict_path_data

class KmeanOptions:

    def __init__(self):
        pass

    def wig20_options_for_kmean(self):

        path = dict_path_data['wse_wig20']
        wig20_df = pd.read_csv(path,
                               delimiter=";")
        wig20Abbrev = dict(zip(wig20_df['Nazwa giełdowa'],
                               wig20_df['Ticker']))
        wig20_options_for_kmean = []

        for key, value in wig20Abbrev.items():
            tmp_dict_1 = {'label': key}
            tmp_dict_2 = {'value': value}
            tmp_dict_3 = {**tmp_dict_1, **tmp_dict_2}
            wig20_options_for_kmean.append(tmp_dict_3)

        return wig20_options_for_kmean

    def mwig40_options_for_kmean(self):

        path = dict_path_data['wse_mwig40']
        mwig40_df = pd.read_csv(path,
                                delimiter=";")
        mwig40Abbrev = dict(zip(mwig40_df['Nazwa giełdowa'],
                                mwig40_df['Ticker']))
        mwig40_options_for_kmean = []

        for key, value in mwig40Abbrev.items():
            tmp_dict_1 = {'label': key}
            tmp_dict_2 = {'value': value}
            tmp_dict_3 = {**tmp_dict_1, **tmp_dict_2}
            mwig40_options_for_kmean.append(tmp_dict_3)

        return mwig40_options_for_kmean

    def swig80_options_for_kmean(self):

        path = dict_path_data['wse_swig80']
        swig80_df = pd.read_csv(path,
                                delimiter=";")
        swig80Abbrev = dict(zip(swig80_df['Nazwa giełdowa'],
                                swig80_df['Ticker']))
        swig80_options_for_kmean = []

        for key, value in swig80Abbrev.items():
            tmp_dict_1 = {'label': key}
            tmp_dict_2 = {'value': value}
            tmp_dict_3 = {**tmp_dict_1, **tmp_dict_2}
            swig80_options_for_kmean.append(tmp_dict_3)

        return swig80_options_for_kmean

    def wse_options_for_indicators(self):

        path = dict_path_data['wseDataframe']
        wse_df = pd.read_csv(path,
                             delimiter=";")
        wseAbbrev = dict(zip(wse_df['Nazwa giełdowa'],
                             wse_df['Ticker']))
        wse_options_for_kmean = []

        for key, value in wseAbbrev.items():
            tmp_dict_1 = {'label': key}
            tmp_dict_2 = {'value': value}
            tmp_dict_3 = {**tmp_dict_1, **tmp_dict_2}
            wse_options_for_kmean.append(tmp_dict_3)

        excluded_comp_tickers = ['aat', 'apl', 'bsc', 'arr', 'asm', 'atp', 'ats', 'atm', 'atr', 'aug', 'wis', 'bbd',
                                 'brg', 'bst', 'bik', 'bow', 'cpg', 'ene', 'cng', 'cts', 'cce', 'cmp', 'cpl', 'cpd',
                                 'opg', 'czt', 'dek', 'dtr', 'idm', 'dre', 'drp', 'dpl', 'edi', 'bdz', 'elt', 'emc',
                                 'est', 'enp', 'eni', 'egs', 'ehg', 'raf', 'ffi', 'gop', 'gbk', 'vin', 'gob', 'pce',
                                 'hmi', 'hld', 'hub', 'i2d', 'iia', 'ipl', 'imp', 'idg', 'inp', 'itb', 'inf', 'itm',
                                 'ipf', 'inl', 'ifc', 'ifr', 'inv', 'ipe', 'izb', 'jhm', 'jjo', 'kci', 'kdm', 'kbd',
                                 'kpd', 'kch', 'kri', 'kzs', 'krk', 'bkm', 'pri', 'prf', 'pma', 'pth', 'prd', 'prm',
                                 'pjp', 'prt', 'bal', 'wax', 'kmp', 'qnt', 'rdn', 'reg', 'rdh', 'rnc', 'rlp', 'res',
                                 'sgr', 'snw', 'iag', 'sco', 'swg', 'sek', 'sel', 'ses', 'svrs', 'sfg', 'skl', 'szl',
                                 'shd', 'son', 'sph', 'shg', 'stl', 'suw', 'tnx', 'tar', 'tmr', 'ksg', 'lrk', 'lkd',
                                 'lsi', 'mwt', '06n', 'mbw', 'mcp', 'mtl', 'meg', 'mnc', 'mlk', 'mlg', 'mbr', 'moj',
                                 'msw', 'mza', 'nng', 'ibs', 'nct', 'ntv', '08n', 'oex', 'fmg', 'opm', 'obl', 'ots',
                                 'ovo', 'nva', 'awb', 'pbg', 'pbf', 'pcg', 'pex', 'pgo', 'piw', 'plz', 'pgm', 'trr',
                                 'tow', 'tri', 'txm', 'ugg', 'u2k', 'vti', 'wxf', 'wik', 'yol', 'zre', 'kan', 'otm',
                                 'zuk', 'zmt', 'zst', 'zep']

        excluded_comp_tickers = [x.upper() for x in excluded_comp_tickers]

        for dictionary in wse_options_for_kmean:
            for key in dictionary:
                value = dictionary[key]
                try: delete = [key for key in dictionary if value in excluded_comp_tickers]
                except: print("wse_options_for_kmean ERROR - check wsedfIntoDict.py")

            for key in delete: del dictionary[key]

        wse_options_for_kmean = list(filter(None, wse_options_for_kmean))

        return wse_options_for_kmean
