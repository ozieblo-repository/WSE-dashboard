import urllib.request
import zipfile
import os
from dataTransformations.dict_path import dict_path_data
from dataTransformations.wsedfIntoDict import KmeanOptions

HOME_DIR = os.getcwd()
#
# x = KmeanOptions()
# x = x.wse_options_for_indicators()
#
# abbrev_list = []
# for i,j in enumerate(x):
#     abbrev_list.append(j['value'])
#
# print('update in progress...')
#
# # download .zip file with raw data
#
# class AppURLopener(urllib.request.FancyURLopener):
#     version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"
#
# print('update in progress...')
#
# urllib._urlopener = AppURLopener()
# urllib._urlopener.retrieve("https://static.stooq.pl/db/h/d_pl_txt.zip",
#                            "d_pl_txt.zip")
#
# print('update in progress...')
#
# # unzip only needed files
#
# archive = zipfile.ZipFile(f'{HOME_DIR}/d_pl_txt.zip')
# path = dict_path_data['wseStocks']
#
#
# path_stocks_catalogue = 'data/daily/pl/wse stocks/'
#
# for file in archive.namelist():
#     if file.startswith(path_stocks_catalogue):
#         archive.extract(file, path)
#         print(file)
#
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
#
# # a set of index names to download
# company_indices = set(['wig.txt', 'wig20.txt', 'mwig40.txt', 'swig80.txt'])
#
# # variable containing path
# path_indices_catalogue = 'data/daily/pl/wse indices/'
#
# # a loop that checks if the file comes from the appropriate folder and if it is
# # named according to the file defined earlier
# for file in archive.namelist():
#     if file.startswith(path_indices_catalogue) and os.path.basename(file) in company_indices:
#         archive.extract(file, path)
#         print(file)
#
# # variable containing path
# path_stocks_indicators = 'data/daily/pl/wse stocks indicators/'
#
# for file in archive.namelist():
#     if file.startswith(path_stocks_indicators) and os.path.basename(file)[-6:-4] == 'pe':
#         archive.extract(file, path)
#         print(file)
#
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
#
# myfile = "d_pl_txt.zip"
# try: os.remove(myfile) # delete used .zip file
# except OSError as e: print ("Error: %s - %s." % (e.filename, e.strerror))
#
# excluded_comp_tickers = ['aat', 'apl', 'bsc', 'arr', 'asm', 'atp', 'ats', 'atm', 'atr', 'aug', 'wis', 'bbd', 'brg',
#                          'bst', 'bik', 'bow', 'cpg', 'ene', 'cng', 'cts', 'cce', 'cmp', 'cpl', 'cpd', 'opg', 'czt',
#                          'dek', 'dtr', 'idm', 'dre', 'drp', 'dpl', 'edi', 'bdz', 'elt', 'emc', 'est', 'enp', 'eni',
#                          'egs', 'ehg', 'raf', 'ffi', 'gop', 'gbk', 'vin', 'gob', 'pce', 'hmi', 'hld', 'hub', 'i2d',
#                          'iia', 'ipl', 'imp', 'idg', 'inp', 'itb', 'inf', 'itm', 'ipf', 'inl', 'ifc', 'ifr', 'inv',
#                          'ipe', 'izb', 'jhm', 'jjo', 'kci', 'kdm', 'kbd', 'kpd', 'kch', 'kri', 'kzs', 'krk', 'bkm',
#                          'pri', 'prf', 'pma', 'pth', 'prd', 'prm', 'pjp', 'prt', 'bal', 'wax', 'kmp', 'qnt', 'rdn',
#                          'reg', 'rdh', 'rnc', 'rlp', 'res', 'sgr', 'snw', 'iag', 'sco', 'swg', 'sek', 'sel', 'ses',
#                          'svrs', 'sfg', 'skl', 'szl', 'shd', 'son', 'sph', 'shg', 'stl', 'suw', 'tnx', 'tar', 'tmr',
#                          'ksg', 'lrk', 'lkd', 'lsi', 'mwt', '06n', 'mbw', 'mcp', 'mtl', 'meg', 'mnc', 'mlk', 'mlg',
#                          'mbr', 'moj', 'msw', 'mza', 'nng', 'ibs', 'nct', 'ntv', '08n', 'oex', 'fmg', 'opm', 'obl',
#                          'ots', 'ovo', 'nva', 'awb', 'pbg', 'pbf', 'pcg', 'pex', 'pgo', 'piw', 'plz', 'pgm', 'trr',
#                          'tow', 'tri', 'txm', 'ugg', 'u2k', 'vti', 'wxf', 'wik', 'yol', 'zre', 'kan', 'otm', 'zuk',
#                          'zmt', 'zst', 'zep']
#
# path = dict_path_data['wse_stocks']
# for i in excluded_comp_tickers:
#     try: os.remove(os.path.join(path, r'%s.txt' % i))
#     except: print("ERROR IN scrapper.py")
#
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################
#
# path_2 = dict_path_data['wse_stocks_indicators']
# for i in excluded_comp_tickers:
#     try: os.remove(os.path.join(path_2, r'%s_pe.txt' % i))
#     except: print("ERROR IN scrapper.py")
#
# ######################################################################################################################
# ######################################################################################################################
# ######################################################################################################################

print('files updated')

os.system(f'python3 {HOME_DIR}/dataTransformations/attention_companies.py')