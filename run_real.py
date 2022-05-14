from utils import *
from opt_algo import CVaR_opt_l1, cv_CVaR_opt_l1, val_CVaR_opt_l1, CVaR_opt_l1_n, GMV_opt
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_t
from tqdm import tqdm
import os
import sys
from sklearn.model_selection import KFold
from joblib import Parallel, delayed


# Read real data
path_data = '../spo/data/sp/'

df_close = pd.read_csv(path_data+'sp500_PRC.csv', index_col=[0])
df_open = pd.read_csv(path_data+'sp500_OPENPRC.csv', index_col=[0])
df = (df_close-df_open)/df_open

n_days_train = 126
n_days_hold = 63

T, d = df.shape

id_begin = np.where(df.index>=20110101)[0][0]
df_hold = pd.read_csv(path_data+'sp500_RET.csv', index_col=[0])
id_recal = np.arange(id_begin, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(0.)
n_recal = len(id_recal)

test_date = np.array(df.index[id_begin:])

df_listed = pd.read_csv(path_data+'sp500_listed.csv', index_col=[0])
df_listed.index = np.array(pd.to_datetime(df_listed.index).strftime('%Y%m%d')).astype(int)
id_codes_list = []
for idx in id_recal:
    codes = df_listed.columns[df_listed[(df_listed.index<df.index[idx])].iloc[-1,:]==1]
    codes = codes[~df.iloc[idx-n_days_train:idx,:].loc[:,codes].isnull().any()]
    id_codes_list.append(
        np.array([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
        )

# Hyperparameter    
alphas = np.arange(50,95,5)/100

methods = ['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'] + [
    'CVaR-1-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-2-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-L1-1-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-L1-2-%d'%(int(alpha*100)) for alpha in alphas]
short = False
nw = 40
def _run(X_train, method):
    if not method.startswith('CVaR'):
        w = GMV_opt(X_train, method, short=short)
    else:
            
        alpha = float(method.split('-')[-1]) / 100.
        norm = int(method.split('-')[-2])
        
        if method.startswith('CVaR-L1-'):
            _, w, lam = CVaR_opt_l1_n(X_train, alpha, n_lam=100, 
                                        short=short, norm=norm, thres=nw/X_train.shape[1], n=nw)

        else:                    
            w = CVaR_opt_l1(X_train, alpha, lam=0., 
                            short=short, norm=norm)
    return w


df_res = pd.DataFrame()
chunk_size = 40
i_chunk = int(sys.argv[1])
ws_list = np.zeros((len(id_recal),len(methods),d))
for i in tqdm(range(i_chunk*chunk_size, 
                    np.minimum((i_chunk+1)*chunk_size,len(id_recal)))
             ):
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]

    X_train = df.iloc[idx-n_days_train:idx,:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].fillna(0.).values[:,id_codes_list[i]].copy()
    w_list = []
    for method in methods:
        w = _run(X_train + 1., method)
        res = evaluate(X_test, w)
        res['Method'] = method
        res['i'] = i
        df_res = pd.concat([df_res, res])
        w_list.append(w)
    ws_list[i, :, id_codes_list[i]] = np.r_[w_list].T
    df_res.to_csv('result/res_real_in_{}.csv'.format(i_chunk))

    with open('result/ws_in_n_{}.npy'.format(nw), 'wb') as f:
        np.save(f, ws_list[i_chunk*chunk_size:np.minimum((i_chunk+1)*chunk_size,len(id_recal))])

