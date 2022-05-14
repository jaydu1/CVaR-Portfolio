from utils import *
from opt_algo import CVaR_opt_l1, cv_CVaR_opt_l1, val_CVaR_opt_l1, GMV_opt
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_t
from tqdm import tqdm
import os

from sklearn.model_selection import KFold
from joblib import Parallel, delayed

n = 126
d = 250
n_simu = 100
alphas = np.arange(50,95,5)/100

s = 10
mu = np.r_[0.01*np.ones(s), np.zeros(d-s)]


# Read real data
path_data = '../spo/data/sp/'
# df = pd.read_csv(path_data+'sp500_RET.csv', index_col=[0])
# X_real = df[df.index>=20160101].dropna(axis=1)
# X_choice = X_real.sample(n = d, axis = 1).values
# Sigma = np.cov(X_choice, rowvar = False)
# with open('Sigma.npz', 'wb') as f:
#     np.savez(f, Sigma=Sigma)
with open('Sigma.npz', 'rb') as f:
    Sigma = np.load(f)['Sigma']
    
methods = ['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'] + [
    'CVaR-1-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-2-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-L1-1-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-L1-2-%d'%(int(alpha*100)) for alpha in alphas]
short = False

def _run(X_train, X_val, method):
    if not method.startswith('CVaR'):
        w = GMV_opt(X_train.copy(), method, short=short)
    else:
        alpha = float(method.split('-')[-1]) / 100.
        norm = int(method.split('-')[-2])
        
        if method.startswith('CVaR-L1'):
            _, w, _ = val_CVaR_opt_l1(X_train.copy(), X_val, alpha, short=short,
                                     norm=norm)
        else:                    
            w = CVaR_opt_l1(X_train.copy(), alpha, lam=0., 
                            short=short, norm=norm)
    return w

df = pd.DataFrame()
for data in ['normal', 't']:

    for i in tqdm(range(n_simu)):
        if data == 'normal':
            X_train = np.random.multivariate_normal(mu, Sigma, n)
            X_val = np.random.multivariate_normal(mu, Sigma, n)
            X_test = np.random.multivariate_normal(mu, Sigma, n)
        else:
            X_train = multivariate_t(mu, Sigma, df = 3).rvs(n)
            X_val = multivariate_t(mu, Sigma, df = 3).rvs(n)
            X_test = multivariate_t(mu, Sigma, df = 3).rvs(n)

        for method in methods:
            w = _run(X_train + 1., X_val + 1., method)
            # res = parallel(delayed(_run)(X_train, method) for method in methods)
            res = evaluate(X_test, w)
            res['Method'] = method
            res['i'] = i
            res['data'] = data
            df = pd.concat([df, res])
        df.to_csv('result/res_simu_%s.csv'%(data))


