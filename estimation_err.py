from sklearn.model_selection import KFold
from tqdm import tqdm
from utils import *
from joblib import Parallel, delayed
from scipy.stats import norm
import os

from opt_algo import CVaR_opt_l1, cv_CVaR_opt_l1, val_CVaR_opt_l1, CVaR_opt_l1_n


######################################################################
#
# Data generating process
#
######################################################################
a = 1.
sigma = 0.5
def get_params(s, d):
    mu = np.zeros(d)
    mu[:s] = a
    Sigma = np.eye(d) * sigma**2
    return mu, Sigma


err_obs = []
err_tru = []

n_simu = 100
n_list = np.arange(20,110,10)

alpha = 0.8
nw = 10

def _run_simu(mu, Sigma, n, alpha, t_tru, w_tru):
    X = np.random.multivariate_normal(mu, Sigma, n) + 1.
    X_val = np.random.multivariate_normal(mu, Sigma, n) + 1.
    nw = np.sum(w_tru>0.)*2
    t, w, lam = CVaR_opt_l1_n(X, alpha, n_lam=100, 
                              short=False, norm=2, thres=nw/X.shape[1], n=nw)
    t += 1
    err_obs = np.sqrt(np.sum((w - w_tru)**2) + (t_tru - t)**2)
    return err_obs

path_save = 'result/'
os.makedirs(path_save, exist_ok=True)

err_tru = np.zeros(len(n_list))
err_obs = np.zeros((len(n_list), n_simu))
with Parallel(n_jobs=-1, verbose=1) as parallel:
    for i, n in enumerate(n_list):
        d = int(n**2/20)
        s = 5
        mu, Sigma = get_params(s, d)
        
        w_tru = np.zeros(d)
        w_tru[:s] = 1./s
        # w^TX ~ N(a||w||_1, sigma^2||w||_2^2) for w>=0
        t_tru = - a + norm.pdf(norm.ppf(alpha)) / (1-alpha) / np.sqrt(s) * sigma
        err_tru[i] = np.sqrt(s*np.log(d)/n)
    
        res = parallel(delayed(_run_simu)(mu, Sigma, n, alpha, t_tru, w_tru) for _ in range(n_simu))
        err_obs[i,:] = np.array(res)

with open('result/err_d_200.npz', 'wb') as f:
    np.savez(f, err_obs=err_obs, err_tru=err_tru)

    
err_tru = np.zeros(len(n_list))
err_obs = np.zeros((len(n_list), n_simu))
with Parallel(n_jobs=-1, verbose=1) as parallel:
    for i, n in enumerate(n_list):
        d = 200
        s = int(n**(1/2))
        mu, Sigma = get_params(s, d)
        
        w_tru = np.zeros(d)
        w_tru[:s] = 1./s
        # w^TX ~ N(a||w||_1, sigma^2||w||_2^2) for w>=0
        t_tru = - a + norm.pdf(norm.ppf(alpha)) / (1-alpha) / np.sqrt(s) * sigma
        err_tru[i] = np.sqrt(s*np.log(d)/n)
    
        res = parallel(delayed(_run_simu)(mu, Sigma, n, alpha, t_tru, w_tru) for _ in range(n_simu))
        err_obs[i,:] = np.array(res)

with open('result/err_s_5.npz', 'wb') as f:
    np.savez(f, err_obs=err_obs, err_tru=err_tru)
