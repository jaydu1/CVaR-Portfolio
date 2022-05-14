import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg

def var(loss, alpha):
    ind = int(np.ceil(alpha * len(loss))) - 1
    return np.partition(loss, ind)[ind]

def cvar(loss, alpha):
    t = var(loss, alpha)
    return t + np.mean(np.maximum(loss-t, 0.)) / (1-alpha)

def sr(ret):
    sd = np.std(ret)
    return 0. if sd==0. else np.mean(ret)/sd

def evaluate(X, w):
    alphas = np.arange(50,95,5)/100
    
    ret = np.dot(X, w)
    loss = - ret
    cum_ret = np.cumprod(ret+1.)
    max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
    
    df = pd.DataFrame(
        np.array([
            [cum_ret[-1], np.std(loss), np.mean(ret)/np.std(loss), max_drawdown, np.sum(w!=0.), np.sum(np.abs(w)>=1e-4)] +
            [var(loss, alpha) for alpha in alphas] + 
            [cvar(loss, alpha) for alpha in alphas]]),
        columns=['RET', 'STD', 'SR', 'MDD', 's', 's2'] + [
            'VaR-%d'%(int(alpha*100)) for alpha in alphas] + [
            'CVaR-%d'%(int(alpha*100)) for alpha in alphas]
    )
    return df