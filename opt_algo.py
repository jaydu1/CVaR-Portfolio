from utils import *
import numpy as np
import nonlinshrink as nls #nonlinear shrinkage
from sklearn.covariance import LedoitWolf #linear shrinkage
from sklearn.model_selection import KFold, TimeSeriesSplit
from scipy.linalg import block_diag

from joblib import Parallel, delayed
from cvxopt import matrix
from cvxopt import solvers
# import mosek


from statsmodels.robust.scale import qn_scale

######################################################################
#
# CVaR minimization with l1-regularizer
#
######################################################################
import cvxpy as cp


def CVaR_opt_l1(X, alpha, lam=0.,
                short=True, renormalize=True, norm=2):

    n, d = X.shape
    
    w = cp.Variable(d)
    t = cp.Variable(1)
    
    obj = t + cp.sum(cp.maximum(- X @ w - t, 0.))/(1-alpha)/n
    if lam>0.: obj += lam * cp.norm(w,1)
    objective = cp.Minimize(obj)
    constraints = [cp.norm(w, norm) <= 1.,]
    if not short:
        constraints += [w>=0]
    prob = cp.Problem(objective, constraints)

    result = prob.solve(solver='ECOS', verbose=False)
    w = np.array(w.value)

    if np.sum(w**2)>1e-4:
        if lam>0.:
            w[np.abs(w)<np.max(np.abs(w))/1e3] = 0.
        if not short:
            w[w<0.] = 0.
        if renormalize:
            w = w / np.sum(np.abs(w))
    else:
        w = np.zeros(d)
    return w



######################################################################
#
# Cross validation
#
######################################################################
def _fit(X_train, X_val, alpha, **kwargs):
    w = CVaR_opt_l1(X_train, alpha, **kwargs)

    return np.array([np.sum(w!=0.), cvar( - np.dot(X_val, w), alpha)])

def _cv(X_train, X_val, alpha, lams, **kwargs):
    with Parallel(n_jobs=-1, verbose=0) as parallel:
        res = parallel(delayed(_fit)(X_train, X_val, alpha, lam=lam, **kwargs) for lam in lams)
    return np.array(res)


def CVaR_opt_l1_n(X, alpha, n_lam=100, thres=0.5, n=50, **kwargs):
    
    # Determine delta_max and delta_min automatically
    lam_max = np.maximum(np.max(np.mean(np.abs(X), axis=0)) / (1-alpha), 1.) # assuming X>=0
    deltas = np.logspace(0, -2, 100)
    lams = deltas * lam_max
    with Parallel(n_jobs=-1, verbose=0) as parallel:
        res = parallel(delayed(_fit)(X, X, alpha, lam=lam, **kwargs) for lam in lams)
    res = np.array(res)[:,0]
    
    if np.any(res!=0):
        i = np.maximum(np.where(res!=0)[0][0]-1, 0)
        delta_max = deltas[i]
        
        t = int(X.shape[1]*thres)
        if res[i]<t:
            if np.any(res>=t):
                i = np.where(res>=t)[0][0]
                delta_min = np.mean(deltas[i:i+1] )
            else:
                delta_min = deltas[-1] * 0.5
        else:
            delta_min = deltas[i+1]            
    else:
        delta_max = deltas[-1]
        delta_min = delta_max * 0.5

    # first fit
    lams = np.logspace(np.log10(lam_max*delta_min), np.log10(lam_max*delta_max), num=n_lam)[::-1]
    res = _cv(X, X, alpha, lams, **kwargs)
    id_lam = np.argmin(res[:,1])
    
    if np.any(res>=n):
        i = np.where(res>=n)[0][0]        
    else:
        i = -1
    delta_min = lams[i]/lam_max    
    delta_max = lams[np.where(res>0)[0][0]]/lam_max
    
    if delta_max<=delta_min:
        w = np.zeros(X.shape[1])
        lam = np.nan
    else:
        # refit
        lams = np.logspace(np.log10(lam_max*delta_min), np.log10(lam_max*delta_max), num=n_lam)[::-1]
        res = _cv(X, X, alpha, lams, **kwargs)    
        id_lam = np.argmin(res[(0<res[:,0])&(res[:,0]<=n),1])
        lam = lams[(0<res[:,0])&(res[:,0]<=n)][id_lam]

        w = CVaR_opt_l1(X, alpha, lam, **kwargs)
    w0 = var( - np.dot(X, w), alpha)
    return w0, w, lam


def val_CVaR_opt_l1(X, X_val, alpha, n_lam=100, thres=0.5, **kwargs):

    # Determine delta_max and delta_min automatically
    lam_max = np.maximum(np.max(np.mean(np.abs(X), axis=0)) / (1-alpha), 1.) # assuming X>=0
    deltas = np.logspace(0, -2, 100)
    lams = deltas * lam_max
    with Parallel(n_jobs=-1, verbose=0) as parallel:
        res = parallel(delayed(_fit)(X, X, alpha, lam=lam, **kwargs) for lam in lams)
    res = np.array(res)[:,0]
    
    if np.any(res!=0):
        i = np.maximum(np.where(res!=0)[0][0]-1, 0)
        delta_max = deltas[i]
        
        t = int(X.shape[1]*thres)
        if res[i]<t:
            if np.any(res>=t):
                i = np.where(res>=t)[0][0]
                delta_min = np.mean(deltas[i:i+1] )
            else:
                delta_min = deltas[-1] * 0.5
        else:
            delta_min = deltas[i+1]
            
    else:
        delta_max = deltas[-1]
        delta_min = delta_max * 0.5

    lams = np.logspace(np.log10(lam_max*delta_min), np.log10(lam_max*delta_max), num=n_lam)[::-1]
    res = _cv(X, X_val, alpha, lams, **kwargs)
    id_lam = np.argmin(res[:,1])
    
    with Parallel(n_jobs=-1, verbose=0) as parallel:
        res = parallel(delayed(_fit)(X, X, alpha, lam=lam, **kwargs) for lam in lams[id_lam:])
        res = np.array(res)
        if np.any(res[:,0]>0):
            lam = lams[id_lam:][res[:,0]>0][0]
        else:
            lam = lams[-1]
    w = CVaR_opt_l1(X, alpha, lam, **kwargs)
    w0 = var( - np.dot(X, w), alpha)
    return w0, w, lam


def cv_CVaR_opt_l1(X, alpha, n_folds=5, n_lam=100, ts_kfold=False, thres=0.5, **kwargs):
    
    # Determine delta_max and delta_min automatically
    lam_max = np.maximum(np.max(np.mean(np.abs(X), axis=0)) / (1-alpha), 1.)
    deltas = np.logspace(0, -2, 100)
    lams = deltas * lam_max
        
    with Parallel(n_jobs=-1, verbose=0) as parallel:
        res = parallel(delayed(_fit)(X, X, alpha, lam=lam, **kwargs) for lam in lams)
    res = np.array(res)[:,0]

    if np.any(res!=0):
        i = np.maximum(np.where(res!=0)[0][0]-1, 0)
        delta_max = deltas[i]
        
        t = int(X.shape[1]*thres)
        if res[i]<t:
            if np.any(res>=t):
                j = np.where(res>=t)[0][0]
                delta_min_global = deltas[j]
                delta_min = (delta_min_global + deltas[j-1])/2
            else:
                delta_min_global = deltas[-1]/10.
                delta_min = deltas[-1] * 0.5
        else:
            delta_min = deltas[i+1]
            delta_min_global = delta_min
            
    else:
        delta_max = deltas[-1]
        delta_min = delta_max * 0.5
        delta_min_global = deltas[-1]/10.
    
    while True:
        lams = np.logspace(np.log10(lam_max*delta_max), np.log10(lam_max*delta_min), num=n_lam)

        if ts_kfold:
            kf = TimeSeriesSplit(n_splits=n_folds)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)

        with Parallel(n_jobs=-1, verbose=0) as parallel:
            cvar_val = parallel(delayed(_cv)(
                X[train_index], X[val_index], alpha, lams, **kwargs
            ) for (train_index, val_index) in kf.split(X))
            cvar_val = np.array(cvar_val)[:,:,1]

        # if 0 actually obtains minimum, then no allocation
        # otherwise, lower lambda and keep searching
        if np.any(cvar_val!=0.):
            id_lam = np.argmin(np.mean(cvar_val, axis=0))
            lam = lams[id_lam]
            break
        else:
            if delta_min==delta_min_global:
                lam = lams[-1]
                break
            delta_min, delta_max = delta_min * 0.5, delta_min

    w = CVaR_opt_l1(X, alpha, lam, **kwargs)
    w0 = var( - np.dot(X, w), alpha)
    return w0, w, lam


######################################################################
#
# Global minimum variance optimization with l1-constraints
#
######################################################################
def GMV_opt(X, method, r_0=1., short=True):
    Sigma = cov(X, method)
    d = Sigma.shape[0]

    eps = 1e-6
    delta = np.finfo(np.float64).eps
    while eps<1.:
        try:
            if short:
                # variables [w, p] dimension [d, d]
                # min w^TXw
                # -p <= w <= p
                # -p <= 0
                # 1^Tp <= r_0
                P = matrix(block_diag(Sigma, np.zeros((d,d))), tc='d')
                q = matrix(np.zeros(2*d), tc='d')
                G = matrix(np.block([
                    [np.zeros((d,d)), -np.eye(d)], 
                    [np.eye(d), -np.eye(d)],
                    [-np.eye(d), -np.eye(d)]]), tc='d')
                h = matrix(np.zeros((3*d,1)), tc='d')
                A = matrix(np.c_[np.zeros((1,d)), np.ones((1,d))], tc='d')
                b = matrix(np.ones((1,1))*r_0, tc='d')
                
            else:
                # variables [w, p] dimension [d, d]
                # min w^TXw
                # -w <= 0
                # 1^Tw == r_0
                P = matrix(Sigma, tc='d')
                q = matrix(np.zeros(d), tc='d')
                G = matrix(-np.eye(d), tc='d')
                h = matrix(np.zeros(d), tc='d')
                A = matrix(np.ones((1,d)), tc='d')
                b = matrix(np.ones((1,1))*r_0, tc='d')

            sol = solvers.qp(P,q,G,h,A,b, options={'show_progress': False, 
                                      'abstol':1e-12, 'reltol':1e-11, 
                                      'maxiters':int(1e4), #'feastol':1e-16
                                                  })
            res = np.array(sol['x']).flatten()
            w = res[:d]
            if short:
                w[np.abs(w)<=delta] = 0.
                w = w/np.sum(np.abs(w))*r_0
            else:
                w[w<=delta] = 0.
                w = proj(w/r_0)*r_0
            
            break
        except:
            print('singular')
            Sigma = Sigma + np.identity(d) * eps
            eps *= 10
    
    return w


def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None)
    tmp = (np.cumsum(sort_w) - 1) * (1.0/np.arange(1,d+1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w

######################################################################
#
# Covariance matrix estimation
#
######################################################################
def Qn_corr(n):
    if n <= 12:
        return 1/ np.array(
            [.399356, .99365, .51321, .84401, .61220,
         .85877, .66993, .87344, .72014, .88906, .75743])[n-2]
    else:
        if n%2==1:
            c = 1.60188 +(-2.1284 - 5.172/n)/n
        else:
            c = 3.67561 +( 1.9654 +(6.987 - 77/n)/n)/n
        return c/n + 1 
    

def QNE(X, corr=True):
    n, d = X.shape
    if n == 1: return 0.
    Q = np.vectorize(
        lambda i:
        np.r_[np.zeros(i), (qn_scale(X[:,i:i+1]+X[:,i:])**2 - qn_scale(X[:,i:i+1]-X[:,i:])**2)/4], 
        signature='()->(n)')(np.arange(d))
    Q = Q + Q.T - np.diag(np.diag(Q))
    if corr:
        Q /= Qn_corr(n)**2

    eigval, eigvec = np.linalg.eig(Q)
    eigval[eigval < 1e-5] = 0.
    Q = eigvec.dot(np.diag(eigval)).dot(eigvec.T)
    return Q


def cov(X, method):
    '''
    Parameters
    ----------
    X : np.array
        The sample matrix with size \(n, p\).
    method : str
        The method used to estimate the covariance.

    Returns
    ----------
    Cov : np.array
        The estimated covariance matrix.
    '''
    if method.startswith('GMV-P'):
        return np.cov(X, rowvar = False)
    elif method.startswith('GMV-LS'):
        cov = LedoitWolf(assume_centered = False).fit(X) 
        return cov.covariance_
    elif method.startswith('GMV-NLS'):
        return nls.shrink_cov(X)
    elif method.startswith('QNE'):
        return QNE(X)


