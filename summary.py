from utils import *
from opt_algo import CVaR_opt_l1, cv_CVaR_opt_l1, val_CVaR_opt_l1, GMV_opt
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_t
from tqdm import tqdm
import os

from sklearn.model_selection import KFold
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import seaborn as sns


#######################################################
#
# Error
#
#######################################################
sns.set(font_scale = 1.2)
fig, axes = plt.subplots(1,2, figsize=(12,3), sharey=True)

for i, name in enumerate(['s_5', 'd_200']):
    err_obs = []
    err_tru = []
    with open('result/err_{}.npz'.format(name), 'rb') as f:
        res = np.load(f)
        err_obs.append(res['err_obs'])
        err_tru.append(res['err_tru'])
    err_obs = np.concatenate(err_obs)
    err_tru = np.concatenate(err_tru)

    df = pd.DataFrame(
        err_obs
    )
    n_list = np.arange(10,100,10)
    df['n'] = n_list
    df = pd.melt(df, id_vars='n', value_vars=np.arange(100), var_name='i')

    axes[i] = sns.lineplot(data=df, x="n", y="value", legend='brief', label='l1-reguarlized CVaR', ax=axes[i])
    axes[i].plot(n_list, err_tru, label='Theoretical error bound')
    axes[i].set_ylabel('$\|\|\hat{\omega}_n-\omega^*\|\|$')
    axes[i].set_xlabel('$n$')
    axes[i].legend()
axes[0].set_title('$d=\\lfloor0.05 \\times n^2\\rfloor$, $s=5$')
axes[1].set_title('$d=200$, $s=\\lfloor n^{1/2}\\rfloor$')

fig.tight_layout()
plt.savefig('err.png', dpi=300, bbox_inches='tight', pad_inches=0) 


#######################################################
#
# Simulation
#
#######################################################

df_res = pd.DataFrame() 

for data in ['normal', 't']:
    _df = pd.read_csv('result/res_simu_{}_2.csv'.format(data)).iloc[:,1:]
    df_res = pd.concat([df_res, _df])
df_res = df_res[~(df_res['Method'].str.startswith('CVaR-1-') | df_res['Method'].str.startswith('CVaR-L1-1'))]

old_group = {
    'Benchmark':['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'],
    'CVaR':['CVaR-2-60','CVaR-2-70','CVaR-2-80','CVaR-2-90'],
    'CVaR-L1':['CVaR-L1-2-60','CVaR-L1-2-70','CVaR-L1-2-80','CVaR-L1-2-90']
}
group = {
    'Benchmark':['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'],
    'CVaR':['CVaR-0.6','CVaR-0.7','CVaR-0.8','CVaR-0.9'],
    'CVaR-L1':['CVaR-L1-0.6','CVaR-L1-0.7','CVaR-L1-0.8','CVaR-L1-0.9']
}

df_res['Method'] = df_res['Method'].replace({old_group[i][j]:group[i][j] for i in old_group for j in range(len(old_group[i]))})
    
    
df_alpha = pd.wide_to_long(df_res, stubnames=['VaR','CVaR'], i=['Method','i','data'], j='alpha', sep='-')
df_alpha = df_alpha.reset_index()


_df = df_alpha[df_alpha['Method'].isin([j for i in group for j in group[i]])]
for i in group:
    _df.loc[df_alpha['Method'].isin(group[i]), 'Group'] = i
    
    
sns.set_theme()

fig, axes = plt.subplots(2,3,figsize=(12,6), sharey=False)
cbar_ax = fig.add_axes([.92, .2, .02, .6])

for i,data in enumerate(['normal', 't']):

    axes[i,0] = sns.violinplot(y="Method", x="MDD", data=_df[_df['data']==data], ax=axes[i,0], orient='h')
    axes[0,0].set_title('Maximum Drawdown', fontsize=16)

    tmp = _df[_df['data']==data].groupby(["Method", "alpha"]).mean().reset_index().pivot("Method", "alpha", "CVaR")
    tmp.index = pd.CategoricalIndex(tmp.index, categories= [j for i in group for j in group[i]])
    tmp.sort_index(level=0, inplace=True)
    axes[i,1] = sns.heatmap(tmp, 
                     annot=False, ax=axes[i,1], vmin=-0.01, vmax=0.02, 
                         cmap=sns.color_palette("YlGn", as_cmap=True),
                         cbar=False,)
    axes[0,1].set_title('CVaR', fontsize=16)
    axes[i,1].set_yticks([])
    axes[i,1].set_ylabel('')
    axes[i,1].set_xticks(axes[i,1].get_xticks()[::2])
    
    tmp = _df[_df['data']==data].groupby(["Method", "alpha"]).mean().reset_index().pivot("Method", "alpha", "VaR")
    tmp.index = pd.CategoricalIndex(tmp.index, categories= [j for i in group for j in group[i]])
    tmp.sort_index(level=0, inplace=True)

    axes[i,2] = sns.heatmap(tmp, 
                     annot=False, ax=axes[i,2], vmin=-0.01, vmax=0.02,
                         cmap=sns.color_palette("YlGn", as_cmap=True),
                         cbar_ax=cbar_ax)
    axes[0,2].set_title('VaR', fontsize=16)
    axes[i,2].set_ylabel('')
    axes[i,2].set_yticks([])
    axes[i,2].set_xticks(axes[i,2].get_xticks()[::2])

axes[0,0].set_xlabel('')
axes[0,1].set_xlabel('')
axes[0,2].set_xlabel('')
axes[1,0].set_xlabel('value', fontsize=12)
axes[1,1].set_xlabel('$\\alpha$', fontsize=12)
axes[1,2].set_xlabel('$\\alpha$', fontsize=12)





left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height
axes[0,0].text(-0.45, 0.5*(bottom+top), 'Gaussian',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=axes[0,0].transAxes, color=sns.color_palette('deep')[2], fontsize=20)

axes[1,0].text(-0.45, 0.5*(bottom+top), 'Multivariate $t$',
        horizontalalignment='right',
        verticalalignment='center',
        rotation='vertical',
        transform=axes[1,0].transAxes, color=sns.color_palette('deep')[2], fontsize=20)



axes[0,0].text(-0.05, 1.08, '(a)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,0].transAxes, fontsize=16)
axes[0,1].text(-0.05, 1.08, '(b)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,1].transAxes, fontsize=16)
axes[0,2].text(-0.05, 1.08, '(c)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[0,2].transAxes, fontsize=16)
axes[1,0].text(-0.05, 1.08, '(d)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[1,0].transAxes, fontsize=16)
axes[1,1].text(-0.05, 1.08, '(e)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[1,1].transAxes, fontsize=16)
axes[1,2].text(-0.05, 1.08, '(f)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=axes[1,2].transAxes, fontsize=16)


fig.tight_layout(rect=[0, 0, .92, 1])
plt.savefig('simu.png', dpi=300, bbox_inches='tight', pad_inches=0) 


#######################################################
#
# Real Data
#
#######################################################


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
df_rf = pd.read_csv(path_data+'../rf.csv')

with open('result/ws_in_n_40.npy', 'rb') as f:
    ws_list = np.load(f)

alphas = np.arange(50,95,5)/100

methods = ['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'] + [
    'CVaR-1-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-2-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-L1-1-%d'%(int(alpha*100)) for alpha in alphas] + [
    'CVaR-L1-2-%d'%(int(alpha*100)) for alpha in alphas]



res = []
for j,method in enumerate(methods):
    score_test_list = []
    for i in range(len(id_recal)):#:
        i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
        idx = id_recal[i]
        X_test = df_hold.iloc[idx:idx+n_days_hold,:].fillna(0.).values[:,id_codes_list[i]].copy()
        score_test_list.append(
            X_test @ ws_list[i,j,id_codes_list[i]]
        )
    score_test_list = np.concatenate(score_test_list)# - 1.
    

    rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
    ret = score_test_list

    cum_ret = np.cumprod(ret+1)
    returns = cum_ret[-1] - 1

    max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
    ret = np.log(1+ret)
    sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
    print('%s & %.4f & %.4f & %.4f & %.4f & %d \\\\'%(
        method, returns, max_drawdown, sharpe_ratio, cvar(-ret, 0.9),
        int(np.mean(np.sum(ws_list[:,j,:]>1e-4, axis=1)))))
    res.append([method, returns, np.std(ret), 
                sharpe_ratio, max_drawdown, 
                int(np.mean(np.sum(ws_list[:,j,:]>1e-4, axis=1)))
               ] + 
                [var(-ret, alpha) for alpha in alphas] + 
                [cvar(-ret, alpha) for alpha in alphas]
    
    
df_res = pd.DataFrame(res, columns=['Method','RET', 'STD', 'SR', 'MDD', 's'] + [
            'VaR-%d'%(int(alpha*100)) for alpha in alphas] + [
            'CVaR-%d'%(int(alpha*100)) for alpha in alphas]
            )
df_res = df_res[~(#df_res['Method'].str.startswith('CVaR-1-') | 
                  df_res['Method'].str.startswith('CVaR-L1-1')
                 )]
old_group = {
    'Benchmark':['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'],
    'CVaR-1':['CVaR-1-%d'%(int(alpha*100)) for alpha in alphas],
    'CVaR-2':['CVaR-2-%d'%(int(alpha*100)) for alpha in alphas],
    'CVaR-L1':['CVaR-L1-2-%d'%(int(alpha*100)) for alpha in alphas]
}
group = {
    'Benchmark':['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'],
    'CVaR-1':['CVaR-1-%d'%(int(alpha*100)) for alpha in alphas],
    'CVaR-2':['CVaR-2-%d'%(int(alpha*100)) for alpha in alphas],
    'CVaR-L1':['CVaR-L1-%d'%(int(alpha*100)) for alpha in alphas]
}
df_res['Method'] = df_res['Method'].replace(
    {old_group[i][j]:group[i][j] for i in old_group for j in range(len(old_group[i]))})

df_tmp = df_res[df_res['Method'].isin([
    'GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE',
#     'CVaR-1-60','CVaR-1-70', 'CVaR-1-80', 'CVaR-1-90',
    'CVaR-2-60', 'CVaR-2-70', 'CVaR-2-80', 'CVaR-2-90', 
    'CVaR-L1-60', 'CVaR-L1-70', 'CVaR-L1-80', 'CVaR-L1-90'])]
df_tmp = df_tmp[['Method','STD','SR','MDD','VaR-60','VaR-70','VaR-80','CVaR-60','CVaR-70','CVaR-80','CVaR-90']]

methods = [['GMV-P', 'GMV-LS', 'CVaR-2-60', 'CVaR-2-70', 'CVaR-2-80', 'CVaR-2-90'],
    ['GMV-NLS', 'QNE', 'CVaR-L1-60', 'CVaR-L1-70', 'CVaR-L1-80', 'CVaR-L1-90']]
for i in range(6):
    print(
        methods[0][i] + ' & ' + ' & '.join([
        '%.02f'%j for j in df_tmp.loc[df_tmp['Method']==methods[0][i], ['SR','STD','MDD']].values[0]
               ]) + ' & ' +
        methods[1][i] + ' & ' + ' & '.join([
        '%.02f'%j for j in df_tmp.loc[df_tmp['Method']==methods[1][i], ['SR','STD','MDD']].values[0]
               ]) +
        ' \\\\')

    
    
    
df_tmp = df_res[df_res['Method'].isin([
    'GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE',
    'CVaR-L1-70','CVaR-L1-80','CVaR-L1-90'])]
df_alpha = pd.wide_to_long(df_tmp, stubnames=['VaR','CVaR'], i=['Method'], j='alpha', sep='-')
df_alpha = df_alpha.reset_index()
df_alpha = df_alpha[df_alpha['alpha']>=60]
df_alpha['alpha'] = df_alpha['alpha']/100
df_alpha['VaR'] = df_alpha['VaR']*100
df_alpha['CVaR'] = df_alpha['CVaR']*100


old_group = {
    'Benchmark':['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'],
    'CVaR-L1':['CVaR-L1-60','CVaR-L1-70','CVaR-L1-80','CVaR-L1-90']
}
group = {
    'Benchmark':['GMV-P', 'GMV-LS', 'GMV-NLS', 'QNE'],
    'CVaR-L1':['CVaR-L1-0.6','CVaR-L1-0.7','CVaR-L1-0.8','CVaR-L1-0.9']
}

df_alpha['Method'] = df_alpha['Method'].replace({old_group[i][j]:group[i][j] for i in old_group for j in range(len(old_group[i]))})


_df = df_alpha[df_alpha['Method'].isin([j for i in group for j in group[i]])]
for i in group:
    _df.loc[df_alpha['Method'].isin(group[i]), 'Group'] = i
    
sns.set_theme()
fig, axes = plt.subplots(1, 2, figsize=(12, 3), #gridspec_kw=dict(width_ratios=[4, 3])
                     )
axes[0] = sns.lineplot(data=_df, x="alpha", y="VaR", 
             hue="Method", style='Group', #dashes=[(3, 1), (1, 3), (2,0)],
                       markers=True, legend=False, 
                       ax=axes[0], )
axes[0].set_ylabel('VaR$_{\\alpha}$ (%)')
axes[0].set_xlabel('$\\alpha$')

axes[1] = sns.lineplot(data=_df, x="alpha", y="CVaR", 
             #shrink=.8, alpha=.8, legend=False, 
             hue="Method", style='Group', #dashes=[(3, 1), (1, 3), (2,0)],
             markers=True, legend=True, #size=['.','s','o'],
             ax=axes[1])
axes[1].set_ylabel('CVaR$_{\\alpha}$ (%)')
axes[1].set_xlabel('$\\alpha$')

fig.legend(loc=7)
axes[1].get_legend().remove()
fig.tight_layout()
fig.subplots_adjust(right=0.85)

plt.savefig('real.png', dpi=300, bbox_inches='tight', pad_inches=0)     