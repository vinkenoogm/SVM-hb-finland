import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from itertools import product

data_path = Path('../../data')
scaler_path = Path('../results/scalers')
output_path = Path('../results/distributions')
output_path.mkdir(parents=True, exist_ok=True)
output_path2 = output_path / 'distributions_by_var'
output_path2.mkdir(parents=True, exist_ok=True)

def load_data(data_path, scaler_path, sex):
    dfs = []

    for nback, split in product(range(1, 6), ['train', 'test']):
        df = pd.read_pickle(data_path / f'{sex}_{nback}_{split}.pkl')
        scaler = pd.read_pickle(scaler_path / f'{sex}_{nback}.pkl')
        
        df_unscaled = df.copy()
        df_unscaled[df_unscaled.columns[:-1]] = scaler.inverse_transform(df_unscaled[df_unscaled.columns[:-1]])
        dfs.append(df_unscaled)
        
    df = pd.concat(dfs).reset_index()
    df = df.drop_duplicates(subset=['index'], keep='last')
    df = df.set_index('index')
    
    dfs2 = []
    for i in range(0, len(dfs), 2):
        dfs2.append(pd.concat([dfs[i], dfs[i+1]]))
    
    return dfs2, df

def marginal_distrs(data):
    cols = data.columns
    distrs = dict()
    distrs['errorcols'] = []
    for col in cols:
        if col in ['age', 'month', 'NumDon', 'FerritinPrev'] or col.startswith('HbPrev') or col.startswith('DaysSince') or col.startswith('prs'):
            distrs[col] = {'median': np.nanmedian(data[col]),
                           'q1': np.nanpercentile(data[col], 25),
                           'q3': np.nanpercentile(data[col], 75),
                           'min': np.min(data[col]),
                           'max': np.max(data[col]),
                           'mean': np.mean(data[col]),
                           'stdev': np.std(data[col])}
            if col == 'NumDon':
                distrs['NumDon2'] = dict(data[col].value_counts())
        elif col.startswith('snp'):
            distrs[col] = dict(data[col].value_counts())
        elif col == 'HbOK':
            distrs[col] = {'ndonations': sum(data[col] == 1),
                           'ndeferrals': sum(data[col] == 0),
                           'defrate': 1-np.mean(data[col])}
        else:
            distrs['errorcols'].append(col) 
    return distrs

def hb_distr_by_donor(data):
    grouped = data.groupby('vdonor')
    res = []
    for name, group in grouped:
        hbmean = np.mean(group['HbPrev1'])
        hbvar = np.var(group['HbPrev1'])
        nvisits = group.shape[0]
        sex = list(group['sex'])[0]
        snp_1 = list(group['snp_1_169549811'])[0]
        snp_6 = list(group['snp_6_32617727'])[0]
        snp_15 = list(group['snp_15_45095352'])[0]
        snp_17 = list(group['snp_17_58358769'])[0]
        res.append([sex, hbmean, hbvar, nvisits, snp_1, snp_6, snp_15, snp_17])
    df = pd.DataFrame(res, columns=['sex', 'hbmean', 'hbvar', 'nvisits', 'snp_1', 'snp_6', 'snp_15', 'snp_17'])
    return df
    

def main():
    df = pd.read_pickle(data_path / 'df_2016_2020.pkl')
    hb_distrs = hb_distr_by_donor(df)
    hb_distrs.to_pickle(output_path / f'hb_distrs_by_donor.pkl')
#     for sex in ['men', 'women']:        
#         distrs = marginal_distrs(df)
#         pickle.dump(distrs, open(output_path / f'distributions_{sex}_all.pkl', 'wb'))
        
#         for nback, subdf in enumerate(dfs):
#             subdistrs = marginal_distrs(subdf)
#             pickle.dump(subdistrs, open(output_path / f'distributions_{sex}_{nback+1}.pkl', 'wb'))
            
#         for snp1, snp6, snp15, snp17 in product('012', repeat=4):
#             df_sub = df.loc[(df.snp_1_169549811 == int(snp1)) & 
#                             (df.snp_6_32617727 == int(snp6)) & 
#                             (df.snp_15_45095352 == int(snp15)) & 
#                             (df.snp_17_58358769 == int(snp17)), ]
#             subdistrs = marginal_distrs(df_sub)
#             pickle.dump(subdistrs, open(output_path2 / f'distr_{sex}_snps_{snp1}{snp6}{snp15}{snp17}.pkl', 'wb'))

                
            
if __name__ == '__main__':
    main()
        
        