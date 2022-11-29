import argparse
import datetime
from pathlib import Path
import pickle
import warnings

import pandas as pd
import shap

warnings.filterwarnings('ignore')
data_path = Path('../../data')
results_path = Path('../results')

# Paths for testing
# data_path = Path('../data')
# results_path = Path('../testresults')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('nback', type=int,
                        help='[int] number of previous Hb values to use in prediction')
    parser.add_argument('sex', type=str, choices=['men', 'women'],
                        help='[men/women] sex to use in model')
    parser.add_argument('--n', type=int, default=100,
                        help='[int] number of donors to calculate SHAP values on')
    parser.add_argument('--foldersuffix', type=str, default='',
                        help='[str] optional suffix indicating non-default run')
    args = parser.parse_args()
    return args

def calc_shap(args):
    filename = results_path / f'models{args.foldersuffix}/clf_{args.sex}_{args.nback}.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    test = pd.read_pickle(data_path / f'scaled/{args.sex}_{args.nback}_test.pkl')
    X_test = test[test.columns[:-1]]
    
    scaler = pd.read_pickle(results_path / f'scalers/{args.sex}_{args.nback}.pkl')
    sc_index = list(scaler.feature_names_in_).index('snp_17_58358769')
    sc_mean = scaler.mean_[sc_index]
    sc_scale = scaler.scale_[sc_index]
    
    sc_lim = (0.6 - sc_mean) / sc_scale
    X_shap_snp1 = shap.sample(X_test.loc[X_test['snp_17_58358769'] > sc_lim, ], args.n)
    X_shap_snp0 = shap.sample(X_test.loc[X_test['snp_17_58358769'] < sc_lim, ], args.n)
    
    if args.foldersuffix == '_hbonly':
        X_shap_snp1 = X_shap_snp1.drop(columns=['snp_17_58358769', 'snp_6_32617727', 'snp_15_45095352', 
                                                'snp_1_169549811', 'prs_anemia', 'prs_ferritin', 'prs_hemoglobin'])
        X_shap_snp0 = X_shap_snp0.drop(columns=['snp_17_58358769', 'snp_6_32617727', 'snp_15_45095352', 
                                                'snp_1_169549811', 'prs_anemia', 'prs_ferritin', 'prs_hemoglobin'])
    
    for name, X_shap in zip(['snp1', 'snp0'], [X_shap_snp1, X_shap_snp0]):
        print(name, X_shap.shape)
        explainer = shap.KernelExplainer(clf.predict, X_shap)
        shapvals = explainer.shap_values(X_shap)

        output_path = results_path / f'shap_subset{args.foldersuffix}/'
        output_path.mkdir(parents=True, exist_ok=True)
        filename1 = f'Xshap_{name}_{args.sex}_{args.nback}.pkl'
        filename2 = f'shapvals_{name}_{args.sex}_{args.nback}.pkl'

        pickle.dump(X_shap, open(output_path / filename1, 'wb'))
        pickle.dump(shapvals, open(output_path / filename2, 'wb'))
        
def anon_shap(args):
    input_path = results_path / f'shap{args.foldersuffix}/'
    
    for name in ['snp1', 'snp0']:
        filename1 = f'Xshap_{name}_{args.sex}_{args.nback}_{args.n}.pkl'
        filename2 = f'shapvals_{name}_{args.sex}_{args.nback}_{args.n}.pkl'

        Xshap = pd.read_pickle(input_path / filename1)
        shapvals = pd.read_pickle(input_path / filename2)

        shapdf = pd.DataFrame({'variable': list(Xshap.columns) * Xshap.shape[0],
                               'value': Xshap.values.flatten(),
                               'shap': shapvals.flatten()}).sample(frac=1).sort_values('variable').reset_index(drop=True)

        output_path = results_path / f'anonshap_subset{args.foldersuffix}'
        output_path.mkdir(parents=True, exist_ok=True)

        shapdf.to_pickle(output_path / f'shapdf_{name}_{args.sex}_{args.nback}.pkl')

def main(args):
    calc_shap(args)
    print(f'    SHAP values for SNP subset from SVM-{args.nback}, {args.sex}, {args.foldersuffix} calculated and saved')
    anon_shap(args)
    print(f'    SHAP values for SNP subset from SVM-{args.nback}, {args.sex}, {args.foldersuffix} anonymized and saved')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)