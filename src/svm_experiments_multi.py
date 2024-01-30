import json
import os
import pickle
from argparse import ArgumentParser
from time import ctime, time

import numpy as np
import pandas as pd
from glob import glob

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm

from config import DATA_DIR, FEATURE_DIR, LOG_DIR


# methods
from eval import eval_binary
from svm_common import _load_sentence_data, _normalise_data, create_class_weights, _create_grids, _split_data, \
    _optimise_svm

FEATURE_LEVEL = 'feature_level'
LATE_FUSION = 'late_fusion'
OTHER = 'other'
# TODO is this really used? - remove late fusion here...
METHODS = [FEATURE_LEVEL, LATE_FUSION, OTHER]


def parse_args():
    parser = ArgumentParser()
    # data
    # sentence level only
    #parser.add_argument('--level', required=True, choices=LEVELS)
    parser.add_argument('--feature_a', type=str, required=True)
    parser.add_argument('--feature_t', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    # hparams
    parser.add_argument('--Cs', nargs='+', type=float, default=[1.])
    parser.add_argument('--kernels', nargs='+', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid'], default=['linear'])
    parser.add_argument('--degrees', nargs='+', type=int, default=[3])
    parser.add_argument('--class_weights', nargs='+', required=False, type=str, help='Either an integer or "balanced" or omit - then None')
    parser.add_argument('--gammas', nargs='+', type=str, default=['scale'])
    # experiment config
    # TODO make clear usage of seeds here
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--metric', type=str, default='uar', choices=['f1', 'uar'])
    parser.add_argument('--run_name', type=str, required=False)
    parser.add_argument('--experiment_family', type=str, required=False)
    parser.add_argument('--use_linear_svc', action='store_true', help='Only applicable for linear-only --kernel')
    parser.add_argument('--method', choices=METHODS, required=True)


    args = parser.parse_args()
    args.run_name = "" if args.run_name is None else args.run_name
    args.run_name = args.run_name + "_" + ctime(time()).replace(":","-").replace(" ","_")

    if args.use_linear_svc:
        assert args.kernels == ['linear'], "--kernel must be 'linear', if --use_linear_svc is set"

    args.class_weights = create_class_weights(args.class_weights)

    return args


def _load_data(feature_a, feature_t, args):
    data_df_a = _load_sentence_data(feature_a)
    data_df_t = _load_sentence_data(feature_t)
    data_dct_a = _split_data(data_df_a)
    data_dct_t = _split_data(data_df_t)

    if args.normalize:
        data_dct_a = _normalise_data(data_dct_a)
        data_dct_t = _normalise_data(data_dct_t)

    for partition in ['train', 'dev', 'test']:
        data_dct_a[partition]['X'] = np.nan_to_num(data_dct_a[partition]['X'], copy=False)
        data_dct_t[partition]['X'] = np.nan_to_num(data_dct_t[partition]['X'], copy=False)

    return data_dct_a, data_dct_t


def _init_log_dir(args):
    log_dir = os.path.join(LOG_DIR, 'svm_multi', args.split, args.method, f'{args.feature_a}_{args.feature_t}', args.run_name)
    os.makedirs(log_dir)
    pred_dir = os.path.join(log_dir, 'predictions')
    os.makedirs(pred_dir)
    with open(os.path.join(log_dir, 'config.json'), 'w+') as f:
        json.dump(vars(args), f)
    return log_dir, pred_dir


def _merge_representations(data_a, data_t):
    data_dct_merged = {}
    for partition in data_a.keys():
        data_dct_merged[partition] = {}
        assert (data_a[partition]['y'] == data_t[partition]['y']).all()
        data_dct_merged[partition]['y'] = data_a[partition]['y']
        data_dct_merged[partition]['X'] = np.hstack([data_a[partition]['X'], data_t[partition]['X']])
    return data_dct_merged


# TODO rename
def train_feature_level(data_a, data_t, grid, args):
    data_dct = _merge_representations(data_a, data_t)
    best_config, best_metric, best_model = _optimise_svm(data_dct, grid, args)
    # metrics for predictions
    dev_predictions = best_model.predict(data_dct['dev']['X'])
    test_predictions = best_model.predict(data_dct['test']['X'])
    dev_metrics = eval_binary(data_dct['dev']['y'], dev_predictions)
    assert dev_metrics[args.metric] == best_metric
    test_metrics = eval_binary(data_dct['test']['y'], test_predictions)
    dev_df = pd.DataFrame({'pred':dev_predictions, 'GS': data_dct['dev']['y']})
    test_df = pd.DataFrame({'pred': test_predictions, 'GS': data_dct['test']['y']})
    return best_config, dev_metrics, test_metrics, dev_df, test_df


if __name__ == '__main__':
    args = parse_args()

    log_dir, pred_dir = _init_log_dir(args)

    grid = _create_grids(args)
    print(f'Training {len(list(grid))} configurations')

    log_dct = {'params':vars(args)}

    for seed in range(args.seed, args.n_seeds + args.seed):
        np.random.seed(seed)
        feature_a = f'{args.feature_a}_{seed}'
        feature_t = f'{args.feature_t}_{seed}'
        data_dct_a, data_dct_t = _load_data(feature_a, feature_t, args)

        print(f'Training for seed {seed}')


        best_config, dev_metrics, test_metrics, dev_df, test_df = train_feature_level(
                data_dct_a, data_dct_t, grid, args)

        seed_dct = {'config': best_config}
        for k,v in dev_metrics.items():
            seed_dct.update({f'dev_{k}': v})
        for k,v in test_metrics.items():
            seed_dct.update({f'test_{k}': v})

        log_dct[seed] = seed_dct

        with open(os.path.join(log_dir, 'log.json'), 'w+') as f:
            json.dump(log_dct, f)

        dev_df.to_csv(os.path.join(pred_dir, f'dev_{seed}.csv'), index=False)
        test_df.to_csv(os.path.join(pred_dir, f'test_{seed}.csv'), index=False)

    # summarise results
    summ_dct = {}
    for k in log_dct[args.seed].keys():
        if k == 'config':
            continue
        values = [log_dct[i][k] for i in log_dct.keys() if type(i)==int]
        for m, f in zip(['mean', 'std'], [np.mean, np.std]):
            summ_dct[f'{k}_{m}'] = f(values)
    # save summary as json
    with open(os.path.join(log_dir, 'summary.json'), 'w+') as f:
        json.dump(summ_dct, f)

    # add to experiment family
    if not (args.experiment_family is None):
        row_dct = vars(args)
        row_dct.update(summ_dct)
        row_df = pd.DataFrame({k:[v] for k,v in row_dct.items()})
        family_csv = os.path.join(os.path.dirname(os.path.dirname(log_dir)), f'{args.experiment_family}.csv')
        if os.path.exists(family_csv):
            df = pd.read_csv(family_csv)
            df = pd.concat([df, row_df])
        else:
            df = row_df
        df.to_csv(family_csv, index=False)




