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
from eval import eval_binary
from svm_common import _load_sentence_data, _normalise_data, create_class_weights, _create_grids, _split_data, \
    _optimise_svm


def parse_args():
    parser = ArgumentParser()
    # data
    # sentence level only
    #parser.add_argument('--level', required=True, choices=LEVELS)
    parser.add_argument('--feature', type=str, required=True)
    parser.add_argument('--normalize', action='store_true')
    # hparams
    parser.add_argument('--Cs', nargs='+', type=float, default=[1.])
    parser.add_argument('--kernels', nargs='+', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid'], default=['linear'])
    parser.add_argument('--degrees', nargs='+', type=int, default=[3])
    parser.add_argument('--class_weights', nargs='+', required=False, type=str, help='Either an integer or "balanced" or omit - then None')
    parser.add_argument('--gammas', nargs='+', type=str, default=['scale'])
    # experiment config
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--metric', type=str, default='uar', choices=['f1', 'uar'])
    parser.add_argument('--run_name', type=str, required=False)
    parser.add_argument('--experiment_family', type=str, required=False)
    parser.add_argument('--use_linear_svc', action='store_true', help='Only applicable for linear-only --kernel')


    args = parser.parse_args()
    args.run_name = "" if args.run_name is None else args.run_name
    args.run_name = args.run_name + "_" + ctime(time()).replace(":","-").replace(" ","_")

    if args.use_linear_svc:
        assert args.kernels == ['linear'], "--kernel must be 'linear', if --use_linear_svc is set"

    args.class_weights = create_class_weights(args.class_weights)

    return args


def _load_data(args):
    data_df = _load_sentence_data(args.feature)
    data_dct = _split_data(data_df)
    if args.normalize:
        data_dct = _normalise_data(data_dct)
    for partition in ['train', 'dev', 'test']:
        data_dct[partition]['X'] = np.nan_to_num(data_dct[partition]['X'], copy=False)
    return data_dct


def _init_log_dir(args):
    log_dir = os.path.join(LOG_DIR, 'svm', args.split, args.feature, args.run_name)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'config.json'), 'w+') as f:
        json.dump(vars(args), f)
    return log_dir


if __name__ == '__main__':
    args = parse_args()

    log_dir = _init_log_dir(args)

    data_dct = _load_data(args)
    grid = _create_grids(args)
    print(f'Training {len(list(grid))} configurations')

    log_dict = {'params':vars(args)}

    best_config, best_metric, best_model = _optimise_svm(data_dct, grid, args)
    # best_config = None
    # best_metric = -100
    # best_model = None
    #
    # for config in tqdm(list(grid)):
    #     if args.use_linear_svc:
    #         svc = LinearSVC(C=config['C'], class_weight=config['class_weight'], dual=False, random_state=seed)
    #     else:
    #         svc = SVC(**config)
    #     svc.fit(data_dct['train']['X'], data_dct['train']['y'])
    #     dev_preds = svc.predict(data_dct['dev']['X'])
    #     metrics = eval_binary(data_dct['dev']['y'], dev_preds)
    #     relevant_metric = metrics[args.metric]
    #     if relevant_metric > best_metric:
    #         best_metric = relevant_metric
    #         best_config = config
    #         best_model = svc


    dev_predictions = best_model.predict(data_dct['dev']['X'])
    test_predictions = best_model.predict(data_dct['test']['X'])
    dev_metrics = eval_binary(data_dct['dev']['y'], dev_predictions)
    assert dev_metrics[args.metric] == best_metric
    test_metrics = eval_binary(data_dct['test']['y'], test_predictions)

    log_dict['best_config'] = best_config
    res_dct = {}
    for k,v in dev_metrics.items():
        res_dct.update({f'dev_{k}': v})
    for k,v in test_metrics.items():
        res_dct.update({f'test_{k}': v})

    log_dict['results'] = res_dct

    with open(os.path.join(log_dir, 'log.json'), 'w+') as f:
        json.dump(log_dict, f)


    # add to experiment family
    if not (args.experiment_family is None):
        row_dct = vars(args)
        row_dct.update(log_dict['results'])
        row_df = pd.DataFrame({k:[v] for k,v in row_dct.items()})
        family_csv = os.path.join(os.path.dirname(os.path.dirname(log_dir)), f'{args.experiment_family}.csv')
        if os.path.exists(family_csv):
            df = pd.read_csv(family_csv)
            df = pd.concat([df, row_df])
        else:
            df = row_df
        df.to_csv(family_csv, index=False)