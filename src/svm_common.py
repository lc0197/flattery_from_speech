import pandas as pd
import os

from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

from config import DATA_DIR, FEATURE_DIR
from src.eval import eval_binary


def _load_sentence_data(features):
    db = pd.read_csv(os.path.join(DATA_DIR, 'final_db.csv'))[['sentence_id', 'speaker', 'flattery']]
    # each csv corresponds to one sentence
    feature_csv = os.path.join(FEATURE_DIR, f'{features}.csv')
    feature_df = pd.read_csv(feature_csv)
    feature_df.rename(columns={'ID':'sentence_id'}, inplace=True)
    db_len = len(db)
    #feature_df.fillna(0., inplace=True)
    full_df = db.merge(feature_df, on='sentence_id', how='outer')
    assert len(full_df) == db_len
    full_df.sort_values(by='sentence_id', inplace=True)
    return full_df


def _normalise_data(data_dct):
    train_x = data_dct['train']['X']
    normaliser = MinMaxScaler().fit(train_x)
    for partition in ['train', 'dev', 'test']:
        data_dct[partition]['X'] = normaliser.transform(data_dct[partition]['X'])
    return data_dct


def create_class_weights(cw_list):
    # bring args.class_weight into right format
    if cw_list is None:
        return [None]
    ls = []
    for w in cw_list:
        if w == 'balanced':
            ls.append('balanced')
        else:
            ls.append({0: 1., 1:int(w)})
    return ls


def _create_grids(args):
    grids = []
    num_configs = 0
    # linear
    if 'linear' in args.kernels:
        grids.append({
            'kernel': ['linear'],
            'C': args.Cs,
            'class_weight': args.class_weights
        })
        num_configs += len(args.Cs) * len(args.class_weights)
    if 'poly' in args.kernels:
        grids.append({
            'kernel': ['poly'],
            'C': args.Cs,
            'degree': args.degrees,
            'class_weight': args.class_weights,
            'gamma': args.gammas
        })
        num_configs += len(args.Cs) * len(args.degrees) * len(args.class_weights) * len(args.gammas)
    if 'rbf' in args.kernels:
        grids.append({
            'kernel': ['rbf'],
            'C': args.Cs,
            'class_weight': args.class_weights,
            'gamma': args.gammas
        })
        num_configs += len(args.Cs) * len(args.class_weights) * len(args.gammas)
    if 'sigmoid' in args.kernels:
        grids.append({
            'kernel': ['sigmoid'],
            'C': args.Cs,
            'class_weight': args.class_weights,
            'gamma': args.gammas
        })
        num_configs += len(args.Cs) * len(args.class_weights) * len(args.gammas)
    param_grid = ParameterGrid(grids)
    return param_grid


def _split_data(data_df):
    split_df = pd.read_csv(os.path.join(DATA_DIR, f'split.csv'))
    dct = {}
    for partition in ['train', 'dev', 'test']:
        ids = split_df[split_df.partition==partition].speaker.values
        p_df = data_df[data_df.speaker.isin(ids)]
        # meta: sentence_id, speaker, flattery
        x = p_df.iloc[:, 3:].values
        if np.any(np.isnan(x)):
            print(f'Warning, found {np.sum(np.isnan(x))} NaNs in X of {partition} (of {len(x)} values)')
        dct[partition] = {
            # 5: start, end, sentence_id, speaker, flattery
            'X': p_df.iloc[:,3:].values,
            'y': p_df.flattery.values,
            'meta': p_df[['sentence_id', 'speaker']]
        }
    return dct


def _optimise_svm(data_dct, grid, args):
    best_config = None
    best_metric = -100
    best_model = None

    for config in tqdm(list(grid)):
        if args.use_linear_svc:
            svc = LinearSVC(C=config['C'], class_weight=config['class_weight'], dual=False, random_state=101)
        else:
            svc = SVC(**config)
        svc.fit(data_dct['train']['X'], data_dct['train']['y'])
        dev_preds = svc.predict(data_dct['dev']['X'])
        metrics = eval_binary(data_dct['dev']['y'], dev_preds)
        relevant_metric = metrics[args.metric]
        if relevant_metric > best_metric:
            best_metric = relevant_metric
            best_config = config
            best_model = svc

    return best_config, best_metric, best_model


