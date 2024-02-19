import json
from argparse import ArgumentParser
import os
from glob import glob
from time import ctime, time

import pandas as pd
import numpy as np

from config import LOG_DIR, DATA_DIR, PRED_DIR
from eval import eval_binary

UNW = 'unweighted'
WEIGHTED = 'weighted'

METHODS = [UNW, WEIGHTED]

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--audio', required=True, help='Path to folder with audio predictions. This is a directory '
                                                       'directly below "predictions/audio" that contains one folder '
                                                       'per seed for the predicting model (e.g. "101", "102",...).')
    parser.add_argument('--text', required=True, help='Analogously to --audio, a directory under "predictions/textual". '
                                                      'Expected to contain the exact same seed directories as given for audio.')
    parser.add_argument('--run_name', required=True, help='A unique name for logging')
    parser.add_argument('--method', required=True, choices=METHODS, help='Choose either weighted or unweighted late fusion. '
                                                                         'Weights are calculated based on UAR on development set.')
    args = parser.parse_args()
    args.run_name = args.run_name + ctime(time()).replace(":","_").replace(" ", "_")
    return args


def _init_log_dir(args):
    log_dir = os.path.join(LOG_DIR, 'late_fusion', args.method, f'{args.audio}_{args.text}', args.run_name)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'config.json'), 'w+') as f:
        json.dump(vars(args), f)
    return log_dir

# load the gold standard
def _load_gs():
    data_df = pd.read_csv(os.path.join(DATA_DIR, 'final_db.csv'))[['sentence_id', 'speaker', 'flattery']]
    split_df = pd.read_csv(os.path.join(DATA_DIR, 'split.csv'))
    gss = {}
    for partition in ['dev', 'test']:
        ids = split_df[split_df.partition == partition].speaker.values
        p_df = data_df[data_df.speaker.isin(ids)]
        gss[partition] = p_df.flattery.values
    return gss


def unweighted_lf(text_preds, audio_preds):
    scores = (text_preds + audio_preds) / 2
    return np.where(scores >= .5, 1, 0)


# weights based on difference between UAR and chance UAR (0.5)
def get_weights(text_preds, audio_preds, gs):
    text_uar = eval_binary(gs, np.where(text_preds >= 0.5, 1, 0))['uar']
    audio_uar = eval_binary(gs, np.where(audio_preds >= 0.5, 1, 0))['uar']
    text_weight = (text_uar - 0.5) / (text_uar + audio_uar - 1)
    audio_weight = 1 - text_weight
    return text_weight, audio_weight


def weighted_lf(text_preds, audio_preds, text_weight, audio_weight):
    text_preds = text_weight * text_preds
    audio_preds = audio_weight * audio_preds
    scores = text_preds + audio_preds
    return np.where(scores >= .5, 1, 0)


if __name__ == '__main__':
    args = parse_args()
    text_dir = os.path.join(PRED_DIR, 'textual', args.text)
    audio_dir = os.path.join(PRED_DIR, 'audio', args.audio)
    text_seeds = sorted([os.path.basename(d) for d in [x for x in glob(os.path.join(text_dir, '*')) if os.path.isdir(x)]])
    audio_seeds = sorted(
        [os.path.basename(d) for d in [x for x in glob(os.path.join(audio_dir, '*')) if os.path.isdir(x)]])
    assert audio_seeds == text_seeds
    log_dir = _init_log_dir(args)
    log_dct = {'params': vars(args)}

    gss = _load_gs(args)


    for seed in audio_seeds:

        seed_dct = {}

        for part in ['dev', 'test']:
            text_preds = pd.read_csv(os.path.join(text_dir, seed, f'predictions_{part}.csv'), header=None).values
            audio_preds = pd.read_csv(os.path.join(audio_dir, seed, f'predictions_{part}.csv'), header=None).values
            if args.method == UNW:
                lf_preds = unweighted_lf(text_preds, audio_preds)
            elif args.method == WEIGHTED:
                if part == 'dev':
                    text_w, audio_w = get_weights(text_preds, audio_preds, gss['dev'])
                lf_preds = weighted_lf(text_preds, audio_preds, text_weight=text_w, audio_weight=audio_w)
            else:
                print(f'Method {args.method} not supported')
                exit(-1)

            seed_dct.update({f'{part}_{k}':v for k,v in eval_binary(gss[part], lf_preds).items()})
        log_dct[int(seed)] = seed_dct

    # summarise results
    summ_dct = {}
    for k in log_dct[int(audio_seeds[0])].keys():
        values = [log_dct[i][k] for i in log_dct.keys() if type(i) == int]
        for m, f in zip(['mean', 'std'], [np.mean, np.std]):
            summ_dct[f'{k}_{m}'] = f(values)
    # save summary as json
    with open(os.path.join(log_dir, 'summary.json'), 'w+') as f:
        json.dump(summ_dct, f)
