import os
from pathlib import Path

import torch

ROOT_DIR = Path(os.path.dirname(__file__)).parent
#ROOT_DIR = '/home/lukas/Desktop/nas/data_work/LukasChrist/flattery_audio'

DATA_DIR = os.path.join(ROOT_DIR, 'data')
FEATURE_DIR = os.path.join(DATA_DIR, 'features')
#SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
#CACHE_DIR = os.path.join(DATA_DIR, 'cache')
#SPEAKER_EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'speaker_embeddings')
#MFA_DIR = os.path.join(DATA_DIR, 'mfa_inputs')
#SHAP_DIR = os.path.join(DATA_DIR, 'shap')

#RAW_DIR = os.path.join(DATA_DIR, 'audio')

#ASR_DIR = os.path.join(DATA_DIR, 'asr_out')

CP_DIR = os.path.join(ROOT_DIR, 'checkpoints')
#EMBEDDINGS_DIR = "/home/lukas/Desktop/nas/data_work/LukasChrist/retracted/embeddings"
LOG_DIR = os.path.join(ROOT_DIR, 'logs')

#UTTERANCES = 'utterances'
#SENTENCES = 'sentences'
#LEVELS = [UTTERANCES, SENTENCES]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')