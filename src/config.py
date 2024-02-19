import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(__file__)).parent

DATA_DIR = os.path.join(ROOT_DIR, 'data')
FEATURE_DIR = os.path.join(DATA_DIR, 'features')
PRED_DIR = os.path.join(ROOT_DIR, 'predictions')

CP_DIR = os.path.join(ROOT_DIR, 'checkpoints')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')