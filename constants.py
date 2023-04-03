#!/usr/bin/env python3
import os
from os.path import join

# code
ASI_CODE_PATH = os.environ.get('ASI_CODE')
CONFIGS_FOLDER = os.path.join(ASI_CODE_PATH, 'configs')

# data
ASI_DATA_PATH = os.environ.get('ASI_DATA')
EXP_PATH = join(ASI_DATA_PATH, 'exp')
PRETRAINED_MODELS = join(ASI_DATA_PATH, 'models')
LERP_PATH = join(ASI_DATA_PATH, 'lerp')
FACE_LANDMARK_PREDICTOR = join(
    PRETRAINED_MODELS,
    'dlib/shape_predictor_68_face_landmarks.dat',
)

# prod
PROD_EXP = '28'
PROD_PATH = join(ASI_DATA_PATH, 'prod')
PROD_INPUT_FOLDER = join(PROD_PATH, 'x')
PROD_OUTPUT_FOLDER = join(PROD_PATH, 'y', PROD_EXP)

# datasets
ASI_DATASETS_PATH = os.environ.get('ASI_DATASETS')
# relative to ASI_DATASETS_PATH
SUPPLEMENTAL_DATASET_FOLDER_NAME = "supplemental"

# dev server
DEVSERVER_PATH = join(ASI_CODE_PATH, 'devserver/')
DEVSERVER_SECRET_KEY = join(DEVSERVER_PATH, 'secret.key')
USERS_FILE = join(DEVSERVER_PATH, 'data/auth/users.csv')
SERVER_LOG = join(DEVSERVER_PATH, 'logs/server.log')
LABELER_RESULTS = join(DEVSERVER_PATH, 'data/labeler/results.csv')
TOKEN_FILE = join(DEVSERVER_PATH, 'token.key')

# real server
SERVER_PATH = join(ASI_CODE_PATH, 'server/')
SERVER_CONFIG_DIR_PATH = join(SERVER_PATH, 'config/')
SERVER_SECRET_KEY = join(SERVER_PATH, 'secret.key')

# align
ALIGN_TRANSFORM_SIZE = 2048
ALIGNED_MASK_BUFFER = 2

# clip attrs
CLIP_ATTRS = [
    'female face',
    'male face',
    'face with short hair',
    'face with long hair',
    'face with blonde hair',
    'face with brown hair',
    'face with black hair',
    'face with red hair',
    'face with curly hair',
    'face with straight hair',
    'face with makeup',
    'face without makeup',
]
