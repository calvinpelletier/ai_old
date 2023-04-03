#!/usr/bin/env python3
import os
import ai_old.constants as c
import numpy as np
import pickle
from tqdm import tqdm

import asyncio
import io
import glob
import sys
import time
import uuid
import requests
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials


IMG_DIR = os.path.join(
    c.ASI_DATASETS_PATH,
    c.SUPPLEMENTAL_DATASET_FOLDER_NAME,
    'face_image_256',
    'ffhq-128',
)
ATTR_DIR = os.path.join(c.ASI_DATASETS_PATH, 'ffhq-128', 'attributes')
TMP_DIR = os.path.join(c.ASI_DATASETS_PATH, 'ffhq-128', 'tmp')

with open(os.path.join(c.ASI_DATA_PATH, 'auth/microsoft_face.key'), 'r') as f:
    KEY = f.read().rstrip('\n')

ENDPOINT = 'https://ahanu-azure0.cognitiveservices.azure.com/'

API_ATTRIBUTES = [
    'age',
    'gender',
    'headPose',
    'smile',
    'facialHair',
    'glasses',
    # 'emotion',
    'hair',
    # 'makeup',
    # 'occlusion',
    # 'accessories',
    # 'blur',
    # 'exposure',
    # 'noise',
]


def _parse_attr(x):
    attr = {
        'age': x.age,
        'gender': 1. if x.gender == 'male' else 0.,
        'smile': x.smile,
        'moustache': x.facial_hair.moustache,
        'beard': x.facial_hair.beard,
        'sideburns': x.facial_hair.sideburns,
        'glasses': 0. if str(x.glasses) == 'GlassesType.no_glasses' else 1.,
        'roll': x.head_pose.roll,
        'pitch': x.head_pose.pitch,
        'yaw': x.head_pose.yaw,
        'baldness': x.hair.bald,
    }
    attr['facial_hair'] = attr['moustache'] + attr['beard'] + \
        attr['sideburns']


def fetch():
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    for fname in tqdm(os.listdir(IMG_DIR)):
        id = fname.split('.')[0]

        detected_faces = face_client.face.detect_with_stream(
            open(os.path.join(IMG_DIR, fname), 'rb'),
            return_face_id=True,
            return_face_landmarks=False,
            return_face_attributes=API_ATTRIBUTES,
            recognition_model='recognition_04',
            return_recognition_model=False,
            detection_model='detection_01',
            face_id_time_to_live=86400,
            custom_headers=None,
            raw=False,
            callback=None,
        )

        if not detected_faces:
            continue
        assert len(detected_faces) > 0

        if len(detected_faces) == 1:
            x = detected_faces[0].face_attributes
            attr = _parse_attr(x)
            with open(os.path.join(ATTR_DIR, f'{id}.pickle'), 'wb') as f:
                pickle.dump(attr, f)
        else:
            dir = os.path.join(TMP_DIR, id)
            os.makedirs(dir)
            for i in range(len(detected_faces)):
                x = detected_faces[i].face_attributes
                attr = _parse_attr(x)
                with open(os.path.join(dir, f'{i}.pickle'), 'wb') as f:
                    pickle.dump(attr, f)


if __name__ == '__main__':
    fetch()
