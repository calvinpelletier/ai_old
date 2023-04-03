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
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person


BASE_DIR = os.path.join(c.ASI_DATASETS_PATH, 'e4e-facegen-9-1')
IMG_DIR = os.path.join(BASE_DIR, 'imgs')
LIGHTING_DIR = os.path.join(BASE_DIR, 'lighting')
RAW_DIR = os.path.join(BASE_DIR, 'raw-attributes')
PREPROCESSED_DIR = os.path.join(BASE_DIR, 'attributes-16')
FFHQ_LABELS_PATH = os.path.join(c.ASI_DATA_PATH, 'ffhq_aging_labels.csv')

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

OUTPUT_ATTRIBUTES = [
    'age',
    'gender',
    'smile',
    # 'facial_hair',
    'pitch',
    'yaw',
    'glasses',
    'sunglasses',
    # 'baldness',
    'lighting0',
    'lighting1',
    'lighting2',
    'lighting3',
    'lighting4',
    'lighting5',
    'lighting6',
    'lighting7',
    'lighting8',
]


def fetch():
    face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))

    raw_attrs = {}
    for fname in tqdm(os.listdir(IMG_DIR)):
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

        if not detected_faces or len(detected_faces) != 1:
            continue

        x = detected_faces[0].face_attributes
        raw_attr = {
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
        raw_attr['facial_hair'] = raw_attr['moustache'] + raw_attr['beard'] + \
            raw_attr['sideburns']

        id = fname.split('.')[0]
        raw_attrs[id] = raw_attr

        with open(os.path.join(RAW_DIR, f'{id}.pickle'), 'wb') as handle:
            pickle.dump(raw_attr, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return raw_attrs


def load():
    raw_attrs = {}
    for fname in os.listdir(RAW_DIR):
        id = fname.split('.')[0]
        with open(os.path.join(RAW_DIR, fname), 'rb') as f:
            data = pickle.load(f)
        raw_attrs[id] = data
    # print(raw_attrs)
    return raw_attrs


def add_lighting(raw_attrs):
    for id, raw_attr in raw_attrs.items():
        lighting = np.load(os.path.join(LIGHTING_DIR, f'{id}.npy'))
        for i, x in enumerate(lighting):
            raw_attr[f'lighting{i}'] = x
    return raw_attrs


def add_ffhq_labels(raw_attrs):
    n_total = 0
    n_bad_gender = 0
    n_bad_glasses = 0
    with open(FFHQ_LABELS_PATH, 'r') as f:
        next(f)
        for line in f:
            id, _, _, gender, _, _, _, _, _, _, glasses = line.strip().split(',')
            id = f'{int(id):05d}'

            assert gender in ['male', 'female']
            gender = 1. if gender == 'male' else 0.

            if id not in raw_attrs:
                print(f'missing id {id}, skipping')
                continue

            n_total += 1

            if gender != raw_attrs[id]['gender']:
                n_bad_gender += 1
                print(f'gender misclassification for {id}, fixing')
                raw_attrs[id]['gender'] = gender

            if glasses == '-1':
                # sometimes happens, just assume none
                glasses = 'None'
            assert glasses in ['None', 'Normal', 'Dark']
            if glasses == 'None' and raw_attrs[id]['glasses'] == 1. or \
                    glasses != 'None' and raw_attrs[id]['glasses'] == 0.:
                n_bad_glasses += 1
                print(f'glasses misclassification for {id}, fixing')
            raw_attrs[id]['sunglasses'] = 1. if glasses == 'Dark' else 0.
            raw_attrs[id]['glasses'] = 1. if glasses == 'Normal' else 0.

    print(f'n bad gender: {n_bad_gender} / {n_total}')
    print(f'n bad glasses: {n_bad_glasses} / {n_total}')
    return raw_attrs


def preprocess(raw_attrs, scale=False):
    if scale:
        mins = {}
        maxes = {}
        for attr in OUTPUT_ATTRIBUTES:
            for x in raw_attrs.values():
                if attr not in mins or mins[attr] > x[attr]:
                    mins[attr] = x[attr]
                if attr not in maxes or maxes[attr] < x[attr]:
                    maxes[attr] = x[attr]

    for id, x in raw_attrs.items():
        preprocessed = []
        for attr in OUTPUT_ATTRIBUTES:
            if scale:
                preprocessed.append(
                    (x[attr] - mins[attr]) / (maxes[attr] - mins[attr]),
                )
            else:
                preprocessed.append(x[attr])
        preprocessed = np.array(preprocessed)
        np.save(os.path.join(PREPROCESSED_DIR, f'{id}.npy'), preprocessed)


if __name__ == '__main__':
    if os.path.exists(os.path.join(RAW_DIR, '00000.pickle')):
        raw_attrs = load()
    else:
        assert False
        raw_attrs = fetch()
    raw_attrs = add_lighting(raw_attrs)
    raw_attrs = add_ffhq_labels(raw_attrs)
    preprocess(raw_attrs)
