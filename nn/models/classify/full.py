#!/usr/bin/env python3
import torch.nn as nn
from ai_old.nn.models.encode.arcface import ArcFaceWrapper
from external.sg2.unit import FullyConnectedLayer
from external.sg2 import persistence


@persistence.persistent_class
class FullAttrClassifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # encoder
        self.e = ArcFaceWrapper(frozen=False)

        # classifiers
        self.gender = self._build_classifier()
        self.age = self._build_classifier()
        self.glasses = self._build_classifier()
        self.smile = self._build_classifier()
        self.mouth = self._build_classifier()
        self.hair_len = self._build_classifier()
        self.hair_curl = self._build_classifier()
        self.baldness = self._build_classifier()
        self.facial_hair = self._build_classifier()
        self.makeup = self._build_classifier()

    def _build_classifier(self):
        return nn.Sequential(
            FullyConnectedLayer(512, 256, activation='lrelu'),
            FullyConnectedLayer(256, 128, activation='lrelu'),
            FullyConnectedLayer(128, 64, activation='lrelu'),
            FullyConnectedLayer(64, 1, activation='linear'),
        )

    def forward(self, x):
        feat = self.e(x)
        return {
            'gender': self.gender(feat),
            'age': self.age(feat),
            'glasses': self.glasses(feat),
            'smile': self.smile(feat),
            'mouth': self.mouth(feat),
            'hair_len': self.hair_len(feat),
            'hair_curl': self.hair_curl(feat),
            'baldness': self.baldness(feat),
            'facial_hair': self.facial_hair(feat),
            'makeup': self.makeup(feat),
        }

    def prep_for_train_phase(self):
        self.e.requires_grad_(True)
        self.gender.requires_grad_(True)
        self.age.requires_grad_(True)
        self.glasses.requires_grad_(True)
        self.smile.requires_grad_(True)
        self.mouth.requires_grad_(True)
        self.hair_len.requires_grad_(True)
        self.hair_curl.requires_grad_(True)
        self.baldness.requires_grad_(True)
        self.facial_hair.requires_grad_(True)
        self.makeup.requires_grad_(True)
