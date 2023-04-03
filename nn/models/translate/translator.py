#!/usr/bin/env python3
from ai_old.lasi.nn.models import Unit
from ai_old.lasi.iit import ResIIT


# morphs a cropped and aligned face
# inputs: original image and controls
# output: translated image
class Translator(Unit):
    def __init__(self):
        super().__init__()

    def forward(self, img, ctrl):
        pass


# discards the controls input and sends the image through a resnet
class MinimalTranslator(Translator):
    def __init__(self):
        super().__init__()
        self.model = ResIIT()

    def forward(self, img, _ctrl):
        return self.model(img)
