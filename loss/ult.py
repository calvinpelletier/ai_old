#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.loss import ComboLoss
from ai_old.loss.gan import get_gan_loss_for_g
from ai_old.loss.perceptual.trad import PerceptualLoss
from ai_old.loss.perceptual.face import FaceLoss
from ai_old.loss.unet import GEncLoss, GDecLoss


def has_at_least_one_attr(conf, attrs):
    for x in attrs:
        if hasattr(conf, x):
            return True
    return False


class UltLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)

        if has_at_least_one_attr(conf, [
            'real_rec_perceptual',
            'male_rec_perceptual',
            'female_rec_perceptual',
        ]):
            percep = PerceptualLoss()
        if has_at_least_one_attr(conf, [
            'real_rec_perceptual',
            'male_rec_perceptual',
            'female_rec_perceptual',
        ]):
            face = FaceLoss()


        # real
        if hasattr(conf, 'real_rec_pixel'):
            self.create_subloss(
                'real_rec_pixel',
                nn.MSELoss(),
                ('real_rec', 'real'),
            )
        if hasattr(conf, 'real_rec_perceptual'):
            self.create_subloss(
                'real_rec_perceptual',
                percep,
                ('real_rec', 'real'),
            )
        if hasattr(conf, 'real_rec_face'):
            self.create_subloss(
                'real_rec_face',
                face,
                ('real_rec', 'real'),
            )
        if hasattr(conf, 'gender'):
            self.create_subloss(
                'gender',
                nn.BCEWithLogitsLoss(),
                ('real_gender_pred', 'real_gender')
            )

        # fake
        if hasattr(conf, 'gan_enc'):
            self.create_subloss(
                'gan_enc',
                GEncLoss(),
                ('d_fake_enc',),
            )
        if hasattr(conf, 'gan_dec'):
            self.create_subloss(
                'gan_dec',
                GDecLoss(),
                ('d_fake_dec',),
            )

        # ss
        if hasattr(conf, 'dz_mtf'):
            self.create_subloss(
                'dz_mtf',
                nn.MSELoss(),
                ('female_z_pred', 'female_z'),
            )
        if hasattr(conf, 'dz_ftm'):
            self.create_subloss(
                'dz_ftm',
                nn.MSELoss(),
                ('male_z_pred', 'male_z'),
            )
        if hasattr(conf, 'identity'):
            self.create_subloss(
                'identity',
                _identity_loss,
                ('male_identity', 'female_identity'),
            )
        if hasattr(conf, 'male_rec_pixel'):
            self.create_subloss(
                'male_rec_pixel',
                nn.MSELoss(),
                ('male_rec', 'male'),
            )
        if hasattr(conf, 'male_rec_perceptual'):
            self.create_subloss(
                'male_rec_perceptual',
                percep,
                ('male_rec', 'male'),
            )
        if hasattr(conf, 'male_rec_face'):
            self.create_subloss(
                'male_rec_face',
                face,
                ('male_rec', 'male'),
            )
        if hasattr(conf, 'female_rec_pixel'):
            self.create_subloss(
                'female_rec_pixel',
                nn.MSELoss(),
                ('female_rec', 'female'),
            )
        if hasattr(conf, 'female_rec_perceptual'):
            self.create_subloss(
                'female_rec_perceptual',
                percep,
                ('female_rec', 'female'),
            )
        if hasattr(conf, 'female_rec_face'):
            self.create_subloss(
                'female_rec_face',
                face,
                ('female_rec', 'female'),
            )


def _identity_loss(x, y):
    return torch.mean(torch.abs(x - y))
