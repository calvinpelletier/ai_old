#!/usr/bin/env python3
import torch.nn as nn
from ai_old.loss import ComboLoss
from ai_old.loss.patch import PatchFeatMatchLoss
from ai_old.loss.gan import get_gan_loss_for_g
from ai_old.loss.perceptual.trad import PerceptualLoss
from ai_old.loss.perceptual.lpips import LpipsLoss
from ai_old.loss.perceptual.face import FaceLoss


class GuidedGLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)

        # gan loss
        self.create_subloss(
            'gan',
            get_gan_loss_for_g(conf.gan.type),
            ('d_fake',),
        )

        # l2 pixel loss
        if hasattr(conf, 'pixel'):
            self.create_subloss(
                'pixel',
                nn.MSELoss(),
                ('g_fake', 'y'),
            )

        # perceptual loss (either traditional perceptual or lpips)
        if hasattr(conf, 'perceptual'):
            assert not hasattr(conf, 'lpips'), 'probably a mistake'
            self.create_subloss(
                'perceptual',
                PerceptualLoss(),
                ('g_fake', 'y'),
            )
        if hasattr(conf, 'lpips'):
            assert not hasattr(conf, 'perceptual'), 'probably a mistake'
            self.create_subloss(
                'lpips',
                LpipsLoss(),
                ('g_fake', 'y'),
            )

        # face loss
        if hasattr(conf, 'face'):
            self.create_subloss(
                'face',
                FaceLoss(),
                ('g_fake', 'y'),
            )

        # feature matching loss (requires patch discriminator)
        if hasattr(conf, 'feat_match'):
            assert conf.gan.type == 'patch_hinge'
            self.create_subloss(
                'feat_match',
                PatchFeatMatchLoss(),
                ('d_fake_feat', 'd_real_feat'),
            )
