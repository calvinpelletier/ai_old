#!/usr/bin/env python3
import torch.nn as nn
from ai_old.loss import ComboLoss
from ai_old.loss.gan import get_gan_loss_for_d
from ai_old.loss.reg import gradient_penalty


class RegularizedDLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)
        self.create_subloss(
            'gan',
            get_gan_loss_for_d(conf.gan.type),
            ('d_fake', 'd_real'),
        )

        if hasattr(conf, 'gp'):
            self.create_subloss(
                'gp',
                gradient_penalty,
                ('d_real', 'y'),
                requires_grad_for_y=True,
            )


class UltDLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)
        self.create_subloss(
            'gan',
            get_gan_loss_for_d(conf.gan.type),
            ('d_fake', 'd_real'),
        )

        if hasattr(conf, 'gp'):
            self.create_subloss(
                'gp',
                gradient_penalty,
                ('d_real', 'real'),
                requires_grad_for_y=True,
            )


class AutoencoderRegularizedDLoss(ComboLoss):
    def __init__(self, conf):
        super().__init__(conf)
        self.create_subloss(
            'gan',
            get_gan_loss_for_d(conf.gan.type),
            ('d_fake', 'd_real'),
        )
        self.create_subloss(
            'rec',
            nn.MSELoss(),
            ('d_real_rec', 'y'),
        )

        if hasattr(conf, 'gp'):
            self.create_subloss(
                'gp',
                gradient_penalty,
                ('d_real', 'y'),
                requires_grad_for_y=True,
            )
