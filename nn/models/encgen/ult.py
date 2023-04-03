#!/usr/bin/env python3
import torch
import torch.nn as nn
from ai_old.nn.models import Unit
from ai_old.nn.models.encode.simple import SimpleZEncoder
from ai_old.nn.models.encode.adalin import AdalinModulatedEncoder
from ai_old.nn.models.facegen.adalin import AdalinModulatedGenerator
from ai_old.nn.models.facegen.excitation import ExcitationModulatedGenerator, \
    BlendExcitationModulatedGenerator
from ai_old.nn.models.facegen.convertible_style import BlendConvertibleStyleGenerator
from ai_old.nn.models.encode.excitation import ExcitationModulatedEncoder
from ai_old.nn.models.transform.disentangle import Disentangler
from ai_old.nn.models.classify.gender import MinimalZGenderClassifier
from ai_old.nn.models.transform.gender import GenderLatentTransform


class BlendUltModel(Unit):
    def __init__(self,
        # configuration
        real_enabled=True,
        gan_enabled=True,
        swap_enabled=True,
        swap_rec_enabled=False,

        # general
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
        nc_max=512,
        norm='batch',
        weight_norm=False,
        actv='mish',

        # e pri
        e_pri='simple',

        # e mod
        e_mod='adalin',

        # c gender
        c_gender='affine',

        # f
        f='mlp',
        f_n_layers=6,
        f_lr_mul=0.1,

        # g
        g='excitation',
        g_k_blend=1,

        # t
        t='affine',
    ):
        super().__init__()
        self.real_enabled = real_enabled
        self.gan_enabled = gan_enabled
        self.swap_enabled = swap_enabled
        self.swap_rec_enabled = swap_rec_enabled

        assert e_pri == 'simple'
        self.e_pri = SimpleZEncoder(
            input_imsize=imsize,
            smallest_imsize=4,
            z_dims=z_dims,
            k=3,
            k_init=3,
            nc_in=3,
            nc_base=nc_base,
            im2vec='mlp',
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        if e_mod == 'adalin':
            e_mod_cls = AdalinModulatedEncoder
        elif e_mod == 'excitation':
            e_mod_cls = ExcitationModulatedEncoder
        else:
            raise ValueError(e_mod)
        self.e_mod = e_mod_cls(
            input_imsize=imsize,
            smallest_imsize=smallest_imsize,
            nc_in=3,
            nc_base=nc_base,
            nc_max=nc_max,
            z_dims=z_dims,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        assert c_gender == 'affine'
        self.c_gender = MinimalZGenderClassifier(z_dims=z_dims)

        assert f == 'mlp'
        self.f = Disentangler(
            z_dims=z_dims,
            n_layers=f_n_layers,
            lr_mul=f_lr_mul,
        )

        if g == 'excitation':
            g_cls = BlendExcitationModulatedGenerator
        elif g == 'style':
            g_cls = BlendConvertibleStyleGenerator
        else:
            raise ValueError(g)
        self.g = g_cls(
            output_imsize=imsize,
            init_imsize=smallest_imsize,
            nc_base=nc_base,
            nc_max=nc_max,
            z_dims=z_dims,
            k_blend=g_k_blend,
            norm=norm,
            weight_norm=weight_norm,
            actv=actv,
        )

        assert t == 'affine'
        self.t = GenderLatentTransform(z_dims=z_dims)
        self.male_const = torch.ones((1, 1), requires_grad=False).cuda()
        self.female_const = torch.zeros((1, 1), requires_grad=False).cuda()

    def forward(self, data, is_eval):
        bs = data['seed'].shape[0]

        # real rec and classify
        if self.real_enabled:
            real_z = self.e_pri(data['real_fg'])
            real_identity = self.e_mod(data['real_fg'], real_z)
            real_rec = self.g(real_identity, real_z, data['real_ibg'])
            real_gender_pred = self.c_gender(real_z).squeeze()
        else:
            real_rec = None
            real_gender_pred = None

        # fake gen
        if self.gan_enabled:
            fake_z = self.f(data['seed'])
            fake_gen = self.g(real_identity, fake_z, data['real_ibg'])
        else:
            fake_gen = None

        # swap
        if self.swap_enabled:
            male_z = self.e_pri(data['male_fg'])
            male_identity = self.e_mod(data['male_fg'], male_z)

            female_z = self.e_pri(data['female_fg'])
            female_identity = self.e_mod(data['female_fg'], female_z)

            female_z_pred = self.t(
                male_z.clone().detach(),
                self.male_const.expand(bs, 1),
            )
            male_z_pred = self.t(
                female_z.clone().detach(),
                self.female_const.expand(bs, 1),
            )

            if self.swap_rec_enabled or is_eval:
                male_rec = self.g(male_identity, male_z, data['male_ibg'])
                female_rec = self.g(
                    female_identity,
                    female_z,
                    data['female_ibg'],
                )
            else:
                male_rec = None
                female_rec = None
        else:
            raise Exception('todo')

        # inference
        real_swap = None
        mtf = None
        ftm = None
        if is_eval:
            if self.real_enabled:
                real_swap = self.g(
                    real_identity,
                    self.t(real_z, data['real_gender'].unsqueeze(1)),
                    data['real_ibg'],
                )
            if self.swap_enabled:
                mtf = self.g(male_identity, female_z_pred, data['male_ibg'])
                ftm = self.g(female_identity, male_z_pred, data['female_ibg'])

        return {
            'real_rec': real_rec,
            'real_gender_pred': real_gender_pred,
            'real_swap': real_swap,
            'fake_gen': fake_gen,
            'male_z': male_z,
            'male_identity': male_identity,
            'male_z_pred': male_z_pred,
            'male_rec': male_rec,
            'female_z': female_z,
            'female_identity': female_identity,
            'female_z_pred': female_z_pred,
            'female_rec': female_rec,
            'mtf': mtf,
            'ftm': ftm,
        }

    def init_params(self):
        self.e_pri.init_params()
        self.e_mod.init_params()
        self.c_gender.init_params()
        self.f.init_params()
        self.g.init_params()
        self.t.init_params()

    def print_info(self):
        self.e_pri.print_info()
        self.e_mod.print_info()
        self.c_gender.print_info()
        self.f.print_info()
        self.g.print_info()
        self.t.print_info()



class UltModel(Unit):
    def __init__(self,
        # configuration
        real_enabled=True,
        gan_enabled=True,
        swap_enabled=True,
        swap_rec_enabled=True,

        # general
        imsize=128,
        smallest_imsize=4,
        z_dims=512,
        nc_base=32,
        nc_max=512,

        # e pri
        e_pri='simple',

        # e mod
        e_mod='adalin',

        # c gender
        c_gender='affine',

        # f
        f='mlp',
        f_n_layers=6,
        f_lr_mul=0.1,

        # g
        g='adalin',

        # t
        t='affine',
    ):
        super().__init__()
        self.real_enabled = real_enabled
        self.gan_enabled = gan_enabled
        self.swap_enabled = swap_enabled
        self.swap_rec_enabled = swap_rec_enabled

        assert e_pri == 'simple'
        self.e_pri = SimpleZEncoder(
            input_imsize=imsize,
            smallest_imsize=4,
            z_dims=z_dims,
            k=3,
            k_init=3,
            nc_in=3,
            nc_base=nc_base,
            im2vec='mlp',
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        if e_mod == 'adalin':
            e_mod_cls = AdalinModulatedEncoder
        elif e_mod == 'excitation':
            e_mod_cls = ExcitationModulatedEncoder
        else:
            raise ValueError(e_mod)
        self.e_mod = e_mod_cls(
            input_imsize=imsize,
            smallest_imsize=smallest_imsize,
            nc_in=3,
            nc_base=nc_base,
            nc_max=nc_max,
            z_dims=z_dims,
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        assert c_gender == 'affine'
        self.c_gender = MinimalZGenderClassifier(z_dims=z_dims)

        assert f == 'mlp'
        self.f = Disentangler(
            z_dims=z_dims,
            n_layers=f_n_layers,
            lr_mul=f_lr_mul,
        )

        if g == 'adalin':
            g_cls = AdalinModulatedGenerator
        elif g == 'excitation':
            g_cls = ExcitationModulatedGenerator
        else:
            raise ValueError(g)
        self.g = g_cls(
            output_imsize=imsize,
            init_imsize=smallest_imsize,
            nc_in=3,
            nc_base=nc_base,
            nc_max=nc_max,
            z_dims=z_dims,
            norm='batch',
            weight_norm=False,
            actv='mish',
        )

        assert t == 'affine'
        self.t = GenderLatentTransform(z_dims=z_dims)
        self.male_const = torch.ones((1, 1), requires_grad=False).cuda()
        self.female_const = torch.zeros((1, 1), requires_grad=False).cuda()

    def forward(self, data, is_eval):
        bs = data['real'].shape[0]

        # real rec and classify
        if self.real_enabled or is_eval:
            real_z = self.e_pri(data['real'])
            real_identity = self.e_mod(data['real'], real_z)
            real_rec = self.g(real_identity, real_z)
            real_gender_pred = self.c_gender(real_z).squeeze()
        else:
            real_rec = None
            real_gender_pred = None

        # fake gen
        if self.gan_enabled or is_eval:
            fake_z = self.f(data['seed'])
            fake_gen = self.g(real_identity, fake_z)
        else:
            fake_gen = None

        # swap
        if self.swap_enabled or is_eval:
            male_z = self.e_pri(data['male'])
            male_identity = self.e_mod(data['male'], male_z)

            female_z = self.e_pri(data['female'])
            female_identity = self.e_mod(data['female'], female_z)

            female_z_pred = self.t(
                male_z.clone().detach(),
                self.male_const.expand(bs, 1),
            )
            male_z_pred = self.t(
                female_z.clone().detach(),
                self.female_const.expand(bs, 1),
            )

            if self.swap_rec_enabled or is_eval:
                male_rec = self.g(male_identity, male_z)
                female_rec = self.g(female_identity, female_z)
            else:
                male_rec = None
                female_rec = None
        else:
            raise Exception('todo')

        # inference
        if is_eval:
            mtf = self.g(male_identity, female_z_pred)
            ftm = self.g(female_identity, male_z_pred)
            real_swap = self.g(
                real_identity,
                self.t(real_z, data['real_gender'].unsqueeze(1)),
            )
        else:
            mtf = None
            ftm = None
            real_swap = None

        return {
            'real_rec': real_rec,
            'real_gender_pred': real_gender_pred,
            'real_swap': real_swap,
            'fake_gen': fake_gen,
            'male_z': male_z,
            'male_identity': male_identity,
            'male_rec': male_rec,
            'male_z_pred': male_z_pred,
            'female_z': female_z,
            'female_identity': female_identity,
            'female_rec': female_rec,
            'female_z_pred': female_z_pred,
            'mtf': mtf,
            'ftm': ftm,
        }

    def init_params(self):
        self.e_pri.init_params()
        self.e_mod.init_params()
        self.c_gender.init_params()
        if self.gan_enabled:
            self.f.init_params()
        self.g.init_params()
        self.t.init_params()

    def print_info(self):
        self.e_pri.print_info()
        self.e_mod.print_info()
        self.c_gender.print_info()
        if self.gan_enabled:
            self.f.print_info()
        self.g.print_info()
        self.t.print_info()
