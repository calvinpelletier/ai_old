#!/usr/bin/env python3
import torch
import torch.nn as nn
from external.sg2 import persistence
from ai_old.nn.models.transform.gender import GenderLatentTransform
from ai_old.util.factory import build_model_from_exp
from ai_old.util.params import init_params
from ai_old.util import config
import copy
import external.sg2.misc as misc


@persistence.persistent_class
class GenderModspaceTransform(nn.Module):
    def __init__(self,
        z_dims,
        num_ws,
        n_mlp_layers=2,
    ):
        super().__init__()
        self.z_dims = z_dims
        self.num_ws = num_ws

        mlps = []
        for i in range(num_ws):
            mlps.append(GenderLatentTransform(
                z_dims=z_dims,
                n_mlp_layers=n_mlp_layers,
            ))
        self.mlps = nn.ModuleList(mlps)

    def forward(self, ws, src_gender):
        misc.assert_shape(ws, [None, self.num_ws, self.z_dims])
        ws = ws.to(torch.float32)

        ret = []
        for i in range(self.num_ws):
            ret.append(self.mlps[i](ws.narrow(1, i, 1).squeeze(1), src_gender))

        return torch.stack(ret, dim=1)


@persistence.persistent_class
class RelativisticGenderModspaceTransform(nn.Module):
    def __init__(self,
        z_dims,
        num_ws,
        n_mlp_layers=2,
        n_main_mlp_layers=4,
    ):
        super().__init__()
        self.z_dims = z_dims
        self.num_ws = num_ws

        self.main = GenderLatentTransform(
            z_dims=z_dims,
            n_mlp_layers=n_main_mlp_layers,
        )

        mlps = []
        for i in range(num_ws):
            mlps.append(GenderLatentTransform(
                z_dims=z_dims,
                n_mlp_layers=n_mlp_layers,
            ))
        self.mlps = nn.ModuleList(mlps)

    def forward(self, ws, src_gender):
        misc.assert_shape(ws, [None, self.num_ws, self.z_dims])
        ws = ws.to(torch.float32)
        w_avg = torch.mean(ws, dim=1)
        # print('w_avg', w_avg.shape)

        new_w_avg = self.main(w_avg, src_gender)

        ret = []
        for i in range(self.num_ws):
            w_rel = ws.narrow(1, i, 1).squeeze(1) - w_avg
            new_w_rel = self.mlps[i](w_rel, src_gender)
            ret.append(new_w_rel + new_w_avg)

        return torch.stack(ret, dim=1)


@persistence.persistent_class
class ModspaceSwap(nn.Module):
    def __init__(self,
        cfg,
        rec_exp='blend-rec/1/0',
        t_type='reg',
        n_mlp_layers=2,
        n_main_mlp_layers=4,
    ):
        super().__init__()

        # load the rec model and its config
        og_G, og_cfg = build_model_from_exp(rec_exp, 'G')

        # hparams
        self.z_dims = og_G.z_dims
        self.num_ws = og_G.num_ws
        self.imsize = og_G.imsize

        # transfer modules
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        self.e = copy.deepcopy(og_G.e)
        self.e.requires_grad_(False)

        del og_G

        # build swapper
        if t_type == 'reg':
            self.t = GenderModspaceTransform(
                z_dims=self.z_dims,
                num_ws=self.num_ws,
                n_mlp_layers=n_mlp_layers,
            )
        elif t_type == 'relativistic':
            self.t = RelativisticGenderModspaceTransform(
                z_dims=self.z_dims,
                num_ws=self.num_ws,
                n_mlp_layers=n_mlp_layers,
                n_main_mlp_layers=n_main_mlp_layers,
            )
        else:
            raise Exception('t_type')
        self.t.apply(init_params())

    def forward(self, x_img, x_gender, y_img, y_gender):
        x_ws = self.e(x_img)
        y_ws = self.e(y_img)
        x_ws_pred = self.t(y_ws, y_gender)
        y_ws_pred = self.t(x_ws, x_gender)
        x_full_pred = self.g(x_ws_pred)
        y_full_pred = self.g(y_ws_pred)
        return x_full_pred, y_full_pred

    def prep_for_train_phase(self):
        self.t.requires_grad_(True)



@persistence.persistent_class
class BlendModspaceSwap(nn.Module):
    def __init__(self,
        cfg,
        rec_exp='blend-rec/1/0',
        t_type='reg',
        n_mlp_layers=2,
        n_main_mlp_layers=4,
    ):
        super().__init__()

        # load the rec model and its config
        og_G, og_cfg = build_model_from_exp(rec_exp, 'G')

        # hparams
        self.z_dims = og_G.z_dims
        self.num_ws = og_G.num_ws
        self.imsize = og_G.imsize

        # transfer modules
        self.g = copy.deepcopy(og_G.g)
        self.g.requires_grad_(False)
        self.e = copy.deepcopy(og_G.e)
        self.e.requires_grad_(False)

        del og_G

        # build swapper
        if t_type == 'reg':
            self.t = GenderModspaceTransform(
                z_dims=self.z_dims,
                num_ws=self.num_ws,
                n_mlp_layers=n_mlp_layers,
            )
        elif t_type == 'relativistic':
            self.t = RelativisticGenderModspaceTransform(
                z_dims=self.z_dims,
                num_ws=self.num_ws,
                n_mlp_layers=n_mlp_layers,
                n_main_mlp_layers=n_main_mlp_layers,
            )
        else:
            raise Exception('t_type')
        self.t.apply(init_params())

    def forward(self,
        x_fg,
        x_ibg,
        x_gender,
        y_fg,
        y_ibg,
        y_gender,
    ):
        x_ws, x_ibg_encs = self.e(x_fg, x_ibg)
        y_ws, y_ibg_encs = self.e(y_fg, y_ibg)
        x_ws_pred = self.t(y_ws, y_gender)
        y_ws_pred = self.t(x_ws, x_gender)
        x_full_pred = self.g(y_ibg_encs, x_ws_pred)
        y_full_pred = self.g(x_ibg_encs, y_ws_pred)
        return x_full_pred, y_full_pred

    def prep_for_train_phase(self):
        self.t.requires_grad_(True)
