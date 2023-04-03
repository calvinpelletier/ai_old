#!/usr/bin/env python3
import torch
from ai_old.util.etc import print_, resize_imgs
from ai_old.trainer.aae import AaeTrainer
from ai_old.util.factory import build_model, build_augpipe
import copy
from external.sg2 import training_stats
import external.sg2.misc as misc
from ai_old.loss.perceptual.trad import PerceptualLoss
from external.optimizer import get_optimizer
from ai_old.trainer.phase import TrainingPhase
import lpips
from ai_old.loss.perceptual.ahanu import AhanuPercepLoss
from ai_old.trainer.gan import StyleGanTrainer
from ai_old.nn.models.seg.outer_seg import get_masks
import cv2
from external.sg2.misc import print_module_summary
from ai_old.util.outer import get_dilate_kernel, get_outer_boundary_mask


class SmartOutpaintAaeTrainer(AaeTrainer):
    def parse_seg(self, seg):
        n = seg.shape[0]
        ret = []
        for i in range(n):
            face = seg[i] == 0
            hair = seg[i] == 1
            facehair = torch.bitwise_or(face, hair).float()
            inner_gan_mask = facehair * self.inner_mask

            dilated_facehair = cv2.dilate(
                facehair.numpy() * 255., self.dilate_kernel)
            dilated_facehair = torch.tensor(dilated_facehair / 255.)
            dilated_facehair = (dilated_facehair > 0.5).float()
            inv_inner_gan_mask = 1. - inner_gan_mask
            inpaint_mask = dilated_facehair * inv_inner_gan_mask * \
                self.inv_outer_boundary_mask
            ret.append(inpaint_mask.unsqueeze(0))

            # gt_mask = torch.ones_like(inpaint_mask) * (1. - inpaint_mask) * \
            #     inv_inner_gan_mask

            # return inner_gan_mask.to(self.device).to(torch.float32), \
            #     inpaint_mask.to(self.device).to(torch.float32), \
            #     gt_mask.to(self.device).to(torch.float32)

        ret = torch.cat(ret, dim=0)
        return ret.to(self.device).to(torch.float32)


    def parse_batch(self, x):
        img = x['y'].to(self.device).to(torch.float32) / 127.5 - 1

        if 'mask' in x:
            mask = x['mask'].to(self.device).to(torch.float32) / 255.
        else:
            mask = self.parse_seg(x['seg'])

        return img, mask


    def run_batch(self, batch, batch_idx, cur_step):
        # prep_data
        with torch.autograd.profiler.record_function('data_prep'):
            img, mask = self.parse_batch(batch)
            phase_real = img.split(self.batch_gpu)
            phase_inpaint_mask = mask.split(self.batch_gpu)

        # run training phases
        for phase in self.phases:
            if batch_idx % phase.interval != 0:
                continue

            phase.init_gradient_accumulation()

            # accumulate gradients over multiple rounds
            for round_idx, (
                real,
                # inner_gan_mask,
                inpaint_mask,
                # gt_mask,
            ) in enumerate(zip(
                phase_real,
                # phase_inner_gan_mask,
                phase_inpaint_mask,
                # phase_gt_mask,
            )):
                sync = (round_idx == self.batch_size // \
                    (self.batch_gpu * self.num_gpus) - 1)
                gain = phase.interval
                self.loss.accumulate_gradients(
                    phase=phase.name,
                    real=real,
                    # inner_gan_mask=inner_gan_mask,
                    inpaint_mask=inpaint_mask,
                    # gt_mask=gt_mask,
                    sync=sync,
                    gain=gain,
                )

            phase.update_params()

        self._update_model_ema(cur_step)

        self._run_ada_heuristic(batch_idx)


    def run_G(self, real, inpaint_mask, do_req_losses, sync):
        G = self.ddp_modules['G']
        with misc.ddp_sync(G, sync):
            out = G(real, inpaint_mask)

        if self.cfg.loss.G.rec.enabled and do_req_losses:
            out_256 = resize_imgs(out, 256)
            real_256 = resize_imgs(real, 256)
            perceptual = self.ddp_modules['perceptual']
            with misc.ddp_sync(perceptual, sync):
                percep_loss = perceptual(out_256, real_256)
            return out, percep_loss
        else:
            return out


    def run_D(self, img, mask, sync):
        D = self.ddp_modules['D']
        augment_pipe = self.ddp_modules['augment_pipe']

        # augmentation
        if augment_pipe is not None:
            img = augment_pipe(img)

        # discrimination
        with misc.ddp_sync(D, sync):
            logits = D(img, mask)

        return logits


    def get_eval_fn(self):
        def eval_fn(model, batch, batch_size, device):
            img, mask = self.parse_batch(batch)
            out = model(img, mask)
            return img, out
        return eval_fn


    def get_sample_fn(self):
        def sample_fn(model, batch, batch_size, device):
            img, mask = self.parse_batch(batch)
            out = model(img, mask)
            out = (out * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return out
        return sample_fn


    def _init_modules(self):
        cfg = self.cfg

        mask_provided = False
        if hasattr(cfg.trainer, 'mask_provided'):
            mask_provided = cfg.trainer.mask_provided
        if not mask_provided:
            # init masks and dilate kernel
            self.inner_mask = get_masks(
                cfg.dataset.seg_imsize, cfg.dataset.seg_inner_imsize)[0]
            self.dilate_kernel = get_dilate_kernel(cfg.dataset.seg_imsize)
            self.outer_boundary_mask = get_outer_boundary_mask(
                cfg.dataset.seg_imsize)
            self.inv_outer_boundary_mask = 1. - self.outer_boundary_mask

        # TODO: initialized aae should be considered not from scratch
        from_scratch = not cfg.resume
        assert from_scratch, 'TODO'

        # build models
        print_(self.rank, '[INFO] initializing models...')
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)
        self.D = build_model(
            cfg,
            cfg.model.D,
        ).train().requires_grad_(False).to(self.device)

        # model ema
        self.G_ema = copy.deepcopy(self.G).eval()
        self.ema_ksteps = 10
        self.ema_rampup = 0.05 if from_scratch else None

        # perceptual loss
        if cfg.loss.G.rec.enabled:
            perceptual_type = cfg.loss.G.rec.perceptual_type
            if perceptual_type == 'lpips':
                raise Exception('lpips_alex or lpips_vgg')
            elif perceptual_type == 'lpips_alex':
                percep = lpips.LPIPS(net='alex')
            elif perceptual_type == 'lpips_vgg':
                percep = lpips.LPIPS(net='vgg')
            elif perceptual_type == 'trad':
                percep = PerceptualLoss()
            elif perceptual_type == 'ahanu':
                percep = AhanuPercepLoss(cfg.loss.G.rec.perceptual_version)
            else:
                raise Exception(perceptual_type)
            percep = percep.eval().requires_grad_(False).to(self.device)
            self.perceptual_loss_model = percep

        # augmentation
        print_(self.rank, '[INFO] initializing augmentation...')
        self.augment_pipe = None
        self.ada_stats = None
        if cfg.trainer.aug.enabled:
            self.augment_pipe = build_augpipe(
                cfg.trainer.aug,
            ).train().requires_grad_(False).to(self.device)
            self.augment_pipe.p.copy_(torch.as_tensor(0))
            if cfg.trainer.aug.target is not None:
                self.ada_stats = training_stats.Collector(
                    regex='Loss/signs/real')
            if cfg.trainer.aug.speed == 'auto':
                cfg.trainer.aug.speed = 500 if from_scratch else 100


    def _get_modules_for_distribution(self):
        ret = [
            ('G', self.G, True),
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
        ]
        if self.cfg.loss.G.rec.enabled:
            ret.append(('perceptual', self.perceptual_loss_model, False))
        return ret


    def print_models(self):
        imsize = self.cfg.dataset.imsize
        img = torch.empty(
            [self.batch_gpu, 3, imsize, imsize],
            device=self.device,
        )
        inpaint_mask = torch.empty(
            [self.batch_gpu, imsize, imsize],
            device=self.device,
        )
        out = print_module_summary(self.G, [img, inpaint_mask])
        print_module_summary(self.D, [out, inpaint_mask])


class OutpaintAaeTrainer(AaeTrainer):
    def run_G(self, real, do_req_losses, sync, return_ws=False):
        G = self.ddp_modules['G']
        perceptual_loss_model = self.ddp_modules['perceptual']

        with misc.ddp_sync(G, sync):
            output = G(real)

        if not do_req_losses:
            return output

        with misc.ddp_sync(perceptual_loss_model, sync):
            perceptual_loss = perceptual_loss_model(output, real)

        return output, perceptual_loss


    def _init_modules(self):
        print_(self.rank, '[INFO] initializing models...')
        cfg = self.cfg

        # build generator
        self.G = build_model(
            cfg,
            cfg.model.G,
        ).train().requires_grad_(False).to(self.device)

        # load discriminator
        self.D = build_model(
            cfg,
            cfg.model.D,
        ).train().requires_grad_(False).to(self.device)

        # perceptual loss
        perceptual_type = cfg.loss.G.rec.perceptual_type
        if perceptual_type == 'lpips':
            raise Exception('lpips_alex or lpips_vgg')
        elif perceptual_type == 'lpips_alex':
            percep = lpips.LPIPS(net='alex')
        elif perceptual_type == 'lpips_vgg':
            percep = lpips.LPIPS(net='vgg')
        elif perceptual_type == 'trad':
            percep = PerceptualLoss()
        elif perceptual_type == 'ahanu':
            percep = AhanuPercepLoss(cfg.loss.G.rec.perceptual_version)
        else:
            raise Exception(perceptual_type)
        percep = percep.eval().requires_grad_(False).to(self.device)
        self.perceptual_loss_model = percep

        # resume training
        assert not cfg.resume
        from_scratch = True

        # model ema
        self.G_ema = copy.deepcopy(self.G).eval()
        self.ema_ksteps = 10
        self.ema_rampup = 0.05 if from_scratch else None

        # augmentation
        print_(self.rank, '[INFO] initializing augmentation...')
        self.augment_pipe = None
        self.ada_stats = None
        if cfg.trainer.aug.enabled:
            self.augment_pipe = build_augpipe(
                cfg.trainer.aug,
            ).train().requires_grad_(False).to(self.device)
            self.augment_pipe.p.copy_(torch.as_tensor(0))
            if cfg.trainer.aug.target is not None:
                self.ada_stats = training_stats.Collector(
                    regex='Loss/signs/real')
            if cfg.trainer.aug.speed == 'auto':
                cfg.trainer.aug.speed = 500 if from_scratch else 100


    def _get_modules_for_distribution(self):
        return [
            ('G', self.G, True),
            ('D', self.D, True),
            (None, self.G_ema, False),
            ('augment_pipe', self.augment_pipe, True),
            ('perceptual', self.perceptual_loss_model, False),
        ]
