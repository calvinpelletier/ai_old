#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
from ai_old.util.etc import resize_imgs
import torchvision.transforms.functional as TF
from ai_old.loss.perceptual.face import SoloFaceIdLoss


def normalize_imgs(imgs):
    imgs = (imgs + 1.) / 2. # -1,1 to 0,1
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return TF.normalize(imgs, mean, std)



def embellish_texts(texts):
    noun_embellishments = [
        '{}',
        'a {}',
        'a {}\'s face',
        'the face of a {}',
        'a photo of a {}',
        'a photo of a {}\'s face',
        'a picture of a {}',
        'a picture of a {}\'s face',
        'a painting of a {}',
        'a painting of a {}\'s face',
        'a drawing of a {}',
        'a drawing of a {}\'s face',
        'a portrait of a {}',
        'a portrait of a {}\'s face',
        'a sketch of a {}',
        'a sketch of a {}\'s face',
        'a cropped photo of a {}',
        'a cropped photo of a {}\'s face',
        'a close-up photo of a {}',
        'a close-up photo of a {}\'s face',
    ]
    adj_embellishments = [
        '{} face',
        'a {} face',
        'a photo of a {} face',
        'a picture of a {} face',
        'a painting of a {} face',
        'a drawing of a {} face',
        'a portrait of a {} face',
        'a sketch of a {} face',
        'a cropped photo of a {} face',
        'a close-up photo of a {} face',
    ]
    ret = ([], [])
    for t1, t2, type in texts:
        if type == 'noun':
            embellishments = noun_embellishments
        elif type == 'adj':
            embellishments = adj_embellishments
        else:
            raise Exception(type)
        for embellishment in embellishments:
            ret[0].append(embellishment.format(t1))
            ret[1].append(embellishment.format(t2))
    return ret


class SoloClipSwapDirLoss(nn.Module):
    def __init__(self, base_img, base_gender, base_w):
        super().__init__()

        # load clip model
        self.model, self.preprocess = clip.load(
            'ViT-B/32',
            device='cuda',
            jit=False,
        )
        self.model.eval()

        # load identity model
        self.face_id_model = SoloFaceIdLoss(
            base_img).eval().requires_grad_(False).to('cuda')

        # embellish texts
        texts = [
            ('male', 'female', 'adj'),
            ('boy', 'girl', 'noun'),
            ('man', 'woman', 'noun'),
            ('masculine', 'feminine', 'adj'),
        ]
        gender = base_gender[0]
        assert gender in [0., 1.]
        if gender == 0.:
            text = [(f, m, x) for m, f, x in texts]
        texts_pair = embellish_texts(texts)

        # find average text encoding delta
        avg_encs = []
        for texts in texts_pair:
            tokenized = clip.tokenize(texts).to('cuda')
            encs = self.model.encode_text(tokenized)
            encs = encs / encs.norm(dim=-1, keepdim=True)
            enc = encs.mean(dim=0)
            enc = enc / enc.norm()
            avg_encs.append(enc)
        text_delta = avg_encs[1] - avg_encs[0]
        text_delta = text_delta / text_delta.norm()
        self.register_buffer('text_delta', text_delta.detach())

        # encode base img
        base_img = resize_imgs(base_img, self.model.visual.input_resolution)
        base_img = normalize_imgs(base_img)
        base_img_enc = self.model.encode_image(base_img)
        base_img_enc = base_img_enc / base_img_enc.norm(dim=-1, keepdim=True)
        base_img_enc = base_img_enc.squeeze()
        self.register_buffer('base_img_enc', base_img_enc.detach())

        # save base w
        self.register_buffer('base_w', base_w.detach())

    def forward(self, img, w, id_weight, delta_weight):
        assert img.shape[0] == 1

        # clip loss
        resized_img = resize_imgs(img, self.model.visual.input_resolution)
        normed_img = normalize_imgs(resized_img)
        img_enc = self.model.encode_image(normed_img)
        img_enc = img_enc / img_enc.norm(dim=-1, keepdim=True)
        img_enc = img_enc.squeeze()
        img_delta = img_enc - self.base_img_enc
        img_delta = img_delta / img_delta.norm()
        # clip_loss = F.mse_loss(img_delta, self.text_delta)
        clip_loss = 1. - torch.dot(img_delta, self.text_delta)
        total_loss = clip_loss

        # id loss
        if id_weight > 0.:
            id_loss = self.face_id_model(img)
            total_loss += id_loss * id_weight

        # delta loss
        if delta_weight > 0.:
            delta_loss = F.mse_loss(w, self.base_w)
            total_loss += delta_loss * delta_weight

        return total_loss


class SoloClipSwapLoss(nn.Module):
    def __init__(self, base_img, base_gender, base_w):
        super().__init__()

        # load clip model
        self.model, self.preprocess = clip.load(
            'ViT-B/32',
            device='cuda',
            jit=False,
        )
        self.model.eval()

        # load identity model
        self.face_id_model = SoloFaceIdLoss(
            base_img).eval().requires_grad_(False).to('cuda')

        # embellish texts
        texts = [
            ('male', 'female', 'adj'),
            ('boy', 'girl', 'noun'),
            ('man', 'woman', 'noun'),
            ('masculine', 'feminine', 'adj'),
        ]
        texts_pair = embellish_texts(texts)
        gender = base_gender[0]
        assert gender in [0., 1.]
        if gender == 0.:
            texts = texts_pair[0]
        else:
            texts = texts_pair[1]

        # find average text encoding
        tokenized = clip.tokenize(texts).to('cuda')
        encs = self.model.encode_text(tokenized)
        encs = encs / encs.norm(dim=-1, keepdim=True)
        enc = encs.mean(dim=0)
        enc = enc / enc.norm()
        self.register_buffer('text_enc', enc.detach())

        # save base w
        self.register_buffer('base_w', base_w.detach())

    def forward(self, img, w, id_weight, delta_weight):
        assert img.shape[0] == 1

        # clip loss
        resized_img = resize_imgs(img, self.model.visual.input_resolution)
        normed_img = normalize_imgs(resized_img)
        img_enc = self.model.encode_image(normed_img)
        img_enc = img_enc / img_enc.norm(dim=-1, keepdim=True)
        cosine_sim = self.model.logit_scale.exp() * img_enc @ self.text_enc.t()
        clip_loss = 1. - cosine_sim / 100.
        total_loss = clip_loss.mean()

        # id loss
        if id_weight > 0.:
            id_loss = self.face_id_model(img).mean()
            total_loss += id_loss * id_weight

        # delta loss
        if delta_weight > 0.:
            delta_loss = F.mse_loss(w, self.base_w).mean()
            total_loss += delta_loss * delta_weight

        return total_loss


class GenderSwapClipDirLoss(nn.Module):
    def __init__(self, female_male_target_texts, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(
            'ViT-B/32',
            device=self.device,
            jit=False,
        )
        # self.upsample = nn.Upsample(scale_factor=7)
        # self.avg_pool = nn.AvgPool2d(kernel_size=imsize // 32)

        text_encs = []
        tokenized = clip.tokenize(female_male_target_texts).to(self.device)
        for x in tokenized:
            enc = self.model.encode_text(x.unsqueeze(0))
            text_encs.append(
                enc / enc.norm(dim=-1, keepdim=True))
        self.text_dirs = [
            text_encs[0] - text_encs[1], # ftm
            text_encs[1] - text_encs[0], # mtf
        ]

    def forward(self, src_imgs, swap_imgs, src_genders):
        src_imgs = resize_imgs(src_imgs, self.model.visual.input_resolution)
        swap_imgs = resize_imgs(swap_imgs, self.model.visual.input_resolution)
        src_imgs = normalize_imgs(src_imgs)
        swap_imgs = normalize_imgs(swap_imgs)
        n = src_imgs.shape[0]
        src_genders = src_genders.squeeze().cpu().numpy().astype(np.uint8)

        src_img_enc = self.model.encode_image(src_imgs)
        src_img_enc = src_img_enc / src_img_enc.norm(dim=-1, keepdim=True)

        swap_img_enc = self.model.encode_image(swap_imgs)
        swap_img_enc = swap_img_enc / swap_img_enc.norm(dim=-1, keepdim=True)

        img_dir = swap_img_enc - src_img_enc

        total_loss = 0.
        for i in range(n):
            text_dir = self.text_dirs[src_genders[i]].clone().detach()
            loss_per_img = self.model.logit_scale.exp() * img_dir @ text_dir.t()
            total_loss += 1. - loss_per_img / 100.
        total_loss /= n
        return total_loss


class MultiTextClipLoss(nn.Module):
    def __init__(self, target_texts, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(
            'ViT-B/32',
            device=self.device,
            jit=False,
        )

        self.text_keys = []
        for k, v in target_texts.items():
            assert k != 'all', '"all" is a special keyword'
            self.text_keys.append(k)
            tokenized = clip.tokenize(v).to(self.device)
            enc = self.model.encode_text(tokenized)
            enc = enc / enc.norm(dim=-1, keepdim=True)
            self.register_buffer(f'{k}_text_encs', enc.detach())

    def forward(self, key, imgs, text_idxs):
        n = imgs.shape[0]
        imgs = resize_imgs(imgs, self.model.visual.input_resolution)
        imgs = normalize_imgs(imgs)
        text_idxs = text_idxs.squeeze(dim=1).cpu().numpy().astype(np.uint8)

        img_enc = self.model.encode_image(imgs)
        img_enc = img_enc / img_enc.norm(dim=-1, keepdim=True)

        if key == 'all':
            ret = {}
            for k in self.text_keys:
                text_encs = getattr(self, f'{k}_text_encs')
                ret[k] = self._calc_loss(text_encs, text_idxs, img_enc, n)
            return ret
        else:
            text_encs = getattr(self, f'{key}_text_encs')
            return self._calc_loss(text_encs, text_idxs, img_enc, n)

    def _calc_loss(self, text_encs, text_idxs, img_enc, n):
        total_loss = 0.
        for i in range(n):
            text_enc = text_encs[text_idxs[i]].clone().detach()
            loss_per_img = self.model.logit_scale.exp() * \
                img_enc[i] @ text_enc.t()
            total_loss += 1. - loss_per_img.mean() / 100.
        total_loss /= n
        return total_loss


class ClipLoss(nn.Module):
    def __init__(self, target_text, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(
            'ViT-B/32',
            device=self.device,
            jit=False,
        )

        tokenized = clip.tokenize([target_text]).to(self.device)
        enc = self.model.encode_text(tokenized)
        enc = enc / enc.norm(dim=-1, keepdim=True)
        self.register_buffer('text_enc', enc.detach())

    def forward(self, imgs):
        imgs = resize_imgs(imgs, self.model.visual.input_resolution)
        imgs = normalize_imgs(imgs)
        img_enc = self.model.encode_image(imgs)
        img_enc = img_enc / img_enc.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * img_enc @ self.text_enc.t()
        loss = 1. - logits_per_image / 100.
        return loss.mean()


class ManualClipLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model, self.preprocess = clip.load(
            'ViT-B/32',
            device=self.device,
            jit=False,
        )

    def prep_texts(self, texts):
        ret = []
        for text in texts:
            tokenized = clip.tokenize([text]).to(self.device)
            enc = self.model.encode_text(tokenized)
            enc = enc / enc.norm(dim=-1, keepdim=True)
            ret.append(enc.detach())
        return ret

    def forward(self, imgs, text_enc):
        imgs = resize_imgs(imgs, self.model.visual.input_resolution)
        imgs = normalize_imgs(imgs)
        img_enc = self.model.encode_image(imgs)
        img_enc = img_enc / img_enc.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * img_enc @ text_enc.t()
        loss = 1. - logits_per_image / 100.
        return loss.mean()
