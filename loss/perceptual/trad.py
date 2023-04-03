import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
# from ai_old.models.e.arcface import ArcFace


class PerceptualLoss(nn.Module):
    def __init__(self, type='vgg19', return_x_out=False):
        super().__init__()
        assert type == 'vgg19' # only supporting vgg19 for now
        self.model = _VGG19().cuda()
        self.model.eval()
        self.loss = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # self.return_x_out = return_x_out # because it's reused by a loss in mk2

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

    def forward(self, x, y):
        x = self._imagenet_normalization(x)
        y = self._imagenet_normalization(y)
        x = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        y = F.interpolate(
            y,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        x_out, y_out = self.model(x), self.model(y)
        ret = torch.cuda.FloatTensor(1).fill_(0)
        # iterate over layers
        for i in range(len(x_out)):
            ret += self.weights[i] * self.loss(x_out[i], y_out[i])
        # if self.return_x_out:
        #     return ret, x_out
        # else:
        return ret.squeeze()

    def _imagenet_normalization(self, x):
        # x = (x + 1) / 2
        return (x - self.mean) / self.std


class SoloPerceptualLoss(nn.Module):
    def __init__(self, target_img, type='vgg19'):
        super().__init__()
        assert type == 'vgg19' # only supporting vgg19 for now
        self.model = _VGG19().cuda()
        self.model.eval()
        self.loss = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()

        target = self._imagenet_normalization(target_img)
        target = F.interpolate(
            target,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        target = self.model(target)
        self.register_buffer('target', target.clone().detach())

    def forward(self, x):
        x = self._imagenet_normalization(x)
        x = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        )
        x_out = self.model(x)
        y_out = self.target

        ret = torch.cuda.FloatTensor(1).fill_(0)
        for i in range(len(x_out)):
            ret += self.weights[i] * self.loss(x_out[i], y_out[i])
        return ret.squeeze()

    def _imagenet_normalization(self, x):
        return (x - self.mean) / self.std


# class FacePerceptualLoss(nn.Module):
#     def __init__(self, type='arcface'):
#         super().__init__()
#         assert type == 'arcface' # only supporting arcface resnet 50 for now
#         self.facenet = ArcFace(
#             input_size=112,
#             num_layers=50,
#             drop_ratio=0.6,
#             mode='ir_se',
#         )
#         model_path = os.path.join(c.ASI_DATA_PATH, 'arcface')
#         self.facenet.load_state_dict(torch.load(model_path))
#         self.face_pool = nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()
#
#     def extract_feats(self, x):
#         x = x[:, :, 35:223, 32:220]
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         return x_feats
#
#     def forward(self, y_hat, y, x):
#         n_samples = x.shape[0]
#         x_feats = self.extract_feats(x)
#         y_feats = self.extract_feats(y)
#         y_hat_feats = self.extract_feats(y_hat)
#         y_feats = y_feats.detach()
#         loss = 0
#         sim_improvement = 0
#         id_logs = []
#         count = 0
#         for i in range(n_samples):
#             diff_target = y_hat_feats[i].dot(y_feats[i])
#             diff_input = y_hat_feats[i].dot(x_feats[i])
#             diff_views = y_feats[i].dot(x_feats[i])
#             id_logs.append({'diff_target': float(diff_target),
#                             'diff_input': float(diff_input),
#                             'diff_views': float(diff_views)})
#             loss += 1 - diff_target
#             id_diff = float(diff_target) - float(diff_views)
#             sim_improvement += id_diff
#             count += 1
#         return loss / count, sim_improvement / count, id_logs


class MaskedPerceptualLoss(nn.Module):
    def __init__(self, type, maximize=False):
        super().__init__()
        assert type == 'vgg19' # only supporting vgg19 for now
        self.maximize = maximize
        self.model = _VGG19().cuda()
        self.loss = nn.L1Loss()
        self.loss_sum = nn.L1Loss(reduction='sum')
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def masked_loss(self, x, y, mask):
        n, c, h, w = x.size()
        mask = F.interpolate(mask, size=(h, w), mode='nearest')
        # diff = (x - y).abs()
        # loss = diff.sum() / (label.sum() + 1e-5)
        loss = self.loss_sum(x * mask, y * mask) / (mask.sum() * c + 1e-5)
        return loss

    def forward(self, x, y, mask):
        x_out, y_out = self.model(x), self.model(y)
        mask = torch.unsqueeze(mask, 1)
        # print('x_out', x_out[0], x_out[0].shape)
        # print('y_out', y_out[0], y_out[0].shape)
        # print('mask', mask, mask.shape)
        loss = 0
        for i in range(len(x_out)):
            loss += self.weights[i] * self.masked_loss(
                x_out[i],
                y_out[i],
                mask,
            )
        if self.maximize:
            return 2. - loss
        else:
            return loss


class _VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
