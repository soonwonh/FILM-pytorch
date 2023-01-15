import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torchvision import models
from .utils import AntiAliasInterpolation2d

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
                                       requires_grad=False)
        self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
                                      requires_grad=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class ImagePyramide(torch.nn.Module):
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = []
        for scale in scales:
            downs.append(AntiAliasInterpolation2d(num_channels, scale))
        self.downs = downs

    def forward(self, x):
        out_dict = []
        for down_module in self.downs:
            out_dict.append(down_module(x))
        return out_dict


class PerceptualLoss(nn.Module):
    def __init__(self,
                 scales=[1, 0.5, 0.25, 0.125],
                 loss_weights=[1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]
    ):
        super(PerceptualLoss, self).__init__()
        self.pyramid = ImagePyramide(scales, 3)
        self.vgg = Vgg19()
        self.scales = scales
        self.loss_weights = loss_weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def perceptual(self, p_vgg, g_vgg):
        loss = 0
        for p, g in zip(p_vgg, g_vgg):
            for i, weight in enumerate(self.loss_weights):
                loss += weight * (self.l1(p, g).mean()) 
        return loss
    def gram(self, p_vgg, g_vgg):
        loss = 0
        for p, g in zip(p_vgg, g_vgg):
            for i, weight in enumerate(self.loss_weights):
                loss += weight * (self.l2(p, g).mean()) 
        return loss
    def forward(self, pred, gt):
        preds = self.pyramid(pred)
        gts = self.pyramid(gt)

        perceptual_loss, style_loss = 0, 0
        for p, g in zip(preds, gts):
            p_vgg = self.vgg(p)
            g_vgg = self.vgg(g)

            perceptual_loss += self.perceptual(p_vgg, g_vgg)
            style_loss += self.gram(self.compute_gram(p_vgg), self.compute_gram(g_vgg))

        return (perceptual_loss , style_loss)

    def compute_gram(self, feature_pyramid):
        gram = []
        for x in feature_pyramid:
            #print(f"feature {x.shape}")
            b, c, h, w = x.shape
            f = x.view(b, c, w * h)
            f_T = f.transpose(1,2)
            G = f.bmm(f_T) / (h * w * c)
            gram.append(G)
        return gram
