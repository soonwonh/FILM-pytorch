
from .options import Options
import torch
import torch.nn as nn
import torch.nn.functional as F


class SubTreeExtractor(nn.Module):
    def __init__(self, config):
        super(SubTreeExtractor, self).__init__()
        k = config.filters # 64 filters
        n = config.sub_levels # 4
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size=3, padding='same'))
        self.convs.append(nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, padding='same'))
        self.convs.append(nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding='same'))
        self.convs.append(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding='same'))
        
        for i in range(0, n-2): # todo : need to find in_channels[i]
            self.convs.append(nn.Conv2d(in_channels = (k << (i)), out_channels = (k << (i+1)), kernel_size=3, padding='same'))
            self.convs.append(nn.Conv2d(in_channels = (k << (i+1)), out_channels = (k << (i+1)), kernel_size=3, padding='same'))
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, image, n):

        head = image
        pyramid = []
        for i in range(n):
            head = self.leaky_relu(self.convs[2*i](head))
            head = self.leaky_relu(self.convs[2*i+1](head))
            pyramid.append(head)
            if i < n-1:
                head = self.pool(head)
        return pyramid


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.extract_sublevels = SubTreeExtractor(config)
        self.options = config

    def forward(self, image_pyramid):
        sub_pyramids = []
        for i in range(len(image_pyramid)):
            capped_sub_levels = min(len(image_pyramid) - i, self.options.sub_levels) # (4, 4, 4, 4, 3, 2, 1)
            sub_pyramids.append(self.extract_sublevels(image_pyramid[i], capped_sub_levels))
        feature_pyramid = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.options.sub_levels):
                if j <= i:
                    features = torch.concat([features, sub_pyramids[i-j][j]], axis=1) # we concat feature pyramid along channel axis
            feature_pyramid.append(features)
        return feature_pyramid
