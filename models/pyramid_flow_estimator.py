
from .utils import warp
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowEstimator(nn.Module):
    def __init__(self, num_convs, feature_levels, num_filters):
        super(FlowEstimator, self).__init__()
        #print(f"num_convs {num_convs} num_filters {num_filters}")
        feature_pyramids = [64, 192, 448, 960]
        self._convs = nn.ModuleList()
        self._convs.append(nn.Conv2d(in_channels = feature_pyramids[feature_levels], out_channels = num_filters, kernel_size=3, padding='same'))
        for i in range(1, num_convs):
            self._convs.append(nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size=3, padding='same'))
        self._convs.append(nn.Conv2d(in_channels = num_filters, out_channels = num_filters//2, kernel_size=1, padding='same'))
        self._convs.append(nn.Conv2d(in_channels = num_filters//2, out_channels=2, kernel_size=1, padding='same'))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, features_a, features_b):
        net = torch.concat([features_a, features_b], axis=1)
        for i in range(len(self._convs)-1):
            conv_ = self._convs[i]
            net = self.leaky_relu(conv_(net))
        conv_ = self._convs[-1]
        net = conv_(net)
        return net

""" for your convenience """
# flow_convs=[3, 3, 3, 3]
# specialized_levels=3
# flow_filters=[32, 64, 128, 256]
# 'pyramid_levels': 7

class PyramidFlowEstimator(nn.Module):
    def __init__(self, config):
        super(PyramidFlowEstimator, self).__init__()
        self._predictors = nn.ModuleList()
        for i in range(config.specialized_levels):  # 3 (0, 1, 2)
            self._predictors.append(FlowEstimator(num_convs=config.flow_convs[i], feature_levels=i, num_filters=config.flow_filters[i]))
        shared_predictor = FlowEstimator(num_convs=config.flow_convs[-1], feature_levels=config.specialized_levels, num_filters=config.flow_filters[-1])
        for i in range(config.specialized_levels, config.pyramid_levels):
            self._predictors.append(shared_predictor)

    def forward(self, feature_pyramid_a, feature_pyramid_b):
        
        levels = len(feature_pyramid_a)
        v = self._predictors[-1](feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in reversed(range(0, levels-1)):
            v = F.interpolate(v, scale_factor=2)
            warped = warp(feature_pyramid_b[i], v)
            v_residual = self._predictors[i](feature_pyramid_a[i], warped)
            residuals.append(v_residual)
            v = v_residual + v
        return list(reversed(residuals))
