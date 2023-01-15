
import torch
import torch.nn as nn
import torch.nn.functional as F

from .options import Options

def build_image_pyramid(image, options):

    levels = options.pyramid_levels
    pyramid = []
    pool = nn.AvgPool2d(kernel_size=2, stride=2)
    for i in range(0, levels):
        pyramid.append(image)
        if i < levels-1:
            image = pool(image)
    return pyramid

def warp(image, flow):
    warped = F.grid_sample(image, torch.permute(flow, (0, 2, 3, 1)), align_corners=False)
    return warped

def multiply_pyramid(pyramid, scalar):
    return [ torch.permute(torch.permute(image, (1, 2, 3, 0)) * scalar, [3, 0, 1, 2]) for image in pyramid]

def flow_pyramid_synthesis(residual_pyramid):
    flow = residual_pyramid[-1]
    flow_pyramid = [flow]
    for residual_flow in reversed(residual_pyramid[:-1]):
        level_size = residual_flow.shape[1:3]
        flow = F.interpolate(flow, scale_factor=2)
        flow = residual_flow + flow
        flow_pyramid.append(flow)
    return list(reversed(flow_pyramid))

def pyramid_warp(feature_pyramid, flow_pyramid):
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(features, flow))
    return warped_feature_pyramid

def concatenate_pyramids(pyramid1, pyramid2):
    result = []
    for features1, features2 in zip(pyramid1, pyramid2):
        result.append(torch.concat([features1, features2], axis=1))
    return result
