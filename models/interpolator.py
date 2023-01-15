import torch
import torch.nn as nn

from .feature_extractor import FeatureExtractor
from .fusion import Fusion
from .pyramid_flow_estimator import PyramidFlowEstimator
from . import utils

def create_model(config):
    return FILM_interpolator(config)

class FILM_interpolator(nn.Module):

    def __init__(self, config):
        super(FILM_interpolator, self).__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(self.config)
        self.flow_estimator = PyramidFlowEstimator(self.config)
        self.fuse = Fusion(self.config)

    def forward(self, batch):

        x0, x1, time, y = batch['x0'], batch['x1'], batch['time'], batch['y']

        image_pyramids = [
            utils.build_image_pyramid(x0, self.config),
            utils.build_image_pyramid(x1, self.config)
        ]

        feature_pyramids = [self.feature_extractor(image_pyramids[0]), self.feature_extractor(image_pyramids[1])]

        forward_residual_flow = self.flow_estimator(feature_pyramids[0], feature_pyramids[1])
        backward_residual_flow = self.flow_estimator(feature_pyramids[1], feature_pyramids[0])

        fusion_pyramid_levels = self.config.fusion_pyramid_levels
        forward_flow_pyramid = utils.flow_pyramid_synthesis(forward_residual_flow)[:fusion_pyramid_levels]
        backward_flow_pyramid = utils.flow_pyramid_synthesis(backward_residual_flow)[:fusion_pyramid_levels]

        mid_time = torch.ones_like(time) * 0.5
        backward_flow = utils.multiply_pyramid(backward_flow_pyramid, mid_time[:, 0])
        forward_flow = utils.multiply_pyramid(forward_flow_pyramid, 1 - mid_time[:, 0])

        pyramids_to_warp = [ # fusion_pyramid_levels: 5
            utils.concatenate_pyramids(image_pyramids[0][:fusion_pyramid_levels], feature_pyramids[0][:fusion_pyramid_levels]),
            utils.concatenate_pyramids(image_pyramids[1][:fusion_pyramid_levels], feature_pyramids[1][:fusion_pyramid_levels])
        ]

        forward_warped_pyramid = utils.pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid = utils.pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned_pyramid = utils.concatenate_pyramids(forward_warped_pyramid, backward_warped_pyramid)
        aligned_pyramid = utils.concatenate_pyramids(aligned_pyramid, backward_flow)
        aligned_pyramid = utils.concatenate_pyramids(aligned_pyramid, forward_flow)

        prediction = self.fuse(aligned_pyramid)

        output_color = prediction[...,:]
        outputs = {'image': output_color}

        if self.config.use_aux_outputs:
            outputs.update({
                'x0_warped': forward_warped_pyramid[0][..., 0:3],
                'x1_warped': backward_warped_pyramid[0][..., 0:3],
                'forward_residual_flow_pyramid': forward_residual_flow,
                'backward_residual_flow_pyramid': backward_residual_flow,
                'forward_flow_pyramid': forward_flow_pyramid,
                'backward_flow_pyramid': backward_flow_pyramid,
            })
        
        return outputs

        
