import torch.nn as nn

class Options(nn.Module):
    def __init__(self, pyramid_levels=7, fusion_pyramid_levels=5, specialized_levels=3, flow_convs=[3, 3, 3, 3], fusion_in_channels=[32, 64, 128, 256, 512], flow_filters=[32, 64, 128, 256], sub_levels=4, filters=64, use_aux_outputs=True):
        super(Options, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels
        self.specialized_levels = specialized_levels
        self.flow_convs = flow_convs #or [4, 4, 4, 4]
        self.flow_filters = flow_filters #or [64, 128, 256, 256]
        self.sub_levels = sub_levels
        self.filters = filters
        self.use_aux_outputs = use_aux_outputs
        self.fusion_in_channels = fusion_in_channels

