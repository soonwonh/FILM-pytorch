import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, config):
        super(Fusion, self).__init__()
        
        self.convs = nn.ModuleList()
        self.levels = config.fusion_pyramid_levels # 5
        
        for i in range(config.fusion_pyramid_levels - 1): # (0, 1, 2, 3)
            m = config.specialized_levels # 3
            k = config.filters # 64
            num_filters = (k << i) if i < m else (k << m)
            fusion_in_channels=[128, 256, 512, 970]
            fusion_middle_channels=[138, 330, 714,1482]
            convs = nn.ModuleList()
            convs.append(nn.Conv2d(in_channels = fusion_in_channels[i], out_channels = num_filters, kernel_size=[2, 2], padding='same'))
            convs.append(nn.Conv2d(in_channels = fusion_middle_channels[i], out_channels = num_filters, kernel_size=[3, 3], padding='same'))
            convs.append(nn.Conv2d(in_channels = num_filters, out_channels = num_filters, kernel_size=[3, 3], padding='same'))
            self.convs.append(convs)

        self.output_conv = nn.Conv2d(in_channels = 64, out_channels = 3, kernel_size=1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, pyramid):
        if len(pyramid) != self.levels:
            raise ValueError(
            'Fusion called with different number of pyramid levels '
            f'{len(pyramid)} than it was configured for, {self.levels}.')
        
        net = pyramid[-1]
        for i in reversed(range(0, self.levels-1)):
            level_size = pyramid[i].shape[1:3]
            net = F.interpolate(net, scale_factor=2, mode='nearest')
            net = self.convs[i][0](net)
            net = torch.concat([pyramid[i], net], axis=1)
            net = self.leaky_relu(self.convs[i][1](net))
            net = self.leaky_relu(self.convs[i][2](net))
        net = self.output_conv(net)
        return net
