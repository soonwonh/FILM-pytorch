
from .vgg19_loss import PerceptualLoss
import torch
import numpy as np

from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure as ssim

PSNR = PeakSignalNoiseRatio().cuda()
vgg = PerceptualLoss().cuda()


def vgg_loss(example, prediction):
    return vgg(prediction['image'], example['y'])[0]

def style_loss(example, prediction):
    return vgg(prediction['image'], example['y'])[1]

def perceptual_loss(example, prediction):
    return sum(vgg(prediction['image'], example['y']))

def l1_loss(example, prediction):
    return torch.mean(torch.abs(prediction['image'] - example['y']))

def l1_warped_loss(example, prediction):
    loss = torch.zeros(1, dtype=torch.float32)
    if 'x0_warped' in prediction:
        loss += torch.mean(torch.abs(prediction['x0_warped'] - example['y']))
    if 'x1_warped' in prediction:
        loss += torch.mean(torch.abs(prediction['x1_warped'] - example['y']))
    return loss

def l2_loss(example, prediction):
    return torch.mean(torch.square(prediction['image'] - example['y']))

def ssim_loss(example, prediction):
    return ssim(prediction['image'], example['y']) # to do : max_val=1.0

def psnr_loss(example, prediction):
    return PSNR(prediction['image'], example['y'])

def get_loss(loss_name):
    if loss_name == 'l1':
        return l1_loss
    elif loss_name == 'l2':
        return l2_loss
    elif loss_name == 'ssim':
        return ssim_loss
    elif loss_name == 'vgg':
        return vgg_loss
    elif loss_name == 'style':
        return style_loss
    elif loss_name == 'psnr':
        return psnr_loss
    elif loss_name == 'l1_warped':
        return l1_warped_loss
    elif loss_name == 'perceptual':
        return perceptual_loss
    else:
        raise ValueError('Invalid loss function %s' % loss_name)

def get_loss_op(loss_name):
    loss = get_loss(loss_name)
    return lambda example, prediction: loss(example, prediction)

def get_weight_op(weight_schedule):
    return lambda iterations: weight_schedule(iterations)

def create_losses(loss_names, loss_weight=None):
    losses = dict()
    for name in (loss_names): # to do : loss_weight
        """#unique_values = np.unique(weight_schedule.values)
        #if len(unique_values) == 1: #and unique_values[0] == 1.0:
        #    weighted_name = name
        #else:
        #    weighted_name = 'k*' + name
        #losses[weighted_name] = (get_loss_op(name), get_weight_op(weight_schedule)) # to do
        #print(f"name {str(name)}")"""
        losses[name] = (get_loss_op(name))
    return losses

def training_losses(loss_names, loss_weights=None, loss_weight_schedules=None, loss_weight_parameters=None):
    weight_schedules = [] # to do
    """if not loss_weights:
        for weight_schedule, weight_parameters in zip(loss_weight_schedules, loss_weight_parameters):
            weight_schedules.append(weight_schedule(**weight_parameters))
    else:
        for loss_weight in loss_weights:
            weight_parameters = {
                'boundaries': [0],
                'values': 2 * [loss_weight,]
            }
            weight_schedules.append(torch.optim.lr_scheduler.ConstantLR(optimizer)) # to do : lr parameter"""
    return create_losses(loss_names, weight_schedules)
