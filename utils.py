import os, torch
import numpy as np
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics import PeakSignalNoiseRatio
from torchvision.utils import save_image
import cv2

def to_gpu(batch):
    batch = {'x0': batch['x0'].cuda(non_blocking=True), 'x1': batch['x1'].cuda(non_blocking=True), 'y': batch['y'].cuda(non_blocking=True), 'time': batch['time'].cuda(non_blocking=True)}
    return batch

def save_checkpoint(args, model, optimizer, step):
    save_dir = os.path.join(args.checkpoint_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {'step': step,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_latest.pt'))
    if step % 10 ==0:
        torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{step}.pt'))

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    for key in list(checkpoint['state_dict'].keys()):
        checkpoint['state_dict'][key.replace('module.','')] = checkpoint['state_dict'].pop(key)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['step']

def metrics(predictions, batch, summary, PSNR, SSIM, global_step):
    psnr = PSNR(predictions['image'], batch['y'])
    ssim = structural_similarity_index_measure(predictions['image'], batch['y'])
    summary.add_scalar('train/psnr', float(psnr), global_step=global_step)
    summary.add_scalar('train/ssim', float(ssim), global_step=global_step)
    return psnr, ssim

def log_image(batch, predictions, args, summary, epoch, i, global_step):
    b, c, h, w = batch['x0'].shape
    img = np.zeros((h, w*4, c), dtype=np.uint8)
    x0 = denorm255(batch['x0'][0])
    prediction = denorm255(predictions['image'][0])
    ground_truth = denorm255(batch['y'][0])
    x1 = denorm255(batch['x1'][0])

    img[:,:w, :] = np.transpose(x0.detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    img[:,w:2*w, :] = np.transpose(prediction.detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    img[:,2*w:3*w, :] = np.transpose(ground_truth.detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    img[:,3*w:4*w, :] = np.transpose(x1.detach().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    summary.add_image('(x0, prediction, ground truth, x1', img[:,:,::-1], global_step=global_step, dataformats='HWC')
    save_dir = os.path.join(args.log_img, args.exp_name)
    os.makedirs(save_dir,exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f'epoch_{epoch}_iter_{i}.png'), img)

def denorm255(x):
    out = (x + 1.0) / 2.0
    return out.clamp_(0.0, 1.0) * 255.0
