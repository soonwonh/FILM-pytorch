
import os, argparse, torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam
from tensorboardX import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

import data_lib
import train_lib
from losses import losses
from config import *
from models import interpolator as film_net_interpolator
from models.interpolator import FILM_interpolator
from models import options as film_net_options
from utils import load_checkpoint

def create_model():
    
    options = film_net_options.Options()
    return film_net_interpolator.create_model(options)

def parse_args():
    desc = "Pytorch implementation for FILM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--exp_name', type=str, default='230111', help='experiment name')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint_dir', help='path to save checkpoint')
    parser.add_argument('--log_dir', type=str, default='log_dir', help='path to save tensorboard log')
    parser.add_argument('--train_data', type=str, default='./datasets/vimeo_triplet', help='path to train data')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--epoch', type=int, default=100, help='batch size')
    parser.add_argument('--log_img', type=str, default='log_img', help='path to save image')
    parser.add_argument('--need_patch', type=bool, default=False, help='whether to use patch or full resol. image')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    summary_writer = SummaryWriter(os.path.join(args.log_dir, args.exp_name))

    options = film_net_options.Options()
    model = FILM_interpolator(options).cuda()

    optimizer = Adam(model.parameters(), lr = train_params['learning_rate'], betas=(0.9, 0.999), weight_decay = train_params['weight_decay'])
    
    if args.resume:
        global_step = load_checkpoint(args.resume, model, optimizer)
    
    train_lib.train(
        args,
        model=model,
        summary=summary_writer,
        optimizer=optimizer,
        create_losses_fn=losses.training_losses,
        #create_metrics_fn=metrics_lib.create_metrics_fn,
        dataloader=data_lib.create_training_dataset(args, augmentation_fns=None),
        #eval_datasets=data_lib.create_eval_datasets(),
        #resume=args.resume
    )

