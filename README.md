# FILM-pytorch
- PyTorch Implementation of FILM: Frame Interpolation for Large Motion (https://film-net.github.io/)
- Easy to use, read, debug than original TF code
- It shows comparable performances as reported by original paper (PSNR ~ 34 on Vimeo90K)
- Tensorboard logging for metrics(PSNR, SSIM), generated images (x0, prediction, ground truth, x1)

## Requirements
- Python 3.11.0
- Pytorch 1.13.1
- CUDA 11.6

## Installation

*  Option 1) Copy created conda environment
```
git clone https://github.com/google-research/frame-interpolation
cd frame-interpolation
conda env create -f film-pytorch.yaml
conda activate film-pytorch
```

*  Option 2) Install requirements yourself
```
git clone https://github.com/google-research/frame-interpolation
cd frame-interpolation
conda env create -n film-pytorch
conda activate film-pytorch
conda install -c conda-forge python
pip install torch==1.13.1+cu116 torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install scipy torchmetrics tensorboardX opencv-python tqdm
```

## Training
- It accepts Vimeo-like data directory for train, you need to pass argument for --train_data

```
film-pytorch
└── datasets
      └── vimeo_triplet
          ├──  sequences
          readme.txt
          tri_testlist.txt
          tri_trainlist.txt
```
- for training, you can specify train_data path, batch_size, epoch, resume(for loading checkpoint), exp_name
```
python train.py --train_data datasets/vimeo_triplet --exp_name 230115_exp1 --batch_size 8 --epoch 100 --resume 'path to checkpoint'
```

## Inference
to be released very soon (in progress)
*  1) One mid-frame interpolation

To generate an intermediate photo from the input near-duplicate photos, simply run:

```
python inference.py --frame1 data/one.png --frame2 data/two.png --model_path pretrained/film_style --output temp/output.png
```

This will produce the sub-frame at `t=0.5` and save as 'photos/output_middle.png'.

*  2) Many in-between frames interpolation

It takes in a set of directories identified by a glob (--pattern). Each directory
is expected to contain at least two input frames, with each contiguous frame
pair treated as an input to generate in-between frames. Frames should be named such that when sorted (naturally) with `natsort`, their desired order is unchanged.


```
python inference.py --data "photos" --model_path pretrained/film_style --times_to_interpolate 3 --output_video
```

You will find the interpolated frames (including the input frames) in
'photos/interpolated_frames/', and the interpolated video at
'photos/interpolated.mp4'.
