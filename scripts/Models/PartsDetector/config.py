import torch
from pathlib import Path


class CFG:
    model_name = 'Unet'
    img_size = (512, 512)

    backbone = 'resnet50'
    weights = 'imagenet'
    decoder_channels = (256, 128, 64, 32, 16)
    encoder_depth = 5
    activation = None

    in_channels = 3
    num_classes = 6

    train_bs = 16
    valid_bs = train_bs * 3
    n_epoch = 150

    lr = 0.001
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = int(400 / train_bs * n_epoch) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sigma = 10  # we use sigma for dot size when generating masks
    model_out_dir = "./"
    video_dir = Path("/kaggle/input/mouse-video")
    annotations_dir = Path("/kaggle/input/mouse-annotations")
    videos_name = ["WIN_20201123_1-1"]

