import torch
from pathlib import Path


class CFG:
    model_name       = 'Unet'
    backbone         = 'resnet18'
    img_size         = (512, 512)
    decoder_channels = (256, 128, 64, 32)
    encoder_depth    = 4
    weights          = None
    activation       = None
    in_channels      = 1
    num_classes      = 1
    n_accumulate     = 1
    train_bs         = 16
    valid_bs         = 16
    n_epoch          = 50
    lr               = 0.001

    device           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sigma            = 40  # we use sigma for dot size when generating masks
    model_out_dir    = "/scripts/Models/CenterDetector"
    video_dir        = Path("/home/raphael/Desktop/mouse_data")
    annotations_dir  = Path("/home/raphael/Desktop/mouse_data/annotations")
    videos_name      = ["WIN_20201123_1-1"]
