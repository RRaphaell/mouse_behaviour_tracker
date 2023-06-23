from scripts.Models.UNet import UNet
import torch
import segmentation_models_pytorch as smp
import streamlit as st


@st.cache_resource
def load_model(use_my_model, CFG, pretrained_model_path):
    if use_my_model:
        model = UNet(CFG.in_channels, CFG.num_classes)
    else:
        model = getattr(smp, CFG.model_name)(
            encoder_name=CFG.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=CFG.weights,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=CFG.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=CFG.num_classes,  # model output channels (number of classes in your dataset)
            decoder_channels=CFG.decoder_channels,
            # List of integers which specify in_channels parameter for convolutions used in decoder.
            encoder_depth=CFG.encoder_depth,
            # A number of stages used in encoder.
            # Each stage generate features two times smaller in spatial dimensions than previous one
            activation=CFG.activation,  # An activation function to apply after the final convolution layer
        )

    model.to(CFG.device)
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path, map_location=CFG.device))
    model.eval()
    return model


class ModelBuilder:
    def __init__(self, config, use_my_model=True, pretrained_model_path=''):
        self.CFG = config
        self.use_my_model = use_my_model
        self.pretrained_model_path = pretrained_model_path

    def get_model(self):
        return load_model(self.use_my_model, self.CFG, self.pretrained_model_path)
