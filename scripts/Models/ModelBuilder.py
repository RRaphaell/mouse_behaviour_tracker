from scripts.Models.UNet import UNet
import torch
import segmentation_models_pytorch as smp
import streamlit as st


class ModelBuilder:
    def __init__(self, config, use_my_model=True, pretrained_model_path=''):
        self.CFG = config
        self.use_my_model = use_my_model
        self.pretrained_model_path = pretrained_model_path

    def build_model(self):
        if self.use_my_model:
            model = UNet(self.CFG.in_channels, self.CFG.num_classes)
        else:
            model = getattr(smp, self.CFG.model_name)(
                encoder_name=self.CFG.backbone,    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=self.CFG.weights,  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.CFG.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.CFG.num_classes,      # model output channels (number of classes in your dataset)
                decoder_channels=self.CFG.decoder_channels,
                # List of integers which specify in_channels parameter for convolutions used in decoder.
                encoder_depth=self.CFG.encoder_depth,
                # A number of stages used in encoder.
                # Each stage generate features two times smaller in spatial dimensions than previous one
                activation=self.CFG.activation,    # An activation function to apply after the final convolution layer
            )

        model.to(self.CFG.device)
        return model

    @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True)
    def load_model(self):
        model = self.build_model()
        model.load_state_dict(torch.load(self.pretrained_model_path, map_location=self.CFG.device))
        model.eval()
        return model

    def get_model(self):
        if self.pretrained_model_path:
            return self.load_model()
        else:
            return self.build_model()
