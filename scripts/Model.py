"""
dummy temporary model
"""

import numpy as np
import tensorflow as tf

from scripts.Models.UNet import UNet
from scripts.Models.metrics import iou_map


class Model:
    def __init__(self):

        self.model = UNet()
        self.model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=[iou_map], run_eagerly=True)
        self.model.load_weights('/home/raphael/Desktop/Repos/mouse_behaviour_tracker/scripts/Models/best_model_iou_80.h5')

    def predict(self, img):
        img = np.expand_dims(img, axis=0)[:, :, :, 0]
        img = np.expand_dims(img, axis=3)
        im_predicted = self.model.predict(img)

        return im_predicted[0, :, :, 0]
