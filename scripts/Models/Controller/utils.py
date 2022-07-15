import numpy as np
import torch
from typing import List, Tuple


def get_xy_of_preds(pred: torch.Tensor) -> List[torch.Tensor]:
    """we need single coordinate so this method return specific coordinate of predicted segment"""
    pred = torch.squeeze(pred, 0)
    coords = [(pred[i] == torch.max(pred[i])).nonzero()[0] for i in range(pred.shape[0])]
    return coords


def rescale_coord_for_orig_img(img: np.ndarray, coords: list, img_size: Tuple[int, int]) -> Tuple[int, int]:
    """predicted image is in specific shape, this method rescaled predictions into original shape"""
    height, width = img_size
    y_scale = img.shape[0] / height
    x_scale = img.shape[1] / width
    coords = (int(coords[0][1]*x_scale), int(coords[0][0]*y_scale))
    return coords


def get_cropped_image(img: np.ndarray, backbone_coordinates: tuple, img_size: tuple) -> Tuple[np.ndarray, int, int]:
    """crop image from original frame based on backbone coordinate"""
    height, width = img_size
    x_center_original, y_center_original = backbone_coordinates
    left_to_center, right_to_center = height // 2, width // 2

    crop_from_y = max(y_center_original - left_to_center, 0)
    crop_from_x = max(x_center_original - right_to_center, 0)
    crop_to_y = y_center_original + right_to_center
    crop_to_x = x_center_original + left_to_center

    cropped_image = img[crop_from_y: crop_to_y, crop_from_x: crop_to_x]

    return cropped_image, crop_from_y, crop_from_x
