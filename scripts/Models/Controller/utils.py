import torch


def get_xy_of_preds(pred):
    """we need single coordinate so this method return specific coordinate of predicted segment"""
    pred = torch.squeeze(pred, 0)
    coords = [(pred[i] == torch.max(pred[i])).nonzero()[0] for i in range(pred.shape[0])]
    return coords


def rescale_coord_for_orig_img(img, coords, img_size):
    """predicted image is in specific shape, this method rescaled predictions into original shape"""
    height, width = img_size
    y_scale = img.shape[0] / height
    x_scale = img.shape[1] / width
    coords = [(int(coord[1]*x_scale), int(coord[0]*y_scale)) for coord in coords]
    return coords


def get_cropped_image(img, backbone_coordinates, img_size):
    """crop image from original frame based on backbone coordinate"""
    height, width = img_size
    x_center_original, y_center_original = backbone_coordinates
    left_to_center, right_to_center = height // 2, width // 2

    crop_from_y = y_center_original - left_to_center if y_center_original > left_to_center else 0
    crop_from_x = x_center_original - right_to_center if x_center_original > right_to_center else 0
    crop_to_y = y_center_original + right_to_center
    crop_to_x = x_center_original + left_to_center

    cropped_image = img[crop_from_y: crop_to_y, crop_from_x: crop_to_x]

    return cropped_image, crop_from_y, crop_from_x
