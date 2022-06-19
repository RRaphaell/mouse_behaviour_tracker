import cv2
import torch
from scripts.Models.ModelBuilder import ModelBuilder
from scripts.Models.CenterDetector.config import CFG as CENTER_CFG
from scripts.Models.PartsDetector.config import CFG as PARTS_CFG
from scripts.Models.Controller.config import CFG as CONTROLLER_CFG
import albumentations as A


class Controller:
    """
    This class used to draw segments on video stream, including everything drawn on canvas and all predictions

    Attributes:
        center_model_path (str): pretrained center model path
        parts_model_path (str): pretrained parts model path
        center_transforms (A.Compose): data transformations for center detector model
        parts_transforms (A.Compose): data transformations for parts detector model
        center_detector_model (torch.nn.Module): center detector pytorch model
        parts_detector_model (torch.nn.Module): parts detector pytorch model
        predictions (list[(int, int)]): list of all predicted body part which should track for behaviour analysis
    """

    def __init__(self):
        self.center_model_path = CONTROLLER_CFG.center_model_path
        self.parts_model_path = CONTROLLER_CFG.parts_model_path

        self.center_transforms = A.Compose([
            A.Resize(*CENTER_CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.Normalize()
        ], p=1.0)

        self.parts_transforms = A.Compose([
            A.Normalize(),
            A.PadIfNeeded(PARTS_CFG.img_size[0], PARTS_CFG.img_size[1],
                          position=A.transforms.PadIfNeeded.PositionType.BOTTOM_RIGHT,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0, mask_value=0),
        ], p=1.0)

        self.center_detector_model, self.parts_detector_model = self.build_models()
        self.predictions = []

    def build_models(self):
        """build both center and part detector models.
        use custom unet or segmentation_models_pytorch"""
        model_builder = ModelBuilder(CENTER_CFG, use_my_model=True, pretrained_model_path=self.center_model_path)
        center_detector_model = model_builder.get_model()

        model_builder = ModelBuilder(PARTS_CFG, use_my_model=True, pretrained_model_path=self.parts_model_path)
        parts_detector_model = model_builder.get_model()

        return center_detector_model, parts_detector_model

    def prepare_img_for_center_detector(self, img):
        """image processing before predict center of image"""
        img = self.center_transforms(image=img)["image"]
        img = torch.tensor(img)
        img = img.permute(-1, 0, 1)
        img = torch.unsqueeze(img[0], 0)
        img = img.to(CENTER_CFG.device, dtype=torch.float)
        img = torch.unsqueeze(img, 0)
        return img

    def prepare_img_for_parts_detector(self, img):
        """image processing before predict body parts of image"""
        img = self.parts_transforms(image=img)["image"]
        img = torch.tensor(img)
        img = img.permute(-1, 0, 1)
        img = torch.unsqueeze(img, 0)
        img = img.to(CENTER_CFG.device, dtype=torch.float)
        return img

    def _predict_img(self, img, is_part_detector):
        """run center or part predictions model based on is_part_detector value"""
        with torch.no_grad():
            pred = self.parts_detector_model(img) if is_part_detector else self.center_detector_model(img)
            pred = (torch.nn.Sigmoid()(pred) > 0.5).double()
        return pred.cpu().detach()

    def _get_xy_of_preds(self, pred):
        """we need single coordinate so this method return specific coordinate of predicted segment"""
        pred = torch.squeeze(pred, 0)
        coords = [(pred[i] == torch.max(pred[i])).nonzero()[0] for i in range(pred.shape[0])]
        return coords

    @staticmethod
    def _rescale_coord_for_orig_img(img, coords):
        """predicted image is in specific shape, this method rescaled predictions into original shape"""
        y_scale = img.shape[0] / CENTER_CFG.img_size[0]
        x_scale = img.shape[1] / CENTER_CFG.img_size[1]
        coords = [(int(coord[1]*x_scale), int(coord[0]*y_scale)) for coord in coords]
        return coords

    @staticmethod
    def get_cropped_image(img, backbone_coordinates):
        """crop image from original frame based on backbone coordinate"""
        x_center_original, y_center_original = backbone_coordinates
        left_to_center, right_to_center = CENTER_CFG.img_size[0] // 2, CENTER_CFG.img_size[1] // 2

        crop_from_y = y_center_original - left_to_center if y_center_original > left_to_center else 0
        crop_from_x = x_center_original - right_to_center if x_center_original > right_to_center else 0
        crop_to_y = y_center_original + right_to_center
        crop_to_x = x_center_original + left_to_center

        cropped_image = img[crop_from_y: crop_to_y, crop_from_x: crop_to_x]

        return cropped_image, crop_from_y, crop_from_x

    def predict_img(self, orig_img):
        """run both model. first to find centroid then crop image and run second model
        to predict body parts and return predicted coordinates"""
        # center detector
        img = self.prepare_img_for_center_detector(orig_img)
        center_pred = self._predict_img(img, is_part_detector=False)
        coords = self._get_xy_of_preds(center_pred)
        coords = Controller._rescale_coord_for_orig_img(orig_img, coords)
        center_cropped_image, crop_from_y, crop_from_x = Controller.get_cropped_image(orig_img, coords[0])
        cropped_img_size = center_cropped_image.shape

        # parts detector
        center_cropped_image = self.prepare_img_for_parts_detector(center_cropped_image)
        parts_pred = self._predict_img(center_cropped_image, is_part_detector=True)
        coords = self._get_xy_of_preds(parts_pred)

        # rescale coords to orig image
        coords = [(int(c[1] + crop_from_x - (CENTER_CFG.img_size[1] - cropped_img_size[1])),
                   int(c[0] + crop_from_y - (CENTER_CFG.img_size[0] - cropped_img_size[0]))) for c in coords]

        self.predictions.append(coords[0])

        return coords

    def get_predictions(self):
        return self.predictions
