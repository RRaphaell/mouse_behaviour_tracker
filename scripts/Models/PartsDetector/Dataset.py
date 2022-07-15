import cv2
import numpy as np
from pathlib import Path
from scripts.Models.Dataset import BuildDataset
from config import CFG


class BuildDatasetParts(BuildDataset):
    def __init__(self, *args, **kwargs):
        super(BuildDatasetParts, self).__init__(*args, **kwargs)

        self.all_parts = CFG.parts
        self.generate_image_and_masks()

    def _crop_image(self, image: np.ndarray, backbone_coordinates: tuple) -> np.ndarray:
        """this functions crops original image based on backbone coordinates
        we need closer image so before apply second model we crop it"""

        x_center_original, y_center_original = backbone_coordinates
        left_to_center, right_to_center = self.CFG.cropping_size[0] // 2, self.CFG.cropping_size[1] // 2

        shape_y = y_center_original - left_to_center if y_center_original > left_to_center else 0
        shape_x = x_center_original - right_to_center if x_center_original > right_to_center else 0

        cropped_image = image[shape_y: (y_center_original + right_to_center),
                              shape_x: (x_center_original + left_to_center), :]

        return cropped_image

    def get_mask(self, coords_dict, frame: np.ndarray) -> np.ndarray:
        """this functions generates mask based on annotation"""

        point_num = len(self.all_parts)
        output_y_size, output_x_size, _ = frame.shape
        new_img = np.zeros((output_y_size, output_x_size, point_num), dtype=np.uint8)
        for idx, part in enumerate(self.all_parts):
            c = coords_dict.get(part, None)
            if c is None:
                continue
            for i in range(int(c[0]) - 20, int(c[0]) + 20):
                for j in range(int(c[1]) - 20, int(c[1]) + 20):
                    cm_c1 = 1.5 * np.exp(-((i - c[0]) ** 2 + (j - c[1]) ** 2) / (2 * self.CFG.sigma ** 2))
                    new_img[j, i, idx] = cm_c1

        return new_img

    def generate_image_and_masks(self, img_out_dir=Path("images"), mask_out_dir=Path("masks")) -> None:
        """this function creates images and masks folders and save them"""

        for video_name in self.CFG.videos_name:
            video = cv2.VideoCapture(str(self.CFG.video_dir / (video_name + ".mp4")))
            frames_path = img_out_dir / video_name
            frames_path.mkdir(parents=True, exist_ok=True)

            mask_path = mask_out_dir / video_name
            mask_path.mkdir(parents=True, exist_ok=True)

            frame_idx = 0
            while True:
                success, img = video.read()
                if not success:
                    break

                frame_name = f"{frame_idx}.jpg"
                all_cords = {p: self.annotations_dict[video_name][p][frame_name] for p in self.all_parts if
                             frame_name in self.annotations_dict[video_name][p]}
                center_cords = self.annotations_dict[video_name]["backbone"][frame_name]
                mask = self.get_mask(all_cords, img)

                mask = self._crop_image(mask, center_cords)
                np.save(str(mask_path / f"{frame_idx}.npy"), mask)

                img = self._crop_image(img, center_cords)
                cv2.imwrite(str(frames_path / f"{frame_idx}.jpg"), img)

                frame_idx += 1
