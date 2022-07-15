import cv2
import numpy as np
from pathlib import Path
from scripts.Models.Dataset import BuildDataset


class BuildDatasetCenter(BuildDataset):
    def __init__(self, *args, **kwargs):
        super(BuildDatasetCenter, self).__init__(*args, **kwargs)

        self.generate_image_and_masks()

    def get_mask(self, cords: list, frame: np.ndarray) -> np.ndarray:
        """this functions generates mask based on annotation"""

        output_y_size, output_x_size, _ = frame.shape
        mask = np.zeros((output_y_size, output_x_size, 1), dtype=np.uint8)

        c = {"x": int(cords[0]), "y": int(cords[1])}

        for i in range(c["x"] - 100, c["x"] + 100):
            for j in range(c["y"] - 100, c["y"] + 100):
                cm_c = 1.5*np.exp(-((i - c["x"]) ** 2 + (j - c["y"]) ** 2) / (2 * self.CFG.sigma ** 2))
                if mask[j, i] < cm_c:
                    mask[j, i] = cm_c
        return mask

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

                cords = self.annotations_dict[video_name]["backbone"][f"{frame_idx}.jpg"]
                mask = self.get_mask(cords, img)

                np.save(str(mask_path / f"{frame_idx}.npy"), mask)
                cv2.imwrite(str(frames_path / f"{frame_idx}.jpg"), img)

                frame_idx += 1
