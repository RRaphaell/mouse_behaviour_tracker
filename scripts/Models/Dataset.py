import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
import torch


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, transform, config):

        self.CFG = config
        self.videos, self.images_path, self.annotations_dict = self.process_videos_info()
        self.images_path = self.images_path
        self.transforms = transform
        self.img_size = config.img_size[0]

    def generate_image_and_masks(self, img_out_dir=Path("images"), mask_out_dir=Path("masks")):
        pass
        # will be implemented by child class

    def get_mask(self, cords, frame):
        pass
        # will be implemented by child class

    def process_videos_info(self):
        videos = dict()
        images_path = []
        annotations_dict = dict()

        for v in self.CFG.videos_name:
            # creating videos dict. key: video name value: cv2.VideoCapture
            videos[v] = cv2.VideoCapture(str(self.CFG.video_dir / (v + ".mp4")))

            # accumulating images list. videoname_frameidx
            frame_num = int(videos[v].get(cv2.CAP_PROP_FRAME_COUNT))
            images_path += [(v, f) for f in range(frame_num)]

            # accumulating annotations. key: video_name value: dict(key: body_parths value: dict(key: frame_idx, cords))
            annot_path = self.CFG.annotations_dir / v / "annotations.xml"
            tree = ET.parse(annot_path)
            root = tree.getroot()

            movie_annot = dict()
            for r in root[2:]:
                points = {}
                for child in r.getchildren():
                    points[child.get("frame") + ".jpg"] = list(map(int, map(float, child.get("points").split(","))))
                movie_annot[r.get("label")] = points

            annotations_dict[v] = movie_annot
        return videos, images_path, annotations_dict

    @staticmethod
    def get_video_and_frame_name(s):
        underscore_idx = s.rfind('_')
        video_name = s[:underscore_idx]
        frame_name = s[(underscore_idx + 1):]
        return video_name, frame_name

    def _get_cords(self, video_name, frame_name):
        return self.annotations_dict[video_name]["nose"][frame_name]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        video_name, frame_name = self.images_path[index]
        # video_name, frame_name = BuildDataset.get_video_and_frame_name(img_name)
        img = cv2.imread(f"images/{video_name}/{frame_name}.jpg")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.load(f"masks/{video_name}/{frame_name}.npy")
        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img = data['image']
            msk = data['mask']
        img, msk = torch.tensor(img), torch.tensor(msk)

        img = img.permute(-1, 0, 1)
        msk = msk.permute(-1, 0, 1)

        img = np.expand_dims(img[0], axis=0)
        return img, msk
