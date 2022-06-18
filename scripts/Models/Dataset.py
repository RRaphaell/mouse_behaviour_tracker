import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
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

            points = {part: {} for part in self.CFG.parts}
            for r in root[2:]:
                for child in r.getchildren():
                    label = child.get("label")
                    frame_idx = r.get("id") + ".jpg"
                    points[label][frame_idx] = list(map(int, map(float, child.get("points").split(","))))

            annotations_dict[v] = points
        return videos, images_path, annotations_dict

    def _get_cords(self, video_name, frame_name):
        return self.annotations_dict[video_name]["nose"][frame_name]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        video_name, frame_name = self.images_path[index]
        img = cv2.imread(f"images/{video_name}/{frame_name}.jpg", )
        mask = np.load(f"masks/{video_name}/{frame_name}.npy")

        if self.transforms:
            data = self.transforms(image=img, mask=mask)
            img = data['image']
            msk = data['mask']
        img, msk = torch.tensor(img), torch.tensor(msk)

        img = img.permute(-1, 0, 1)
        msk = msk.permute(-1, 0, 1)
        if self.CFG.in_channels == 1:
            img = torch.unsqueeze(img[0], 0)
        return img, msk

    @staticmethod
    def show_batch_examples(dataset, num_examples, is_parts_detector=False):
        figure = plt.figure(figsize=(10, num_examples * 3))
        cols, rows = 2, num_examples

        for i in range(1, cols * rows + 1, 2):
            sample_idx = torch.randint(len(dataset), size=(1,)).item()
            img, label = dataset[sample_idx]

            ax1 = figure.add_subplot(rows, cols, i)
            ax1.imshow(img[0])
            ax1.set_title("image")

            ax2 = figure.add_subplot(rows, cols, i+1)
            ax2.imshow(torch.mean(label.float(), dim=0))
            ax2.set_title("ground truth")

        plt.tight_layout()
