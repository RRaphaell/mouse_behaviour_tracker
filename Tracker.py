import cv2
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw

from config import KEYPOINT, SEGMENTS


class Tracker:
    def __init__(self, segments_df, img_placeholder):
        self.segments_df = segments_df
        self.img_placeholder = img_placeholder
        self.x = 0

    # dummy function until model is ready
    def _predict_keypoints(self):
        self.x += 5
        return self.x, 100

    def _draw_keypoints(self, draw):
        keypoint_x, keypoint_y = self._predict_keypoints()

        draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                      (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                     outline=KEYPOINT.outline, fill=KEYPOINT.fill)

    def _draw_segments(self, draw):
        if self.segments_df["type"].iloc[0] == "rect":
            draw.rectangle([(self.segments_df["left"], self.segments_df["top"]),
                            (self.segments_df["left"] + self.segments_df["width"],
                             self.segments_df["top"] + self.segments_df["height"])],
                           outline=SEGMENTS.stroke_color, fill=SEGMENTS.fill, width=SEGMENTS.stroke_width)
        else:
            center_x = self.segments_df["left"] + self.segments_df["radius"] * np.cos(np.deg2rad(self.segments_df["angle"]))
            center_y = self.segments_df["top"] + self.segments_df["radius"] * np.sin(np.deg2rad(self.segments_df["angle"]))
            draw.ellipse([(center_x - self.segments_df["radius"], center_y - self.segments_df["radius"]),
                          (center_x + self.segments_df["radius"], center_y + self.segments_df["radius"])],
                         outline=SEGMENTS.stroke_color, fill=SEGMENTS.fill, width=SEGMENTS.stroke_width)

    def draw_predictions(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
        image_pil = image_pil.resize((704, 396))
        draw = ImageDraw.Draw(image_pil, "RGBA")

        self._draw_keypoints(draw)
        self._draw_segments(draw)
        # np.copyto(frame, np.array(image_pil))

        self.img_placeholder.image(image_pil)
