import cv2
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
from scripts.Model import Model
from scripts.config import KEYPOINT, SEGMENTS, CANVAS
from scripts.utils import calculate_circle_center_cords


class Tracker:
    """
    This class used to draw segments on video stream, including everything drawn on canvas and all predictions

    Attributes:
        model (Model): model class for predict mouse keypoint from frame
        segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
        predictions (list): list of all prediction coordinates
    """

    def __init__(self, segments_df, segment_colors):
        """
        initialize tracker class with streamlit widgets and markdowns

        Args:
            segments_df (pd.DataFrame): each row is a segment information such as coordinates, radius etc.
            segment_colors (dict[str, list[float]]): color for each unique segment
        """

        self.model = Model()
        self.segments_df = segments_df
        self.segment_colors = segment_colors
        self.predictions = []

    def _predict_keypoints(self):
        """Save and return predictions from the model"""
        pred_x, pred_y = self.model.predict()
        self.predictions.append((pred_x, pred_y))
        return pred_x, pred_y

    def _draw_keypoints(self, draw):
        """draw model prediction keypoint"""
        keypoint_x, keypoint_y = self._predict_keypoints()

        draw.ellipse([(keypoint_x - KEYPOINT.radius, keypoint_y - KEYPOINT.radius),
                      (keypoint_x + KEYPOINT.radius, keypoint_y + KEYPOINT.radius)],
                     outline=KEYPOINT.outline, fill=KEYPOINT.fill)

    def _draw_segments(self, draw):
        """Draw all segments on the video stream that were drawn on the canvas"""
        if self.segments_df.empty:
            return

        for index, segment in self.segments_df.iterrows():
            color = tuple(self.segment_colors[segment["segment key"]])  # color for each segment

            if segment["type"] == "rect":
                draw.rectangle([(segment["left"], segment["top"]),
                                (segment["left"] + segment["width"], segment["top"] + segment["height"])],
                               outline=SEGMENTS.stroke_color, fill=color, width=SEGMENTS.stroke_width)
            else:
                center_x, center_y = calculate_circle_center_cords(segment)
                draw.ellipse([(center_x - segment["radius"], center_y - segment["radius"]),
                              (center_x + segment["radius"], center_y + segment["radius"])],
                             outline=SEGMENTS.stroke_color, fill=color, width=SEGMENTS.stroke_width)

    def draw_predictions(self, frame):
        """draw all segments and predictions on video stream"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame)
        image_pil = image_pil.resize((CANVAS.width, CANVAS.height))
        draw = ImageDraw.Draw(image_pil, "RGBA")

        predicted_image = self.model.predict(image_pil)
        ind = np.unravel_index(np.argmax(predicted_image, axis=None), predicted_image.shape)
        ind = np.array(ind)
        ind *= 2

        draw.ellipse([(ind[1] - KEYPOINT.radius, ind[0] - KEYPOINT.radius),
                      (ind[1] + KEYPOINT.radius, ind[0] + KEYPOINT.radius)],
                     outline=KEYPOINT.outline, fill=KEYPOINT.fill)

        self._draw_segments(draw)
        self.predictions.append((ind[0], ind[1]))
        return image_pil

    def get_predictions(self):
        return self.predictions
