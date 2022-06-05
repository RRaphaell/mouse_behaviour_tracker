
class Controller:
    def __init__(self, orig_image_size, desire_image_size):
        self.original_image = orig_image_size
        self.desired_image_shape = desire_image_size

    def get_cropped_image(self, image, backbone_coordinates):
        x_center_original = backbone_coordinates[0]
        y_center_original = backbone_coordinates[1]
        left_to_center    = self.desired_image_shape[0] // 2
        right_to_center   = self.desired_image_shape[1] // 2

        shape_y = y_center_original - left_to_center if y_center_original > left_to_center else 0
        shape_x = x_center_original - right_to_center if x_center_original > right_to_center else 0

        cropped_image = image[shape_y: (y_center_original + right_to_center),
                              shape_x: (x_center_original + left_to_center)]

        return cropped_image

    def get_cropped_image_mask(self):
        pass

    def add_cropped_mask_to_orig(self):
        pass

