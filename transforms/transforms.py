import math
import os
# from random import random
from typing import List, Dict, Any

import albumentations as A
import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image
from configs import DatasetConfig

config = DatasetConfig()



class Imposition(ImageOnlyTransform):
    def __init__(self, crop_sizes: tuple, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.backgroundspath = os.path.join(config.PATH, 'train', 'negative')

        self.backgroundspaths = []
        for root, dirs, files in os.walk(self.backgroundspath):
            for file in files:
                if file[:-4].find('background') != -1:
                    self.backgroundspaths.append(os.path.join(root, file))

        self.range = abs(crop_sizes[0] - crop_sizes[1])

        self.other_crop = A.RandomCrop(crop_sizes[0], crop_sizes[0])
        self.check_crop = A.RandomCrop(crop_sizes[1], crop_sizes[1])


    def apply(self, img: np.ndarray, matrix: np.ndarray = None, need_background: bool = False, **params) -> np.ndarray:
        if not need_background:
            return self.other_crop(image=img)['image']
        # angle = int(np.random.uniform(0, 360))
        # im = Image.fromarray(img)
        # im_rotate = im.rotate(angle)

        crop_im = self.check_crop(image=img)['image']

        background_idx = int(np.random.uniform(0, len(self.backgroundspaths)))  # if not work add len() - 1 (b not include)
        background = np.asarray(Image.open(self.backgroundspaths[background_idx]).convert("RGB")).copy()

        background = self.other_crop(image=background)['image']

        # range_width = background.size[0] - img.shape[1]  # cv2 0 - height
        # range_height = background.size[1] - img.shape[0]

        x, y = int(np.random.uniform(0, self.range)), int(np.random.uniform(0, self.range))  # -1
        background[x:x + crop_im.shape[0], y:y + crop_im.shape[1]] = crop_im

        return background


    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        super().update_params(params, **kwargs)
        key = 'need_background'
        if key in kwargs:
            params[key] = kwargs[key]
        return params
    # @property
    # def targets_as_params(self) -> List[str]:
    #     return ["image"]
    #
    # def get_params_dependent_on_targets(self, params: Dict[str, Any]) -> Dict[str, Any]:
    #     angle = random.uniform(self.limit[0], self.limit[1])
    #
    #     image = params["image"]
    #     h, w = image.shape[:2]
    #
    #     # https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    #     image_center = (w / 2, h / 2)
    #
    #     # Rotation Matrix
    #     rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #
    #     # rotation calculates the cos and sin, taking absolutes of those.
    #     abs_cos = abs(rotation_mat[0, 0])
    #     abs_sin = abs(rotation_mat[0, 1])
    #
    #     # find the new width and height bounds
    #     new_w = math.ceil(h * abs_sin + w * abs_cos)
    #     new_h = math.ceil(h * abs_cos + w * abs_sin)
    #
    #     scale_x = w / new_w
    #     scale_y = h / new_h
    #
    #     # Shift the image to create padding
    #     rotation_mat[0, 2] += new_w / 2 - image_center[0]
    #     rotation_mat[1, 2] += new_h / 2 - image_center[1]
    #
    #     # Rescale to original size
    #     scale_mat = np.diag(np.ones(3))
    #     scale_mat[0, 0] *= scale_x
    #     scale_mat[1, 1] *= scale_y
    #     _tmp = np.diag(np.ones(3))
    #     _tmp[:2] = rotation_mat
    #     _tmp = scale_mat @ _tmp
    #     rotation_mat = _tmp[:2]
    #
    #     return {"matrix": rotation_mat, "angle": angle, "scale_x": scale_x, "scale_y": scale_y}
    #
    # def get_transform_init_args_names(self) -> Tuple[str, str, str, str, str]:
    #     return ("limit", "interpolation", "border_mode", "value", "mask_value")