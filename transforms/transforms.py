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
    def __init__(self, crop_sizes: tuple, always_apply: bool = False, p: float = 1):
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