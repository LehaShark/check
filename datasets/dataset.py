import cv2

from datasets.base import DatasetBase
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, DatasetFolder
import numpy as np

from torch.utils.data import Dataset
class ImageLoader(DatasetFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)

        need_background = False
        # change \\ -> / if linux
        name_item = path.split("\\")[-1][:-4]
        if name_item.find('check_wo_background') != -1:
            # tr = A.Compose([])
            need_background = True


        if self.transform is not None:
            sample = self.transform(image=np.asarray(sample), need_background=need_background)['image']
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #
    #     img1, label1 = super().__getitem__(index)
    #     # take different cls
    #     second_idx = int(np.random.uniform(0, len(self.samples)))
    #
    #     if second_idx == index:
    #         return img1, label1
    #
    #     lmd = round(np.random.beta(1, 1), 4)
    #
    #     img2, label2 = super().__getitem__(second_idx)
    #     # return target1, target2
    #     if isinstance(label1, int) and isinstance(label2, int):
    #         label1 = F.one_hot(torch.tensor(label1), num_classes=len(self.classes))
    #         label2 = F.one_hot(torch.tensor(label2), num_classes=len(self.classes))
    #
    #     mix_img = lmd * img1 + (1 - lmd) * img2
    #     mix_target = lmd * label1 + (1 - lmd) * label2 if torch.any(label1 != label2).item() else label1
    #
    #     # print('ya tut')
    #
    #     return mix_img, mix_target


