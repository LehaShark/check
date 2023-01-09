import cv2

from datasets.base import DatasetBase
import torch
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.nn import functional as F
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS, DatasetFolder
from collections import defaultdict
import copy
import random
import shutil
from urllib.request import urlretrieve

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from torch.utils.data import Dataset

# class ImageLoader()

# REGISTRY = Registry('datasets')
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

class CHECK(DatasetBase):
    def __init__(self, root_path, transforms=None, target_transforms=None):
        self.root_path = root_path
        self.transforms = transforms
        self.target_transforms = target_transforms
        super().__init__(self.transforms, self.target_transforms)


    def _take_ways(self) -> dict:
        data = dict()
        groups = os.listdir(self.root_path)

        for group in groups:
            data[group] = []

        for root, dirs, files in os.walk(self.root_path):
            reversed_root = ''.join(reversed(root))
            # change \\ -> / if linux
            dir = ''.join(reversed(reversed_root[: reversed_root.find("\\")]))

            for i, file in enumerate(files):
                files[i] = os.path.join(root, file)
            if dir in data:
                data[dir] = files

        return data

    def _aug(self):
        var = self.ways

    def _read_data(self) -> tuple:
        pass

class CheckDataset(Dataset):
    def __init__(self, images_filepaths, transforms=None):
        self.images_filepaths = images_filepaths
        self.transforms = transforms

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return image








# @REGISTRY.register_module
# class MNIST(Dataset):
#     def __init__(self, path, is_train, transform, target_transform):
#         self.path = path
#         self.data_name = f"{'train' if is_train else 't10k'}-images-idx3-ubyte.gz"
#         self.label_name = f"{'train' if is_train else 't10k'}-labels-idx1-ubyte.gz"
#         self.data_path = DataPath(
#             data_path=os.path.join(self.path, self.data_name[:-3]),
#             label_path=os.path.join(self.path, self.label_name[:-3]))
#
#         super().__init__(is_train, transform, target_transform, np.arange(10))
#
#     def _read_data(self):
#         for key, path in self.data_path.items():
#             if not os.path.exists(path):
#                 self._download_dataset()
#                 break
#
#         data = []
#         for key, path in self.data_path.items():
#             with open(path, "rb") as file:
#                 magic_number = int.from_bytes(file.read(4), 'big')
#                 file_data_count = int.from_bytes(file.read(4), 'big')
#                 shape = (file_data_count, int.from_bytes(file.read(4), 'big'), int.from_bytes(file.read(4), 'big')) \
#                     if key == 'input' else file_data_count
#                 file_data = file.read()
#                 data.append(np.frombuffer(file_data, dtype=np.uint8).reshape(shape))
#         file.close()
#         return data[0].astype(np.float32), data[1].astype(np.int32)
#
#
#
# @REGISTRY.register_module
# class CIFAR10(Dataset):
#     def __init__(self, path, is_train, transform=None, target_transform=None):
#         with open(os.path.join(path, 'batches.meta'), 'rb') as file:
#             data = pickle.load(file, encoding='latin1')
#             classes = data['label_names']
#         file.close()
#         self.path = path
#         self._train_list = ('data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5')
#         self._test_list = ('test_batch',)
#
#         super().__init__(is_train, transform, target_transform, classes)
#
#     def _read_data(self) -> tuple:
#         data = []
#         labels = []
#
#         for file_name in (self._train_list if self.is_train else self._test_list):
#             with open(os.path.join(self.path, file_name), 'rb') as file:
#                 entry = pickle.load(file, encoding='latin1')
#                 data.append(entry['data'])
#                 if 'labels' in entry:
#                     labels.extend(entry['labels'])
#                 else:
#                     labels.extend(entry['fine_labels'])
#         file.close()
#
#         return np.squeeze(data).reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1)).astype(np.float32), np.array(labels)