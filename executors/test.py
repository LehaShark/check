import os

import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from configs import TrainerConfig, DatasetConfig
from datasets import CHECK
from datasets.base import Dataset
from datasets.dataset import ImageLoader
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision import transforms

from transforms import Imposition
from utils import get_mean_std




def show(im, title=None):
    plt.imshow(im)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # dataset_config = DatasetConfig()
    # trainer_config = TrainerConfig()
    # # model = resnet50(pretrained=True, num_classes=1)
    #
    #
    #
    # keys = train_key, valid_key = 'train', 'valid'
    #
    # # dataset = CHECK()
    #
    # if dataset_config.count_mean_std:
    #     dataset = ImageFolder(root=os.path.join(dataset_config.PATH, train_key), transform=transforms.ToTensor())
    #     mean, std = get_mean_std(dataset)
    # path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'dataset')
    # jitter_param = (0.6, 1.4)
    # normalize = [
    #              # A.Normalize(mean=(0.5, 0.5, 0.5),
    #              #             std=(0.5, 0.5, 0.5)),
    #              A.Normalize(mean=mean,
    #                          std=std),
    #              ToTensorV2()]
    # # normalize = [transforms.ToTensor(),
    # #              transforms.Normalize(mean=mean,
    # #                                   std=std)]
    # # dataset = CHECK(path)
    #
    # train_transform = A.Compose([
    #                              A.SafeRotate(limit=60),
    #                              A.RandomScale(scale_limit=0.4, p=0.5),
    #                              A.Resize(height=256, width=256),
    #                              Imposition((224, 175)),
    #                              # A.RandomRotate90(),
    #                              A.ColorJitter(brightness=jitter_param,
    #                                            saturation=jitter_param,
    #                                            hue=(-.2, .2)),
    #                              A.RandomBrightnessContrast(p=0.2),  # ???
    #                              A.GaussNoise(),
    #                              *normalize
    #                              ])
    #
    # # train_transform = A.Compose([A.SafeRotate(limit=60),
    # #                              A.RandomScale(scale_limit=0.4, p=0.5),
    # #                              A.Resize(height=256, width=256),
    # #                              Imposition((224, 170))])
    #
    # unnormalize = A.Compose([A.Normalize(mean=np.asarray(-mean / std),
    #                                      std=np.asarray(1.0 / std))])
    #
    #
    #
    # im = Image.open(os.path.join(path, 'check_wo_background', 'check_wo_background13.png'))
    # aug_im = train_transform(image=np.asarray(im), need_background=True)["image"]
    # aug_im = aug_im.permute(1, 2, 0)
    #
    # aug_im = (aug_im - aug_im.min()) / (aug_im.max() - aug_im.min())
    # Image.fromarray((np.asarray(aug_im*255)).astype(np.uint8)).save(os.path.join(path, 'norm.png'))
    #
    # unnorm_im = unnormalize(image=np.array(aug_im))["image"]
    # Image.fromarray((np.asarray(unnorm_im)).astype(np.uint8)).save(os.path.join(path, 'unnorm.png'))
    # j = 0

    dataset_config = DatasetConfig()
    trainer_config = TrainerConfig()

    keys = train_key, valid_key = 'train', 'valid'

    if dataset_config.count_mean_std:
        dataset = ImageLoader(root=os.path.join(dataset_config.PATH, train_key), transform=A.Compose([ToTensorV2()]))
        mean, std = get_mean_std(dataset)

    normalize = [
                     A.Normalize(mean=(0.5, 0.5, 0.5),
                                 std=(0.5, 0.5, 0.5)),
                     # A.Normalize(mean=mean,
                     #             std=std),
                     ToTensorV2()]

    jitter_param = (0.6, 1.4)

    image_transforms = {train_key: A.Compose([
        A.SafeRotate(limit=60),
        A.RandomScale(scale_limit=0.4, p=0.5),
        A.Resize(height=256, width=256),
        Imposition((224, 175)),
        # A.RandomRotate90(),
        A.ColorJitter(brightness=jitter_param,
                      saturation=jitter_param,
                      hue=(-.2, .2)),
        A.RandomBrightnessContrast(p=0.2),  # ???
        A.GaussNoise(),
        *normalize
    ]),

        valid_key: A.Compose([A.Resize(height=256, width=256),
                              A.CenterCrop(224, 224),
                              *normalize
                              ])}

    target_transforms = {}

    datasets_dict = {k: ImageLoader(root=os.path.join(dataset_config.PATH, k),
                                    transform=image_transforms[k] if k in image_transforms else None,
                                    target_transform=target_transforms[k] if k in target_transforms else None)
                     for k in keys}

    dataloader = DataLoader(datasets_dict[train_key], batch_size=1, shuffle=True)
    for i in range(10):
        for j, (image, targets) in enumerate(dataloader):
            image = (image - image.min()) / (image.max() - image.min())
            Image.fromarray(np.asarray(image[0]*255).astype('uint8').transpose((1, 2, 0))).save(os.path.join('im', str(i) + str(j) + ' class- ' + str(targets[0].item()) + '.png'))


