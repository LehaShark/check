from torchvision.datasets import ImageFolder

import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from configs import DatasetConfig, TrainerConfig
from datasets import ImageLoader
from trainer import Trainer
from nets import resnet50
from transforms import Imposition
from utils import get_mean_std, get_weights
import numpy as np
import albumentations as A
from PIL import Image
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset_config = DatasetConfig()
    trainer_config = TrainerConfig()
    model = resnet50(pretrained=True, num_classes=1)

    keys = train_key, valid_key = 'train', 'valid'

    # dataset = CHECK()

    if dataset_config.count_mean_std:
        dataset = ImageLoader(root=os.path.join(dataset_config.PATH, train_key), transform=A.Compose([ToTensorV2()]))
        mean, std = get_mean_std(dataset)


    # normalize = [transforms.ToTensor(),
    #              transforms.Normalize(mean=mean,
    #                              std=std)]
    normalize = [A.Normalize(mean=mean,
                             std=std),
                 ToTensorV2(),
                 ]

    jitter_param = (0.6, 1.4)

    # train_transform = A.Compose(
    #     [
    #         A.Resize(height=128, width=128),
    #         A.Rotate(),
    #         A.GaussianBlur(sigma_limit=9, p=0.5),
    #         A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #         ToTensorV2(),
    #     ]
    # )

    image_transforms = {train_key: A.Compose([
                                              A.SafeRotate(limit=60),
                                              A.RandomScale(scale_limit=0.4, p=0.5),
                                              A.Resize(height=256, width=256),
                                              Imposition((224, 175)),
                                              # A.RandomCrop(224, 224),
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

    dataloaders_dict = {train_key: DataLoader(datasets_dict[train_key],
                                              batch_size=trainer_config.batch_size, shuffle=True),
                        valid_key: DataLoader(datasets_dict[valid_key],
                                              batch_size=trainer_config.batch_size)}

    if trainer_config.weight_decay is not None:
        w, b = get_weights(model)
        params = [dict(params=w, weight_decay=trainer_config.weight_decay),
                  dict(params=b)]
    else:
        params = model.parameters()

    optimizer = optim.SGD(params, lr=trainer_config.lr)
    criterion = nn.BCELoss()

    writer = SummaryWriter(log_dir=trainer_config.LOG_PATH)

    class_names = datasets_dict[train_key]

    trainer = Trainer(dataloaders=dataloaders_dict,
                      model=model,
                      optimizer=optimizer,
                      criterion=criterion,
                      config=trainer_config,
                      writer=writer)

    # model = trainer.load_model(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs\\exp_1\\40.pth"))

    for epoch in range(trainer_config.epoch_num):

        # trainer.validation(epoch)
        #     trainer.fit(epoch)
        #     trainer.writer.add_scalar(f'scheduler lr', trainer.optimizer.param_groups[0]['lr'], epoch)
        trainer.fit(trainer_config.epoch_num)

        print('\n', '_______', epoch, '_______')
        if epoch % 5 == 0 or epoch == trainer_config.epoch_num - 1:
            trainer.validation(epoch)
            trainer.save_model(epoch, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'))