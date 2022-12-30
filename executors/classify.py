from torchvision.datasets import ImageFolder

import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter

from configs import DatasetConfig, TrainerConfig
from trainer import Trainer
# from torchvision.models import resnet50, ResNet50_Weights
from nets import resnet50
from utils import get_mean_std, get_weights

if __name__ == '__main__':

    dataset_config = DatasetConfig()
    trainer_config = TrainerConfig()
    model = resnet50(pretrained=True, num_classes=1)

    keys = train_key, valid_key = 'train', 'valid'

    if dataset_config.count_mean_std:
        dataset = ImageFolder(root=os.path.join(dataset_config.PATH, train_key), transform=transforms.ToTensor())
        mean, std = get_mean_std(dataset)


    normalize = [transforms.ToTensor(),
                 transforms.Normalize(mean=mean,
                                 std=std)]

    jitter_param = (0.6, 1.4)

    image_transforms = {train_key: transforms.Compose([transforms.RandomResizedCrop(224),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.ColorJitter(brightness=jitter_param,
                                                                              saturation=jitter_param,
                                                                              hue=(-.2, .2)),
                                                       *normalize]),

                        valid_key: transforms.Compose([transforms.Resize(256),
                                                       transforms.CenterCrop(224),
                                                       *normalize])}
    target_transforms = {}

    datasets_dict = {k: ImageFolder(root=os.path.join(dataset_config.PATH, k),
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

    optimizer = optim.SGD(params, lr=trainer_config.lr, momentum=trainer_config.momentum)
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
        if epoch % 4 == 0:
            trainer.validation(epoch)
            trainer.save_model(epoch, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'))