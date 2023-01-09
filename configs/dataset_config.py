import os

# from netlib.module import Module


# class Config(object):
#     def __init__(self):
#         self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#         self.train_key = 'train'
#         self.test_key = 'test'
#
#
# class DataConfig(Config):
#     def __init__(self, dataset: tuple, dataloader: tuple, transform: tuple = None, target_transform: tuple = None):
#         super().__init__()
#         self.dataset = dataset
#         self.dataloader = dataloader
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def get_data(self, transforms_registry, composes_registry, datasets_registry, dataloader_registry):
#
#         def _registry_transforms(transforms):
#             if transforms is None:
#                 return None
#
#             # transform = (Compose:str, Transforms: (list, tuple))
#             if isinstance(transforms[1], (list, tuple)):
#                 tfms = [transforms_registry.get(tfm) for tfm in transforms[1]]
#                 return composes_registry.get((transforms[0], dict(transforms=tfms)))
#             # transform = ('name', dict)
#             return transforms_registry.get(transforms)
#
#         dataset_name, dataset_kwargs = self.dataset
#         dataset_kwargs['path'] = os.path.join(self.ROOT_DIR, dataset_kwargs['path'])
#
#         dataset = datasets_registry.get((dataset_name,
#                                          dict(dataset_kwargs,
#                                               transform=_registry_transforms(self.transform),
#                                               target_transform=_registry_transforms(self.target_transform))))
#
#         dataloader = dataloader_registry.get((self.dataloader[0], dict(self.dataloader[1], dataset=dataset)))
#         return dataset, dataloader
#
#
# class DatasetConfig(Config):
#     def __init__(self, img_shape: tuple, train: DataConfig, valid: DataConfig = None, test: DataConfig = None,
#                  show_dataset: bool = None, show_batch: bool = None, show_each: int = None):
#         super().__init__()
#         self.train = train
#         self.valid = valid
#         self.test = test
#         self.img_shape = img_shape
#         self.show_dataset = show_dataset
#         self.show_batch = show_batch
#         self.show_each = show_each
#
# # import os
# # from configs.base import DataConfig, DatasetConfig
# # from dataloaders.dataloader import SampleType
#
# base_url = "http://yann.lecun.com/exdb/mnist/"
#
# class MNISTConfig(DatasetConfig):
#     def __init__(self, transforms_counter: int = None, with_augs: bool = True):
#         path = os.path.join("data/MNIST/")
#         batch_size = 32
#         img_shape = (28, 28)
#         transform = [('Pad', dict(padding=(2, 2), save_size=True), .5),
#                      ('RandomCrop', dict(size=24, save_size=True), .5),
#                      ('RandomAffine', dict(degrees=(0, 15),
#                                            translate=(0.0, 0.15),
#                                            scale=(0.9, 1.)), .5),
#                      ('GaussianNoise', dict(mean=127, std=255)),
#                      ('Normalize', dict(mean=127, std=255))
#                      ]
#
#         self.num_transform = len(transform)
#         if transforms_counter:
#             transform = transform[-transforms_counter:]
#
#         transform = transform if with_augs else [('Normalize', dict(mean=127, std=255))]
#
#         super().__init__(train=DataConfig(dataset=('MNIST', dict(path=path, is_train=True)),
#                                           transform=('Compose', transform),
#                                           #target_transform=('ToOneHot', dict(num_classes=10)),
#                                           dataloader=('DataLoader', dict(sample_type=SampleType.UPSAMPLE,
#                                                                         # probabilities=[0.5, 0.2, 0.3, 0.7, 0.4, 0.9,
#                                                                         #                0.6, 0.9, 0.8, 0.7],
#                                                                         batch_size=batch_size,
#                                                                         ))),
#                          test=DataConfig(dataset=('MNIST', dict(path=path, is_train=False)),
#                                          transform=('Normalize', dict(mean=127, std=255)),
#                                          dataloader=('DataLoader', dict(sample_type=SampleType.UPSAMPLE,
#                                                                         # probabilities=[0.5, 0.2, 0.3, 0.7, 0.4, 0.9,
#                                                                         #                0.6, 0.9, 0.8, 0.7],
#                                                                         batch_size=batch_size,
#                                                                         ))),
#                          img_shape=img_shape,
#                          show_dataset=True,
#                          show_batch=True,
#                          show_each=100)

class DatasetConfig():
    def __init__(self):
        self.PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
        self.train = os.path.join(self.PATH, 'train')
        self.valid = os.path.join(self.PATH, 'valid')
        self.count_mean_std = True