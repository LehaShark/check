import os

import torch
from configs.dataset_config import DatasetConfig

class TrainerConfig:
    def __init__(self):
        self.epoch_num = 75
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.batch_size = 128

        self.show_statistics = True
        self.device = torch.device('cuda')
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = 'CrossEntropyLoss'
        self.optim = 'SGD'
        self.momentum = 0.9
        self.show_each = 1

        self.label_smoothing = 5e-3

        self.LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')