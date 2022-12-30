import albumentations as A
import cv2
from torchvision import transforms, datasets

import os.path
import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# import model
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self,
                 model,
                 optimizer,
                 criterion,
                 writer,
                 config,
                 dataloaders: dict,
                 scheduler=None,
                 ):

        self.config = config
        self.device = self.config.device
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.dataloaders = dataloaders

        self.writer = writer

        self._global_step = dict()

    def _get_global_step(self, data_type):
        self._global_step[data_type] = -1

    def _epoch_step(self, stage='test', epoch= None):

        if stage not in self._global_step:
            self._get_global_step(stage)

        if stage == 'train':
            self.model.train()
            self._loss_train_step = 0

        else:
            self.model.eval()
            self._loss_eval_step = 0

        len_dataloader = 0
        correct_ep = 0
        for step, (images, targets) in enumerate(self.dataloaders[stage]):
            self._global_step[stage] += 1

            predictions = self.model(images.to(self.device))
            predictions = predictions.reshape(64)

            if stage == 'train':
                self.optimizer.zero_grad()

            loss = self.criterion(predictions, targets.to(self.device))
            # loss2 = tar2
            # loss = loss1 * l + loss2 * (1-l)

            self.writer.add_scalar(f'{stage}/loss', loss, self._global_step[stage])

            if stage == 'train':
                loss.backward()
                self.optimizer.step()

                if self.scheduler is not None:
                    self.scheduler.step()

                self._loss_train_step += loss.item()
                running_loss = self._loss_train_step / (step + 1)
            correct = (torch.argmax(predictions.cpu(), dim=1) == targets.cpu()).sum()
            # correct = (torch.argmax(predictions.cpu(), dim=1) == torch.argmax(targets.cpu(), dim=1)).sum()

            acc = correct / len(targets)
            correct_ep += correct

            # drop last / !drop last
            len_dataloader += len(targets)

            self.writer.add_scalar(f'{stage}/acc', acc, self._global_step[stage])

            if self.config.show_statistics and step % self.config.show_each == 0:
                self._print_overwrite(step, len(self.dataloaders[stage]), loss, acc, stage)

        acc_ep = correct_ep/len_dataloader
        self.writer.add_scalar(f'{stage}/acc_ep', acc_ep, self._global_step[stage])


    def fit(self, epoch_num):
        return self._epoch_step(stage='train', epoch=epoch_num)

    @torch.no_grad()
    def validation(self, i_epoch):
        self._epoch_step(stage='valid', epoch=i_epoch)

    @torch.no_grad()
    def test(self):
        self._epoch_step(stage='test')

    def _print_overwrite(self, step, total_step, loss, acc, stage):
        # sys.stdout.write('\r')
        if stage == 'train':
            print("Train Steps: %d/%d Loss: %.4f Acc: %.4f \n" % (step, total_step, loss, acc))
            # sys.stdout.write("Train Steps: %d/%d Loss: %.4f Acc: %.4f" % (step, total_step, loss, acc))
        else:
            print("Valid Steps: %d/%d Loss: %.4f Acc: %.4f \n" % (step, total_step, loss, acc))
            # sys.stdout.write("Valid Steps: %d/%d Loss: %.4f Acc: %.4f" % (step, total_step, loss, acc))

        sys.stdout.flush()

    def save_model(self, epoch, path=None):
        path = self.config.SAVE_PATH if path is None else path

        if not os.path.exists(path):
            os.makedirs(path)

        path = os.path.join(path, f'{epoch}.pth')

        checkpoint = dict(epoch=self._global_step,
                          model=self.model.state_dict(),
                          optimizer=self.optimizer.state_dict())

        torch.save(checkpoint, path)
        print('model saved, epoch:', epoch)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self._global_step = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('model loaded')