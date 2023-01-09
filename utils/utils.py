import os

from icrawler.builtin import GoogleImageCrawler
from configs import CrawlerConfig
import torch
from PIL import Image


def google_image_downloader(config: CrawlerConfig = None):
    crawler = GoogleImageCrawler(storage={'root_dir': config.download_path})
    crawler.crawl(keyword=config.request_word,
                  max_num=config.count_picture,
                  file_idx_offset='auto'
                  )


def get_mean_std(dataset):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataset:
        data = data/255
        channels_sum += torch.mean(data, dim=[1, 2])
        channels_squared_sum += torch.mean(data ** 2, dim=[1, 2])
        num_batches += 1

        min_num = torch.min(data)
        if min_num < 0:
            print(min_num)

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def get_weights(model):
    _, weights = [], []
    for name, param in model.named_parameters():
        if name.split('.')[-1] == 'weight':
            weights.append(param)
        else:
            _.append(param)
    return weights, _


def jpg2png(path, dir_name):
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    for num, name in enumerate(filelist):
        im = Image.open(name)
        im.save(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'jpg2png', dir_name + str(num) + '.png'))
