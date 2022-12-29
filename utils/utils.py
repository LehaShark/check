from icrawler.builtin import GoogleImageCrawler
from configs import CrawlerConfig
import torch

def google_image_downloader():
    config = CrawlerConfig()
    crawler = GoogleImageCrawler(storage={'root_dir': config.download_path})
    crawler.crawl(keyword=config.request_word,
                  max_num=config.count_picture,
                  file_idx_offset='auto'
                  )

def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std