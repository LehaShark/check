from configs import CrawlerConfig
from utils import google_image_downloader

if __name__ == "__main__":
    config = CrawlerConfig()
    google_image_downloader(config)

