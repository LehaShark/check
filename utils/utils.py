from icrawler.builtin import GoogleImageCrawler
from configs import CrawlerConfig

def google_image_downloader():
    config = CrawlerConfig()
    crawler = GoogleImageCrawler(storage={'root_dir': config.download_path})
    crawler.crawl(keyword=config.request_word,
                  max_num=config.count_picture,
                  file_idx_offset='auto'
                  )