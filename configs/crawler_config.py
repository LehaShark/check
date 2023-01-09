import os

class CrawlerConfig():
    def __init__(self):
        self.request_word = "стены"
        self.count_picture = 1000
        self.download_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'downloads')