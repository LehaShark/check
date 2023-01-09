import os
from utils import jpg2png

if __name__ == '__main__':
    dir_name = 'negative'
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'valid', dir_name)
    jpg2png(path, dir_name)