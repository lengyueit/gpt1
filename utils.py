import time

from config import parser
import numpy as np
import random
import torch
import os
import logging

arg = parser.parse_args()


def read_data(path, num=None):
    with open(path, encoding="utf8") as f:
        all_data = f.read().split("\n\n")

    if num:
        return all_data[:num]
    else:
        return all_data[:-1]


def get_word_2_index(path):
    with open(path, encoding="utf8") as f:
        index_2_word = f.read().split("\n")

    word_2_index = {w: i for i, w in enumerate(index_2_word)}

    return index_2_word, word_2_index


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_logger(args):
    """init logger"""
    current_path = os.path.join(args.log_dir, f"{str(time.asctime())}-{str(args.lr)}-{str(args.epoch)}-output.log")

    if os.path.exists(current_path):
        os.remove(current_path)
    logging.basicConfig(filename=current_path
                        , format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
