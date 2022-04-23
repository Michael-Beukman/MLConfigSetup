import bz2
import datetime
import hashlib
import os
import pickle
import random
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import torch
from common.vars import SAVE_EPS, SAVE_PDF

def get_date() -> str:
    """
    Returns the current date in a nice YYYY-MM-DD_H_m_s format
    Returns:
        str
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def mysavefig(*args, **kwargs):
    """
        Saves a figure in a centralised way, also to a PDFs. Usage is just like normal matplotlib.
    """
    dirname = os.path.join(*args[0].split(os.sep)[:-1])
    os.makedirs(dirname, exist_ok=True)
    if 'dpi' not in kwargs: kwargs['dpi'] = 400
    if 'pad_inches' not in kwargs: kwargs['pad_inches'] = 0
    if 'bbox_inches' not in kwargs: kwargs['bbox_inches'] = 'tight'
    
    
    args = list(args)
    plt.savefig(*args, **kwargs)
    
    if SAVE_PDF:
        args[0] = args[0].split(".png")[0] + ".pdf"
        plt.savefig(*args, **kwargs)
    if SAVE_EPS:    
        args[0] = args[0].split(".pdf")[0] + ".eps"
        plt.savefig(*args, **kwargs)


def set_seed(seed: int = 42):
    """
        Sets the seed of numpy, random and torch
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def path(p: str, *args) -> str:
    """Basically os.path.join, but the first argument can be a full path separated with os.sep
    """
    return os.path.join(*(p.split(os.sep) + list(args)))

# https://betterprogramming.pub/load-fast-load-big-with-compressed-pickles-5f311584507e
def save_compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        pickle.dump(data, f)

def load_compressed_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def get_dir(*paths: List[str]) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name

    Returns:
        str: 
    """
    dir = os.path.join(*paths)
    os.makedirs(dir, exist_ok=True)
    return dir

def get_md5sum_file(file_name: str) -> str:
    with open(file_name, 'rb') as file_to_check:
        # read contents of the file
        data = file_to_check.read()    
        # pipe contents of the file through
        return hashlib.md5(data).hexdigest()