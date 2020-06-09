import os
import os.path as osp
import cPickle
import json
import errno
import numpy as np
import torch


def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    _obj = obj.copy()
    for k, v in _obj.items():
        if isinstance(v, np.ndarray):
            _obj.pop(k) 
    with open(fpath, 'w') as f:
        json.dump(_obj, f, indent=4, separators=(',', ': '))

def save_checkpoint(state, filename):
    torch.save(state, filename)
