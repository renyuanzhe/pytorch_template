import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict

import random
import os
import numpy as np
import torch
from copy import deepcopy
from functools import reduce
import matplotlib.pyplot as plt


def ensure_dir(dirname):#确保目录存在
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)



def set_deterministic(seed): #设置随机种子,保证每次运行结果一致
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#当你设置了相同的种子，那么每次生成的随机数序列都会是一样的
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False # #如果设置为True，那么每次运行程序都会去寻找最适合当前配置的高效算法，这样会产生一定的随机性，可能每次的运行时间都不一样，但是大体上平均下来应该是趋于收敛的。如果设置为False，那么程序每次都会使用同一组算法，如果网络结构不变，那么运行时间也不会变。
    torch.backends.cudnn.deterministic = True ##这句代码是设置PyTorch的随机运算在每次运行时都有相同的结果。这对于确保实验的可重复性是非常有用的。


def send_to_gpu(data_tuple, device, non_blocking=False):
    gpu_tensors = []
    for item in data_tuple:
        gpu_tensors.append(item.to(device, non_blocking=non_blocking))
    return tuple(gpu_tensors)

def load_checkpoint(cfg, model):
    ckpt_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'snapshot')
    if cfg.TEST.test_epoch is None:  # the lastest epoch
        all_ckpts = list(filter(lambda x: x.endswith('.pth'), os.listdir(ckpt_dir)))
        all_epochs = [int(filename.split('.')[-2].split('_')[-1]) for filename in all_ckpts]  # transformer_90.pth
        fids = np.argsort(all_epochs)
        ckpt_file = os.path.join(ckpt_dir, all_ckpts[fids[-1]])
        test_epoch = all_epochs[fids[-1]]
    else:
        assert isinstance(cfg.TEST.test_epoch, int)
        ckpt_file = os.path.join(ckpt_dir, cfg.TRAIN.snapshot_prefix + '%02d.pth'%(cfg.TEST.test_epoch))
        test_epoch = cfg.TEST.test_epoch
    print('Loading the model checkpoint: {}'.format(ckpt_file))
    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model'])
    return model, test_epoch


def save_the_latest(data, ckpt_file, topK=3, ignores=[]):
    """ Only keeping the latest topK checkpoints.
    """
    # find the existing checkpoints in a sorted list
    folder = os.path.dirname(ckpt_file)#获取 ckpt_file 所在的目录
    num_exist = len(os.listdir(folder))#这行代码获取 folder 目录下的文件和子目录的数量。
    if num_exist >= topK + len(ignores):
        # remove the old checkpoints
        ext = ckpt_file.split('.')[-1]
        all_ckpts = list(filter(lambda x: x.endswith('.' + ext), os.listdir(folder)))
        all_epochs = [int(filename.split('.')[-2].split('_')[-1]) for filename in all_ckpts]
        fids = np.argsort(all_epochs)  # transformer_90.pth
        # iteratively remove
        for i in fids[:(num_exist - topK + 1)]:
            if all_epochs[i] in ignores:
                continue
            file_to_remove = os.path.join(folder, all_ckpts[i])
            if os.path.isfile(file_to_remove):
                os.remove(file_to_remove)
    torch.save(data, ckpt_file)






# class MetricTracker:
#     def __init__(self, *keys, writer=None):
#         self.writer = writer
#         self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
#         self.reset()

#     def reset(self):
#         for col in self._data.columns:
#             self._data[col].values[:] = 0

#     def update(self, key, value, n=1):
#         if self.writer is not None:
#             self.writer.add_scalar(key, value)
#         self._data.total[key] += value * n
#         self._data.counts[key] += n
#         self._data.average[key] = self._data.total[key] / self._data.counts[key]

#     def avg(self, key):
#         return self._data.average[key]

#     def result(self):
#         return dict(self._data.average)


# def inf_loop(data_loader):#在一个无限的数据流中不断地获取数据时，这个函数就非常有用。
#     ''' wrapper function for endless data loader. '''
#     for loader in repeat(data_loader):
#         yield from loader

# def prepare_device(n_gpu_use):#设置GPU
#     """
#     setup GPU device if available. get gpu device indices which are used for DataParallel
#     """
#     n_gpu = torch.cuda.device_count()
#     if n_gpu_use > 0 and n_gpu == 0:
#         print("Warning: There\'s no GPU available on this machine,"
#               "training will be performed on CPU.")
#         n_gpu_use = 0
#     if n_gpu_use > n_gpu:
#         print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
#               "available on this machine.")
#         n_gpu_use = n_gpu
#     device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
#     list_ids = list(range(n_gpu_use))
#     return device, list_ids