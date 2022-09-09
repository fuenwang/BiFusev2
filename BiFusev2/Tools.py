import torch
import torch.nn as nn
import random
import numpy as np
import functools
import datetime

def fixSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiGPUs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def normalizeDepth(depth):
    d = depth.clone()
    for i in range(depth.shape[0]):
        d[i ,...] -= d[i ,...].min()
        d[i, ...] /= d[i, ...].max()

    return d

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr): return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def wrap_padding(net, padding):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d) and not isinstance(m, nn.ConvTranspose2d): continue
        [h, w] = m.padding if isinstance(m, nn.Conv2d) else m.output_padding
        assert h == w
        if h == 0: continue
        if isinstance(m, nn.Conv2d): m.padding = (0, 0)
        else: m.output_padding = (0, 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        layer = nn.Sequential(
            padding(h),
            m,
        )
        setattr(root, names[-1], layer)

class MyTqdm:
    def __init__(self, obj, print_step=150, total=None):
        self.obj = iter(obj)
        self.len = len(obj) if total is None else total
        self.print_step = print_step
        self.idx = 0
        self.msg = 'None'

    def __len__(self):
        return self.len

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx == 0: self.start = datetime.datetime.now()
        out = next(self.obj)
        self.idx += 1
        if self.idx % self.print_step == 0 or self.idx == len(self)-1:
            delta = datetime.datetime.now() - self.start
            avg_sec_per_iter = delta.total_seconds() / float(self.idx)

            total_time_pred = datetime.timedelta(seconds=round(avg_sec_per_iter * len(self)))
            delta = datetime.timedelta(seconds=round(delta.total_seconds()))
            if avg_sec_per_iter > 1:
                s = '[%d/%d]  [%.2f s/it]  [%s]  [%s /epoch]'%(self.idx, len(self), avg_sec_per_iter, str(delta), str(total_time_pred))
            else:
                s = '[%d/%d]  [%.2f it/s]  [%s]  [%s /epoch]'%(self.idx, len(self), 1/avg_sec_per_iter, str(delta), str(total_time_pred))
            print (s)
            self.msg = s

        return out

    def getMessage(self):
        return self.msg
