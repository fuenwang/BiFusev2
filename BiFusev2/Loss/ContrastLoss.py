import os
import sys
sys.path.append('../..')
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .BasePhotometric import BasePhotometric

def ContrastNormalize(rgb, pad=nn.ReplicationPad2d(4), gamma=1e-6):
    assert rgb.size()[1] == 1 or rgb.size()[1] == 3
    I_ori = torch.mean(rgb, dim=1).unsqueeze(1)
    I = pad(I_ori).squeeze(1)
    [bs, h, w] = I.size()
    
    Ixx = []
    for row in range(9):
        for col in range(9):
            p = I[:, row:h-8+row, col:w-8+col].unsqueeze(1)
            Ixx.append(p)

    Ixx = torch.cat(Ixx, dim=1)
    mean = torch.mean(Ixx, dim=1).unsqueeze(1)
    std = torch.std(Ixx, dim=1).unsqueeze(1)
    rgb_out = (rgb - mean) / (std + gamma)

    return rgb_out, mean, std


class ContrastPhotometric(BasePhotometric):
    def __init__(self):
        super().__init__()

    def forward(self, ref_rgb, ref_depth_lst, exp_mask_lst, tgts_rgb, pose):
        loss = 0 
        for i in range(len(ref_depth_lst)):
            scale = 2**i
            ref_depth = ref_depth_lst[i]
            exp_mask = exp_mask_lst[i]
            [_, _, h, w] = ref_depth.shape
            ref_rgb_scaled = F.interpolate(ref_rgb, size=(h, w), mode='area')
            tgts_rgb_scaled = [F.interpolate(x, size=(h, w), mode='area') for x in tgts_rgb]
            warp_frame_lst = self.warp_one_scale(ref_rgb_scaled, ref_depth, tgts_rgb_scaled, pose)
            
            _, mean, std = ContrastNormalize(ref_rgb_scaled)
            assert len(warp_frame_lst) == exp_mask.shape[1]
            for j in range(exp_mask.shape[1]):
                diff = ref_rgb_scaled - warp_frame_lst[j]
                rec_error = (exp_mask[:, j:j+1, ...] * diff.abs() * std).mean()
                loss += rec_error
        
        return loss, None