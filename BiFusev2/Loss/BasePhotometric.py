import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .. import Conversion
from .. import Projection

def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

def smooth_loss(pred_map):
    def gradient(pred):
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx)
        dydx, dy2 = gradient(dy)
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss

class BasePhotometric(nn.Module):
    def __init__(self):
        super().__init__()
        self.ET = Conversion.EquirecTransformer(clip=True, mode='torch')
        self.EG = Projection.EquirecGrid()

    def warp_one_scale(self, ref_rgb, ref_depth, tgts_rgb, pose):
        pts = Conversion.homogeneous(self.EG.to_xyz(ref_depth).permute(0, 2, 3, 1))[..., None, :, None]
        nb_frames = len(tgts_rgb)
        Rt = Conversion.pose_vector_to_projection_matrix(pose)[:, None, None, ...]
        pts = (Rt @ pts)[..., 0]
        lonlat = self.ET.xyz2lonlat(pts)
        X = lonlat[..., 0:1].contiguous() / math.pi
        Y = lonlat[..., 1:2].contiguous() / (0.5 * math.pi)
        XY = torch.cat([X, Y], dim=-1)
        assert XY.shape[-1] == 2
        warp_frame_lst = []
        for i in range(nb_frames):
            coord = XY[..., i, :]
            warp_frame = F.grid_sample(tgts_rgb[i], coord, align_corners=True)
            warp_frame_lst.append(warp_frame)
        
        return warp_frame_lst
            

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

            assert len(warp_frame_lst) == exp_mask.shape[1]
            for j in range(exp_mask.shape[1]):
                diff = ref_rgb_scaled - warp_frame_lst[j]
                rec_error = (exp_mask[:, j:j+1, ...] * diff.abs()).mean()
                loss += rec_error
        return loss, None
