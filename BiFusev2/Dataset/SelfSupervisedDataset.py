from PIL.Image import new
import enum
import os
from shutil import SameFileError
import sys
import cv2
import math
import numpy as np
from imageio import imread
from tqdm import tqdm 
import torch
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
import pytorch3d.transforms.rotation_conversions as p3dr
from .. import Conversion
from .BaseDataset import BaseDataset
from ..Projection import EquirecRotate

def SampleEuler(sample_angle):
    x = (np.random.rand() - 0.5) * sample_angle
    y = (np.random.rand() - 0.5) * sample_angle
    z = (np.random.rand() - 0.5) * sample_angle
    euler = torch.FloatTensor([[x, y, z]])

    return euler

class SelfSupervisedDataset(BaseDataset):
    def __init__(self, dataset_path, mode, frame_interval, shape, augmentation=None, **kwargs):
        assert frame_interval >= 1
        assert os.path.isdir(dataset_path) and mode in ['train', 'val']
        super().__init__(**kwargs)
        with open('%s/rotated/%s.txt'%(dataset_path, mode), 'r') as f: self.scenes = [x.rstrip() for x in f]
        self.dataset_path = dataset_path
        self.shape = shape
        self.augmentation = augmentation
        self.frame_interval = frame_interval
        self.equi_rotate = EquirecRotate(shape[0])
        self.baseInit()

    def _loadFrameLabels(self):
        framelabels = {x: dict() for x in self.scenes}
        for scene in self.scenes:
            tj_lst = sorted(os.listdir('%s/rotated/%s'%(self.dataset_path, scene)))
            for tj in tj_lst:
                tj_path = '%s/rotated/%s/%s'%(self.dataset_path, scene, tj)
                total = len(os.listdir(tj_path)) // 2
                color_lst = ['%s/%d_color.png'%(tj_path, x) for x in range(total)]
                depth_lst = ['%s/%d_depth.png'%(tj_path, x) for x in range(total)]
                pose = self._readTrajectory('%s/labels/%s/%s'%(self.dataset_path, scene, tj))
                framelabels[scene][tj] = {
                    'rgb': color_lst,
                    'depth': depth_lst,
                    'pose': pose
                }
        self.framelabels = framelabels

    def _readTrajectory(self, path):
        tj = np.loadtxt(path)
        view_direction = np.zeros_like(tj)
        view_direction[:-1, ...] = tj[1:, ...] - tj[:-1, ...]
        view_direction[-1, ...] = view_direction[-2, ...]
        with np.errstate(invalid='ignore'): view_direction /= np.linalg.norm(view_direction, axis=-1, keepdims=True)
        position = tj.copy()

        A = np.zeros([view_direction.shape[0], 2, 2], np.float32)
        b = np.zeros([view_direction.shape[0], 2], np.float32)
        b[..., 1] = 1
        A[..., 0, 0] = view_direction[:, 0]
        A[..., 0, 1] = view_direction[:, 2]
        A[..., 1, 0] = view_direction[:, 2]
        A[..., 1, 1] = -view_direction[:, 0]
        A_inv = np.linalg.inv(A)
        ra_rb = (A_inv @ b[..., None])[..., 0]
        
        pose = []
        for i in range(ra_rb.shape[0]):
            ra = ra_rb[i, 0]
            rb = ra_rb[i, 1]
            R = np.array([
                [ra, 0, rb],
                [0,  1,  0],
                [-rb, 0, ra]
            ], np.float32)
            # minus for inverse the axis
            r = cv2.Rodrigues(R)[0][:, 0] if not np.isnan(np.array([ra, rb])).any() else np.array([float('NaN'), float('NaN'), float('NaN')], np.float32)
            t = -R @ position[i, ...] if not np.isnan(np.array([ra, rb])).any() else np.array([float('NaN'), float('NaN'), float('NaN')], np.float32)
            p = np.concatenate([r, t])
            pose.append(p)
        
        return pose
        
    def _createFramePairs(self, frame_interval):
        pairs = []
        for scene, scene_val in self.framelabels.items():
            for tj, tj_val in scene_val.items():
                count = len(tj_val['rgb'])
                rgb_lst = tj_val['rgb']
                depth_lst = tj_val['depth']
                pose = tj_val['pose']
                for i in range(frame_interval, count-frame_interval):
                    if np.isnan(pose[i]).any() or np.isnan(pose[i-frame_interval]).any() or np.isnan(pose[i+frame_interval]).any(): continue
                    p = [relativePose(pose[i], pose[i-frame_interval]), relativePose(pose[i], pose[i+frame_interval])]
                    tmp = {
                        'ref': rgb_lst[i],
                        'tgts': [rgb_lst[i-frame_interval], rgb_lst[i+frame_interval]],
                        'pose': p,
                        'depth': depth_lst[i]
                    }
                    pairs.append(tmp)
        return pairs
    
    def __getitem__(self, idx):
        pair = self.data[idx]
        ref_rgb = readImage(pair['ref'], self.shape)
        tgts_rgb = [readImage(x, self.shape) for x in pair['tgts']]
        ref_depth = readDepth(pair['depth'], self.shape)
        pose = np.concatenate([x[None, ...] for x in pair['pose']], axis=0).astype(np.float32)
        
        if self.augmentation:
            if self.augmentation['flip']:
                raise NotImplementedError
            
            if self.augmentation['rotate']:
                sample_angle = self.augmentation['rotate']['sample_angle'] / 180.0 * math.pi
                euler_ref = SampleEuler(sample_angle)
                euler_R_ref = p3dr.euler_angles_to_matrix(euler_ref, convention='XYZ')
                rgb_tensor = torch.FloatTensor(ref_rgb[None, ...])
                new_ref_rgb = self.equi_rotate(rgb_tensor, rotation_matrix=euler_R_ref.transpose(1, 2))[0, ...].numpy()
                depth_tensor = torch.FloatTensor(ref_depth[None, ...])
                new_ref_depth = self.equi_rotate(depth_tensor, rotation_matrix=euler_R_ref.transpose(1, 2), mode='nearest')[0, ...].numpy()
                euler_R_ref_inv = euler_R_ref.transpose(1, 2)

                new_tgts_rgb = []
                new_pose = []
                for i, tgt in enumerate(tgts_rgb):
                    r = torch.FloatTensor(pose[i:i+1, :3])
                    R = Conversion.angle_axis_to_rotation_matrix(r)
                    t = torch.FloatTensor(pose[i:i+1, 3:])
                    
                    euler_tgt = SampleEuler(sample_angle)
                    euler_R_tgt = p3dr.euler_angles_to_matrix(euler_tgt, convention='XYZ')
                    tgt_tensor = torch.FloatTensor(tgt[None, ...])
                    new_tgt_rgb = self.equi_rotate(tgt_tensor, rotation_matrix=euler_R_tgt.transpose(1, 2))[0, ...].numpy()
                    new_tgts_rgb.append(new_tgt_rgb)

                    R = euler_R_tgt @ R @ euler_R_ref_inv
                    t = (euler_R_tgt @ t[..., None])[..., 0]
                    r = Conversion.rotation_matrix_to_angle_axis(R)
                    rt = torch.cat([r, t], dim=-1)
                    new_pose.append(rt)
                new_pose = torch.cat(new_pose, dim=0)

                ref_rgb = new_ref_rgb
                tgts_rgb = new_tgts_rgb
                pose = new_pose
                ref_depth = new_ref_depth
        out = {
            'idx': idx,
            'ref': ref_rgb,
            'tgts': tgts_rgb,
            'pose': pose,
            'depth': ref_depth
        }

        return out

def readImage(path, shape):
    img = np.asarray(imread(path, pilmode='RGB'), np.float32) / 255.0
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_AREA)

    return img.transpose(2, 0, 1)

def readDepth(path, shape):
    img = np.asarray(imread(path, pilmode='I'), np.float32) / 255.0
    if img.shape[0] != shape[0] or img.shape[1] != shape[1]: img = cv2.resize(img, dsize=tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    img *= 10

    return img[None, ...]

def relativePose(ref, tgt):
    # Set tgt as the origin
    R_ref = cv2.Rodrigues(ref[:3])[0]
    t_ref = ref[3:]

    inv_R_ref = R_ref.T
    inv_t_ref = -inv_R_ref @ t_ref

    R_tgt = cv2.Rodrigues(tgt[:3])[0]
    t_tgt = tgt[3:]

    R = R_tgt @ inv_R_ref
    t = t_tgt + R_tgt @ inv_t_ref
    r = cv2.Rodrigues(R)[0][:, 0]
    p = np.concatenate([r, t])

    return p
