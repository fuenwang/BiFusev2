"""
Implementation of custom loss functions.

Notes:
- [TODO] masked L2 loss for sparse ground truth.
- [TODO] masked BerHu loss for sparse ground truth.
- [TODO] add docstring to CrossEntropyLoss.

Last update: 2018/11/05 by Johnson
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#from .depth_level import *


def _assert_no_grad(tensor):
    """ To make sure tensor of ground truth cannot backprop """
    assert not tensor.requires_grad, \
               'nn criterions don\'t compute the gradient w.r.t. targets - please ' \
               'mark these tensors as not requiring gradients'


class L1Loss(nn.Module):
    """ L1 loss but mask out invalid points (value <= 0). """
    def __init__(self, dummy):
        super(L1Loss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach() #torch.Size([8, 1, 228, 304])
        diff = (targets - inputs)[valid_mask] #torch.Size([537908])
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = diff.abs().mean() #torch.Size([])
        import pdb;pdb.set_trace()
        return loss


class L2Loss(nn.Module):
    """ L2 loss but mask out invalid points (value <= 0). """
    def __init__(self, dummy):
        super(L2Loss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach()
        diff = (targets - inputs)[valid_mask]
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = diff.pow(2).mean()

        return loss

class HuberLoss(nn.Module):
    """Huber Loss but mask out invalid points (value <= 0). """
    def __init__(self, dummy):
        super(HuberLoss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach() #Pooling if >0 => True => 1 
        diff = (targets - inputs)[valid_mask]
        loss_fn = nn.SmoothL1Loss(reduce = False, size_average = False)
        if diff.shape[0] == 0: # make sure the case that depth are all masked out
            loss = (targets - inputs).mean() * 0
        else:
            loss = loss_fn(inputs, targets)

        return loss
class Berhu_uncertainty(nn.Module):
    def __init__(self):
        super(Berhu_uncertainty, self).__init__()
    def forward(self, inputs, targets, un):
        _assert_no_grad(targets) #cannot backprop the gradient
        valid_mask = (targets > 0).detach().float()
        diff = (targets - inputs) * valid_mask
        adiff = torch.abs(targets - inputs) * valid_mask
        MAX = adiff.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0] 
        c = 0.2 * MAX
        sqdiff = (diff * diff + c * c) / (2 * c + 1e-6 )
        b_loss1 = adiff * (adiff <= c).cuda().float()
        b_loss2 = sqdiff * (adiff > c).cuda().float()
        b_loss = b_loss1 + b_loss2
        #print(b_loss.max(), b_loss.min())
        #import pdb;pdb.set_trace()
        u_loss1 = torch.exp(-(un.clamp(min=-5))) * b_loss
        u_loss2 = un
        u_loss = ((0.5 * u_loss1 + 0.5 * u_loss2) * valid_mask).sum() / valid_mask.sum()
        return u_loss, (u_loss1*valid_mask).sum()/valid_mask.sum(), (u_loss2[valid_mask.byte().detach()]).mean(), (b_loss[valid_mask.byte().detach()]).mean()

class Berhu_un_2branch(nn.Module):
    def __init__(self):
        super(Berhu_un_2branch,self).__init__()
    def forward(self, d_equi, d_c2e, targets, equi_un, c2e_un):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach().float()
        diff_e = (targets - d_equi) * valid_mask
        adiff_e = torch.abs(targets - d_equi) * valid_mask
        MAX_e = adiff_e.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0]
        c_e = 0.2 * MAX_e
        
        diff_c = (targets - d_c2e) * valid_mask
        adiff_c = torch.abs(targets - d_c2e) * valid_mask
        MAX_c = adiff_c.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0]
        c_c = 0.2 * MAX_c

        sqdiff_e = (diff_e * diff_e + c_e * c_e) / (2 * c_e + 1e-6 )
        b_loss1_e = adiff_e * (adiff_e <= c_e).cuda().float()
        b_loss2_e = sqdiff_e * (adiff_e > c_e).cuda().float()
        b_loss_e = b_loss1_e + b_loss2_e

        sqdiff_c = (diff_c * diff_c + c_c * c_c) / (2 * c_c + 1e-6)
        b_loss1_c = adiff_c * (adiff_c <= c_c).cuda().float()
        b_loss2_c = sqdiff_c * (adiff_c > c_c).cuda().float()
        b_loss_c = b_loss1_c + b_loss2_c 

        merge_b = b_loss_e.clone()
        merge_b[equi_un > c2e_un] = b_loss_c[equi_un > c2e_un]

        #method_1
        merge_un = equi_un.clone()
        merge_un[equi_un > c2e_un] = c2e_un[equi_un > c2e_un]
        merge_un_bad = equi_un.clone()
        merge_un_bad[equi_un < c2e_un] = c2e_un[equi_un < c2e_un]
        #method_2
        #merge_mask = torch.where(equi_un < cube_un, equi_un, cube_un)
        
        u_loss1 = torch.exp(-(merge_un.clamp(min=-5))) * merge_b
        u_loss2 = merge_un
        u_loss3 = merge_un_bad
        u_loss = ((0.5 * u_loss1 + 0.5 * u_loss2 + 0.05 * u_loss3) * valid_mask).sum() / valid_mask.sum()
        return u_loss, (u_loss1*valid_mask).sum()/valid_mask.sum(),(u_loss2[valid_mask.byte().detach()]).mean(), (merge_b[valid_mask.byte().detach()]).mean()

#NCHW
class ReverseHuberLoss(nn.Module):
    def __init__(self, dummy=None):
        super(ReverseHuberLoss, self).__init__()

    def forward(self, inputs, targets):
        _assert_no_grad(targets) #cannot backprop the gradient
        valid_mask = (targets > 0).detach() #Pooling if >0 => True => 1    torch.Size([8, 1, 228, 304])
        valid_mask = valid_mask.float() #uint8 to float32 torch.Size([8, 1, 228, 304])
        diff = (targets - inputs) * valid_mask #torch.Size([8, 1, 228, 304])
        adiff = torch.abs(targets - inputs) * valid_mask #torch.Size([8, 1, 228, 304])
        MAX = adiff.max(dim = -1,keepdim = True)[0].max(dim = -2,keepdim = True)[0] #torch.Size([8, 1, 1, 1])
        c = 0.2 * MAX #torch.Size([8, 1, 1, 1])
        sqdiff = (diff * diff + c * c) / (2 * c + 1e-6 )  #torch.Size([8, 1, 228, 304])
        loss1 = adiff * (adiff <= c).cuda().float() #torch.Size([8, 1, 228, 304])
        loss2 = sqdiff * (adiff > c).cuda().float() #torch.Size([8, 1, 228, 304])
        loss = ((loss1 + loss2)  / (valid_mask.sum() + 1e-6)).sum()
        
        return loss

class CrossEntropyLoss(nn.Module):
    """ Cross entropy loss with logits and regression targets. """
    def __init__(self, depth_level, min_depth, max_depth, n_D):
        super(CrossEntropyLoss, self).__init__()
        # Setup depth level
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_D = n_D
        if depth_level == 'space_increasing':
            self.depth_level = space_increasing_depth_level(self.min_depth,
                                                            self.max_depth,
                                                            self.n_D)
        else:
            raise ValueError('Invalid depth level {}'.format(self.depth_level))
        self.depth_level = np.array(self.depth_level)

        # Setup criterion
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        _assert_no_grad(targets)
        valid_mask = (targets > 0).detach().squeeze(1)

        # Change regression targets to classification
        tiled_targets = targets.repeat(1, self.n_D, 1, 1)
        broadcast_d_1 = torch.Tensor(self.depth_level[np.newaxis, :, 0:1, np.newaxis]).to(targets)
        broadcast_d_2 = torch.Tensor(self.depth_level[np.newaxis, :, 1:2, np.newaxis]).to(targets)
        targets_cls = (tiled_targets >= broadcast_d_1) * (tiled_targets < broadcast_d_2)
        _, targets_cls = targets_cls.max(1)

        # Compute loss
        loss = self.criterion(inputs, targets_cls.long())
        loss = loss[valid_mask].mean()

        return loss


class NStageLoss(nn.Module):
    """ N-stage (multi-scale) loss. Target is only the one in the last stage. """
    def __init__(self, n_stages, interp, weights, criterion):
        super(NStageLoss, self).__init__()
        assert len(weights) == n_stages, 'The length of weights should be the \
                                          same as number of stages.'
        self.n_stages = n_stages
        self.interp = interp
        self.weights = weights
        if criterion == 'L1Loss':
            self.criterion = L1Loss(None)
        elif criterion == 'L2Loss':
            self.criterion = L2Loss(None)
        else:
            raise ValueError('Invalid criterion {}'.format(criterion))

    def forward(self, inputs, targets):
        _assert_no_grad(targets)

        total_loss = 0.
        for i in range(self.n_stages):
            input_i = inputs[i]
            if i == (self.n_stages - 1):
                target_i = targets
            else:
                target_i = F.interpolate(targets, input_i.shape[2:], mode=self.interp)
            loss = self.criterion(input_i, target_i)
            total_loss += self.weights[i] * loss
            print(loss)

        return total_loss


class NStageCrossEntropyLoss(nn.Module):
    """ N-stage (multi-scale) Cross Entropy loss. Target is only the one in the last stage. """
    def __init__(self, n_stages, interp, weights, depth_level, min_depth, max_depth, n_D):
        super(NStageCrossEntropyLoss, self).__init__()
        assert len(weights) == n_stages, 'The length of weights should be the \
                                          same as number of stages.'
        self.n_stages = n_stages
        self.interp = interp
        self.weights = weights
        self.criterion = CrossEntropyLoss(depth_level, min_depth, max_depth, n_D)

    def forward(self, inputs, targets):
        _assert_no_grad(targets)

        total_loss = 0.
        for i in range(self.n_stages):
            input_i = inputs[i]
            if i == (self.n_stages - 1):
                target_i = targets
            else:
                target_i = F.interpolate(targets, input_i.shape[2:], mode=self.interp)
            loss = self.criterion(input_i, target_i)
            total_loss += self.weights[i] * loss
            print(loss)

        return total_loss
