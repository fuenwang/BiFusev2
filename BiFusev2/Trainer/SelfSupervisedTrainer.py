import os
import torch
from pytorch_lightning import loggers as pl_loggers
from .BaseTrainer import BaseLitModule
from .. import Loss, Tools

class SelfSupervisedLitModule(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.photo_loss = Loss.ContrastPhotometric()
    
    def training_step(self, batch, batch_idx):
        ref = batch['ref']
        tgts = batch['tgts']
        pose = batch['pose']
        gt_depth = batch['depth']

        pred_depth, pred_mask, pred_pose = self.model(ref, tgts)
        pred_mask = [x.clamp(0, 1) for x in pred_mask]

        rec_loss, _ = self.photo_loss(ref, pred_depth, pred_mask, tgts, pred_pose)
        exp_loss = Loss.explainability_loss(pred_mask)
        smooth_loss = Loss.smooth_loss(pred_depth)
        loss = rec_loss + 0.1 * exp_loss + 0.01 * smooth_loss

        out = {
            'loss': loss,
            'rec-loss': rec_loss,
            'exp-loss': exp_loss,
            'smooth-loss': smooth_loss
        }
        self.write_logger(out, ref, pred_depth, gt_depth)
        
        return out
    
    def validation_step(self, batch, batch_idx):
        idx = batch['idx']
        ref = batch['ref']
        tgts = batch['tgts']
        pose = batch['pose']
        gt_depth = batch['depth']

        pred_depth, pred_mask, pred_pose = self.model(ref, tgts)
        out = [idx.cpu(), pred_depth[0].cpu(), gt_depth.cpu()]
        self.after_validation_step(out)

    def write_logger(self, loss_dict, ref, pred_depth, gt_depth):
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
            if self.global_step % self.config['exp_args']['exp_freq'] == 0:
                rgb = ref.data.cpu()
                pred_depth = Tools.normalizeDepth(pred_depth[0].clamp(0, 10).data.cpu())
                gt_depth = Tools.normalizeDepth(gt_depth.data.cpu())

                self.logger.experiment.add_images('RGB', rgb, self.global_step)
                self.logger.experiment.add_images('Depth/Pred', pred_depth, self.global_step)
                self.logger.experiment.add_images('Depth/GT', gt_depth, self.global_step)
        elif isinstance(self.logger, pl_loggers.WandbLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
            if self.global_step % self.config['exp_args']['exp_freq'] == 0:
                rgb = ref.data.cpu()
                pred_depth = Tools.normalizeDepth(pred_depth[0].clamp(0, 10).data.cpu())
                gt_depth = Tools.normalizeDepth(gt_depth.data.cpu())
                caption = ['RGB-%d'%(self.global_step), 'Pred-%d'%self.global_step, 'GT-%d'%self.global_step]
                self.logger.log_image('Figures', [rgb, pred_depth, gt_depth], caption=caption)
        else:
            raise ValueError('logger type weird')
        