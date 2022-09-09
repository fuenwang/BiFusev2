import os
import torch
from .BaseTrainer import BaseLitModule
from pytorch_lightning import loggers as pl_loggers
from .. import Loss

class SupervisedLitModule(BaseLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.berHu = Loss.ReverseHuberLoss()
    
    def training_step(self, batch, batch_idx):
        rgb = batch['rgb']
        gt_depth = batch['depth']
        pred_depth = self.model(rgb)
        berhu_loss = self.berHu(pred_depth[0], gt_depth)

        out = {
            'loss': berhu_loss,
        }
        self.write_logger(out)

        return out
    
    def validation_step(self, batch, batch_idx):
        idx = batch['idx']
        rgb = batch['rgb']
        gt_depth = batch['depth']
        pred_depth = self.model(rgb)[0]

        out = [idx.cpu(), pred_depth.cpu(), gt_depth.cpu()]
        self.after_validation_step(out)
    
    def write_logger(self, loss_dict):
        if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
        elif isinstance(self.logger, pl_loggers.WandbLogger):
            for key, val in loss_dict.items(): self.log('Loss/%s'%key, val)
        else:
            raise ValueError('Logger type weird')
