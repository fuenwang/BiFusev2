import yaml
import argparse
import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import BiFusev2


class MM(BiFusev2.Trainer.SupervisedLitModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def main(args, config):
    BiFusev2.Tools.fixSeed(config['exp_args']['seed'])
    model = BiFusev2.BiFuse.SupervisedCombinedModel(**config['network_args'])
    model.Load(args.epoch)
    litmodule = MM(config, model)
    BiFusev2.Trainer.ScriptStart(args, config, litmodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for BiFuse++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', default='train', type=str, choices=['train', 'val'], help='train/val mode')
    parser.add_argument('--epoch', type=int, default=None, help='load epoch')
    args = parser.parse_args()

    with open('./config.yaml', 'r') as f: config = yaml.load(f, Loader=yaml.FullLoader)
    main(args, config)
