import numpy as np
import argparse
from imageio import imread
import matplotlib.pyplot as plt

import torch
import BiFusev2


network_args = {
    'save_path': './save',
    'dnet_args': {
        'layers': 34,
        'CE_equi_h': [8, 16, 32, 64, 128, 256, 512]
    },
    'pnet_args': {
        'layers': 18,
        'nb_tgts': 2
    }
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for BiFuse++', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', type=str, required=True, choices=['supervised', 'selfsupervised'], help='Choose supervised of self-supervised model (the architecture is a bit different)')
    parser.add_argument('--ckpt', type=str, required=True, help='Pretrain weights path (.pkl)')
    parser.add_argument('--img', type=str, required=True, help='Input panorama')
    args = parser.parse_args()


    if args.mode == 'supervised': model = BiFusev2.BiFuse.SupervisedCombinedModel(**network_args)
    elif args.mode == 'selfsupervised': model = BiFusev2.BiFuse.SelfSupervisedCombinedModel(**network_args)
    param = torch.load(args.ckpt)
    model.load_state_dict(param, strict=False)
    model = model.cuda()
    model.eval()

    img = imread(args.img, pilmode='RGB').astype(np.float32) / 255.0
    [h, w, _] = img.shape
    assert h == 512 and w == 1024
    batch = torch.FloatTensor(img).permute(2, 0, 1)[None, ...].cuda()
    with torch.no_grad(): depth = model.dnet(batch)[0]
    if args.mode == 'selfsupervised': depth = 1 / (10 * torch.sigmoid(depth) + 0.01)

    depth = depth[0, 0, ...].cpu().numpy().clip(0, 10)

    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(depth)
    plt.show()
