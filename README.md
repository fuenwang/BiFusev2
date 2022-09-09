<p align='center'>
<img src='src/teaser.jpg'>
</p>

# BiFuse++
This is the official implementation of our TPAMI paper **"BiFuse++: Self-supervised and Efficient Bi-projection Fusion for 360 Depth Estimation"**. 

### [[Paper](https://arxiv.org/abs/2209.02952)]

Our implementation is based on [Pytorch Lightning](https://www.pytorchlightning.ai/). The following features are included:
1. Multiple GPUs Training (DDP)
2. Multiple Nodes Training (DDP)
3. Supervised Depth Estimation
4. Self-Supervised Depth Estimation
5. Support both [Tensorboard](https://www.tensorflow.org/tensorboard) and [W&B](https://wandb.ai/site) for logging.


## Dependency
Install required packages with the following commands.
```bash
conda create -n bifusev2 python=3.9
conda activate bifusev2
pip install pip --upgrade
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
The installation of pytorch3d will take some time.

## Training and Testing
We provide our training/testing codes for both supervised and self-supervised scenarios. 

For supervised scenario, our model is trained on [Matterport3D](https://niessner.github.io/Matterport/). For self-supervised scenario, we adopt [PanoSUNCG](https://aliensunmin.github.io/project/360-depth/) for training.

1. Although we do not provide Matterport3D dataset, we provide a sample dataset which demonstrates the format adopted by our [SupervisedDataset.py](./BiFusev2/Dataset/SupervisedDataset.py). You can download the sample from [here](https://drive.google.com/file/d/1NA5hWrvPGkMjAuktLu6qw91D8WqJv_6U/view?usp=sharing).
2. For PanoSUNCG, please contact **fulton84717@gapp.nthu.edu.tw** for download links.

To train our approach, please refer to [Experiments](./Experiments) for more details.

## To-Do List
1. Release pretrained weights on Matterport3D and PanoSUNCG.
2. Provide simple inference code.

## License
This work is licensed under MIT License. See [LICENSE](./LICENSE) for details.

If you find our code/models useful, please consider citing our paper:
```bibtex
@article{9874253,
  author={Wang, Fu-En and Yeh, Yu-Hsuan and Tsai, Yi-Hsuan and Chiu, Wei-Chen and Sun, Min},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={BiFuse++: Self-Supervised and Efficient Bi-Projection Fusion for $360^{\circ }$ Depth Estimation}, 
  year={2022},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TPAMI.2022.3203516}
}
```
