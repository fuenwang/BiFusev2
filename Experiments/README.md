# Experiments
We provde the training codes of three experiments.
1. [[supervised_ours_mp3d](./supervised_ours_mp3d)]. The experiments "**BiFuse++**" in Table 1 (Matterport3D) of our main paper, which is trained under supervised scenario.
2. [[selfsupervised_ours_panosuncg_spl](./selfsupervised_ours_panosuncg_spl)]. The experiments "**BiFuse++ w/ SPL**" (PanoSUNCG) in Table 8 of our main paper, which is trained under self-supervised scenario.
3. [[selfsupervised_ours_panosuncg_capl](./selfsupervised_ours_panosuncg_capl)]. The experiments "**BiFuse++ w/ CAPL**" (PanoSUNCG) in Table 8 of our main paper, which is trained under self-supervised scenario.

Before starting training, please modify "**dataset_path**" in **config.yaml** to specify the path of your dataset. After that, you can use the following commands for training and validation.
```bash
python main.py --mode train
python main.py --mode val
```

## Experimental Setting
Our code support multi-gpu and multi-node training. 

* You can change the arguments ("**devices**" and "**num_nodes**") in **config.yaml** to specify the number of GPUs and nodes, respectively.
* To switch between Tensorboard and W&B for logging, you can change "**logger_type**" argument in **config.yaml**.
* For supervised training, we found multi-gpu or larger batch size do not improve the final performance. So the total batch size is 8 in [supervised_ours_mp3d](./supervised_ours_mp3d).
* For self-supervised training, a larger batch size can significantly improve the training stability. Hence, the total batch size of [selfsupervised_ours_panosuncg_spl](./selfsupervised_ours_panosuncg_spl) and [selfsupervised_ours_panosuncg_capl](./selfsupervised_ours_panosuncg_capl) is **8 (bs) x 4 (GPUs) = 32**.

For the provided self-supervised experiments, we use **4x Tesla V100** and train the models for 40 hours. The comparison between SPL and CAPL is summarized in our W&B report ([link](https://wandb.ai/fuenwang/360SfM/reports/CAPL-v-s-SPL--VmlldzoyNTcwNzE4?accessToken=2a5crjt6jghnr28mz4vp0jf0x0vvzkxhzwydbsf4a4klfp3vwhf149roe54817sj)).
