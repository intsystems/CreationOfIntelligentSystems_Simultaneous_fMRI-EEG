# Training scripts

This repository contains training code for different parts of our model.

1. To start **contrastive learning** training, run the following:
    ```bash
    accelerate launch train_contrastive.py --config_path=configs/improved-dataloader.yaml   
    ```

2. To start **diffusion prior** training, run the following:
    ```bash
    accelerate launch train_diffusion_prior.py --config_path=configs/diffusion-prior.yaml
    ```