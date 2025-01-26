from pathlib import Path
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys
sys.path.append("../../src")
from eeg_encoder import EEGEncoder
import data_utils

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb


def contrastive_loss(image_features, brain_features, logit_scale) -> dict:
    # Normalize features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    brain_features = brain_features / brain_features.norm(dim=1, keepdim=True)

    # Compute cosine similarity as logits
    logit_scale = logit_scale.exp()
    logits_per_image = logit_scale * image_features @ brain_features.T
    logits_per_brain = logits_per_image.T

    # Compute cross-entropy loss
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=image_features.device, dtype=torch.long)

    total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_brain, labels)) / 2

    return total_loss, logits_per_brain


class LitEggEncoder(L.LightningModule):
    def __init__(self, eeg_encoder: EEGEncoder, logit_scale: nn.Parameter, config: OmegaConf):
        super().__init__()

        self.eeg_encoder = eeg_encoder
        self.optim_config = config.optimizer_kwargs
        self.logit_scale = logit_scale

        self.save_hyperparameters(config)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        subj_ids, eegs, img_latents = batch
        eeg_latents = self.eeg_encoder(eegs, subj_ids)

        loss, logits_per_brain = contrastive_loss(img_latents, eeg_latents, self.logit_scale)

        self.log("Train/CLIP_loss", loss.item(), prog_bar=True, on_step=True)

        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        subj_ids, eegs, img_latents = batch
        eeg_latents = self.eeg_encoder(eegs, subj_ids)

        loss, logits_per_brain = contrastive_loss(img_latents, eeg_latents, self.logit_scale)

        self.log("Test/CLIP_loss", loss.item(), on_epoch=True)

        # calculate rest metrics
        self.log(
            "Test/accuracy",
            logits_per_brain.argmax(dim=1) == torch.arange(logits_per_brain.shape[0]),
            on_epoch=True,
            reduce_fx=lambda x: torch.mean(torch.concat(x))
        )

        prob_per_brain: torch.Tensor = torch.softmax(logits_per_brain)
        self.log(
            "Test/batch_p_true_mean",
            torch.gather(prob_per_brain, dim=0, index=torch.arange(logits_per_brain.shape[0])),
            on_epoch=True,
            reduce_fx=lambda x: torch.mean(torch.concat(x))
        )
        self.log(
            "Test/batch_p_true_std",
            torch.gather(prob_per_brain, dim=0, index=torch.arange(logits_per_brain.shape[0])),
            on_epoch=True,
            reduce_fx=lambda x: torch.std(torch.concat(x))
        )

        self.log(
            "Test/batch_entropy_mean",
            (prob_per_brain * prob_per_brain.log()).sum(dim=1),
            on_epoch=True,
            reduce_fx=lambda x: torch.mean(torch.concat(x))
        )
        self.log(
            "Test/batch_entropy_std",
            (prob_per_brain * prob_per_brain.log()).sum(dim=1),
            on_epoch=True,
            reduce_fx=lambda x: torch.std(torch.concat(x))
        )

    def configure_optimizers(self):
        return optim.Adam(self.eeg_encoder.parameters(), **self.optim_config)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="eeg_sole.yaml",
                        required=False, help="Path to the configuration file.")
    args = parser.parse_args()

    # load config file
    config = OmegaConf.load(args.config_path)

    logger = WandbLogger(
        project="EEG Debug",
        log_model=True
    )

    # load data
    subs_paths = data_utils.parse_subs_eeg_dir(Path(config.subs_dir_path))
    subs_train_paths = {sub_num: path_dict["train"] for sub_num, path_dict in subs_paths.items()}
    subs_test_paths = {sub_num: path_dict["test"] for sub_num, path_dict in subs_paths.items()}

    train_dataset = data_utils.EegImgLatentDataset(
        Path(config.img_latent_train_path),
        subs_train_paths
    )
    test_dataset = data_utils.EegImgLatentDataset(
        Path(config.img_latent_test_path),
        subs_test_paths
    )

    # configure special samplers to assure batch diversity for CLIP-loss
    if config.num_devices == 1:
        train_sampler = data_utils.ClipSampler(train_dataset)
        test_sampler = data_utils.ClipSampler(test_dataset)
    else:
        train_sampler = data_utils.DistributedClipSampler(train_dataset)
        test_sampler = data_utils.DistributedClipSampler(test_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        sampler=test_sampler
    )

    # create lightning model and trainer
    model = LitEggEncoder(
        EEGEncoder(**config.eeg_kwargs),
        nn.Parameter(torch.FloatTensor([config.init_logit_scale])),
        config
    )
    trainer = L.Trainer(
        logger=logger,
        use_distributed_sampler=False,
        accelerator=config.accelerator,
        devices=config.num_devices,
        max_epochs=config.max_epochs
    )

    trainer.fit(model, train_loader, test_loader)
