from pathlib import Path
import argparse
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import sys

import torchmetrics.aggregation
import torchmetrics.classification
sys.path.append("../../src")
from eeg_encoder import EEGEncoder
import data_utils

import lightning as L
import torchmetrics
from lightning.pytorch.loggers import WandbLogger

import wandb

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*The given NumPy array is not writable")
warnings.filterwarnings("ignore", module="lightning", message="The .* does not have many workers")


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
    labels = torch.arange(batch_size).to(logits_per_image).to(dtype=torch.long)

    total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_brain, labels)) / 2

    return total_loss, logits_per_brain


class LitEggEncoder(L.LightningModule):
    def __init__(self, eeg_encoder: EEGEncoder, logit_scale: nn.Parameter, config: OmegaConf):
        super().__init__()

        self.eeg_encoder = eeg_encoder
        self.optim_config = config.optimizer_kwargs
        self.logit_scale = logit_scale

        self.batch_p_true_mean_metric = torchmetrics.MeanMetric()
        # used to compute std
        self.batch_p_true_mean_sq_metric = torchmetrics.MeanMetric()

        self.batch_entropy_mean_metric = torchmetrics.MeanMetric()
        # used to compute std
        self.batch_entropy_mean_sq_metric = torchmetrics.MeanMetric()

        self.overall_accuracy = torchmetrics.classification.MulticlassAccuracy(
            num_classes=config.batch_size,
            average="micro"
        )

        self.save_hyperparameters(config)

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        subj_ids, eegs, img_latents = batch
        eeg_latents = self.eeg_encoder(eegs, subj_ids)

        loss, logits_per_brain = contrastive_loss(img_latents, eeg_latents, self.logit_scale)

        self.log("Train/CLIP_loss", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int):
        subj_ids, eegs, img_latents = batch
        eeg_latents = self.eeg_encoder(eegs, subj_ids)

        loss, logits_per_brain = contrastive_loss(img_latents, eeg_latents, self.logit_scale)

        self.log("Test/CLIP_loss", loss.item(), on_epoch=True)

        # calculate rest metrics

        self.overall_accuracy.update(
            logits_per_brain.argmax(dim=1),
            torch.arange(logits_per_brain.shape[0]).to(device=logits_per_brain.device)
        )

        prob_per_brain: torch.Tensor = torch.softmax(logits_per_brain, dim=1)

        self.batch_p_true_mean_metric.update(
            torch.gather(prob_per_brain, dim=0, index=torch.arange(logits_per_brain.shape[0]).reshape(1, -1).to(device=logits_per_brain.device))
        )
        self.batch_p_true_mean_sq_metric.update(
            torch.gather(prob_per_brain, dim=0, index=torch.arange(logits_per_brain.shape[0]).reshape(1, -1).to(device=logits_per_brain.device)) ** 2
        )

        self.batch_entropy_mean_metric.update(
            (prob_per_brain * prob_per_brain.log()).sum(dim=1)
        )
        self.batch_entropy_mean_sq_metric.update(
            (prob_per_brain * prob_per_brain.log()).sum(dim=1) ** 2
        )

    def on_validation_epoch_end(self):
        # log accuracy
        self.log(
            "Test/accuracy",
            self.overall_accuracy.compute()
        )
        self.overall_accuracy.reset()

        # log p_true metrics
        self.log(
            "Test/batch_p_true_mean",
            self.batch_p_true_mean_metric.compute()
        )
        self.log(
            "Test/batch_p_true_std",
            torch.sqrt(
                self.batch_p_true_mean_sq_metric.compute() - self.batch_p_true_mean_metric.compute() ** 2
            )
        )
        self.batch_p_true_mean_metric.reset()
        self.batch_p_true_mean_sq_metric.reset()

        # log batch_entropy metrics
        self.log(
            "Test/batch_entropy_mean",
            self.batch_entropy_mean_metric.compute()
        )
        self.log(
            "Test/batch_entropy_std",
            torch.sqrt(
                self.batch_entropy_mean_sq_metric.compute() - self.batch_entropy_mean_metric.compute() ** 2
            )
        )
        self.batch_entropy_mean_metric.reset()
        self.batch_entropy_mean_sq_metric.reset()

    def val_dataloader(self):
        # load data
        subs_paths = data_utils.parse_subs_eeg_dir(Path(self.hparams.subs_dir_path))
        subs_test_paths = {sub_num: path_dict["test"] for sub_num, path_dict in subs_paths.items()}

        test_dataset = data_utils.EegImgLatentDataset(
            Path(self.hparams.img_latent_test_path),
            subs_test_paths
        )

        # configure special samplers to assure batch diversity for CLIP-loss
        if self.hparams.num_devices == 1:
            test_sampler = data_utils.ClipSampler(test_dataset)
        else:
            test_sampler = data_utils.DistributedClipSampler(test_dataset)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            sampler=test_sampler
        )

        return test_loader

    def train_dataloader(self):
        # load data
        subs_paths = data_utils.parse_subs_eeg_dir(Path(self.hparams.subs_dir_path))
        subs_train_paths = {sub_num: path_dict["train"] for sub_num, path_dict in subs_paths.items()}

        train_dataset = data_utils.EegImgLatentDataset(
            Path(self.hparams.img_latent_train_path),
            subs_train_paths
        )

        # configure special samplers to assure batch diversity for CLIP-loss
        if self.hparams.num_devices == 1:
            train_sampler = data_utils.ClipSampler(train_dataset)
        else:
            train_sampler = data_utils.DistributedClipSampler(train_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=train_sampler
        )

        return train_loader

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

    # debug: disabled mode on
    logger = WandbLogger(
        project=config.experiment_name,
        log_model="all",
        # mode="disabled"
    )

    # create subject's embeddings
    subj_embeds = nn.Embedding(
        config.num_subjects,
        config.eeg_kwargs.input_length
    )
    # create lightning model and trainer
    model = LitEggEncoder(
        EEGEncoder(participants_embedding=subj_embeds, **config.eeg_kwargs),
        nn.Parameter(torch.FloatTensor([config.init_logit_scale])),
        config
    )
    trainer = L.Trainer(
        logger=logger,
        use_distributed_sampler=False,
        accelerator=config.accelerator,
        devices=config.num_devices,
        max_epochs=config.max_epochs,
        log_every_n_steps=70,
    )

    trainer.fit(model, ckpt_path="Sole EEG contrastive/w4mtrbmo/checkpoints/epoch=0-step=3308.ckpt")
