import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.optim as optim
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter

import sys

sys.path.append("../src")
from dataset import select_random_dimension
from flow_matching_prior.model import FlowMatchingModel
from flow_matching_prior.dataset import EmbeddingDataset, EmbeddingDataLoader


def main(conf):
    device = torch.device(conf.device)

    weight_dtype = torch.float32
    if conf.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif conf.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    log_dir = os.path.join(conf.output_dir, "tensorboard_logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Build dataloader
    dataset = EmbeddingDataset(**conf.dataloader_kwargs.dataset)
    dataloader = EmbeddingDataLoader(dataset, **conf.dataloader_kwargs.dataloader)

    # Initialize the model and move it to the appropriate device
    model = FlowMatchingModel(**conf.model_kwargs).to(device).to(weight_dtype)
    model.train()

    # Initialize the optimize
    optimizer = optim.AdamW(model.parameters(), **conf.optimizer_kwargs)

    # Initialize the learning rate scheduler
    scheduler = get_scheduler(
        name=conf.scheduler_kwargs.name,
        optimizer=optimizer,
        num_warmup_steps=conf.scheduler_kwargs.warmup_steps,
        num_training_steps=conf.num_training_steps,
    )

    # Initialize training variables
    step = 0

    # Handle resuming from a checkpoint
    os.makedirs(os.path.join(conf.output_dir, "checkpoints"), exist_ok=True)
    if conf.get("resume_from_checkpoint"):
        if conf.resume_from_checkpoint == "latest":
            # Find the most recent checkpoint
            files = os.listdir(os.path.join(conf.output_dir, "checkpoints"))
            files = sorted(files, key=lambda x: int(x.split("-")[1].split(".")[0]))
            path = files[-1] if len(files) > 0 else None
        else:
            path = os.path.basename(conf.resume_from_checkpoint)

        if path is not None:
            checkpoint = torch.load(
                os.path.join(conf.output_dir, "checkpoints", path), map_location=device
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
            print(f"Resuming from checkpoint {path} (step: {step})")
        else:
            print(
                f"Checkpoint '{conf.resume_from_checkpoint}' does not exist. Starting a new training run."
            )

    # Initialize the progress bar
    progress_bar = tqdm(range(step, conf.num_training_steps), disable=False)

    # Training loop
    while step < conf.num_training_steps:
        for batch in dataloader:
            # Extract data from the batch
            combined_embeds = batch["combined_embedding"].to(weight_dtype).to(device)
            image_embeds = batch["image_embedding"].to(weight_dtype).to(device)
            image_embeds = select_random_dimension(image_embeds)
            bs = image_embeds.shape[0]

            # Flow matching objective
            x_0 = combined_embeds
            x_1 = image_embeds
            t = torch.sigmoid(torch.randn(bs, device=device))
            t = conf.scheduler_scale * t / (1 + (conf.scheduler_scale - 1) * t)
            t_ = t.reshape(-1, *((1,) * len(x_0.shape[1:])))

            x_t = t_ * x_1 + (1 - t_) * x_0
            velocity = x_1 - x_0

            with torch.autocast(device_type="cuda", dtype=weight_dtype):
                pred_velocity = model(x_t, t)
                loss = F.mse_loss(pred_velocity, velocity)

            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=conf.max_norm,
                norm_type=2.0,
            )
            optimizer.step()
            scheduler.step()

            # Log training metrics
            train_loss = loss.item()
            logs = {"train_loss": train_loss, "lr": scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Log to TensorBoard
            writer.add_scalar("Train/Loss", train_loss, step)
            writer.add_scalar("Train/Learning_Rate", scheduler.get_last_lr()[0], step)

            progress_bar.update(1)
            step += 1

            # Save checkpoints periodically
            if step % conf.checkpointing_steps == 0:
                save_dir = os.path.join(conf.output_dir, "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"checkpoint-{step}.pth")
                torch.save(
                    {
                        "step": torch.tensor(step),
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    save_path,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flow-matching-prior-fmri-monkeys-44000.yaml",
        required=False,
    )
    args = parser.parse_args()
    conf = OmegaConf.load(args.config)
    main(conf)
