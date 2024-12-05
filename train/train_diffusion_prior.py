import argparse
import os
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.schedulers import DDPMScheduler
from diffusers.optimization import get_scheduler

# Import custom scripts
import sys
sys.path.append('../src')
from dataset import select_random_dimension
from diffusion_prior.model import DiffusionPriorUNet
from diffusion_prior.dataset import EmbeddingDataset, EmbeddingDataLoader
from utils import Timer

import wandb

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/diffusion-prior.yaml",
                        required=False, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration file
    config = OmegaConf.load(args.config_path)
    os.environ["WANDB_API_KEY"] = config.wandb_api_key

    # Set up logging directory
    logging_dir = os.path.join(config.output_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)

    # Configure the project for logging and output
    project_config = ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=logging_dir
    )

    # Initialize the Accelerator for distributed training and mixed precision
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.log_with,
        project_config=project_config
    )

    # Determine the weight data type based on mixed precision settings
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Initialize the model and move it to the appropriate device and data type
    diffusion_prior = DiffusionPriorUNet(**config.model_kwargs).to(accelerator.device).to(weight_dtype)
    
    # Initialize diffusion scheduler (DDPM)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps) 

    # Build dataloader
    dataset = EmbeddingDataset(**config.dataloader_kwargs.dataset)
    dataloader = EmbeddingDataLoader(dataset, **config.dataloader_kwargs.dataloader)

    # Initialize the optimizer
    optimizer = optim.AdamW(diffusion_prior.parameters(), **config.optimizer_kwargs)

    # Initialize the learning rate scheduler
    scheduler = get_scheduler(
        name=config.scheduler_kwargs.name,
        optimizer=optimizer,
        num_warmup_steps=config.scheduler_kwargs.warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_epochs * config.gradient_accumulation_steps,
    )

    # Prepare the model, optimizer, scheduler, and data loaders for distributed training
    diffusion_prior, optimizer, scheduler, dataloader = accelerator.prepare(
        diffusion_prior, optimizer, scheduler, dataloader
    )

    # Initialize tracking for experiments
    if accelerator.is_main_process:
        tracker_config = dict(config)
        accelerator.init_trackers(config.experiment_name, tracker_config)

    # Initialize training variables
    global_step = 0
    first_epoch = 0

    # Handle resuming from a checkpoint
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint == 'latest':
            # Find the most recent checkpoint
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        else:
            path = os.path.basename(config.resume_from_checkpoint)

        if path is None:
            accelerator.print(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run.")
            config.resume_from_checkpoint = None
        else:
            if accelerator.is_main_process:
                accelerator.print(f"Resuming from checkpoint {path}")
                global_step = int(path.split("-")[1])
                first_epoch = 0
                accelerator.load_state(os.path.join(config.output_dir, path))

    # Initialize the progress bar
    progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process)

    # Training loop
    for epoch in range(first_epoch, config.max_train_epochs):
        diffusion_prior.train()
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(diffusion_prior):
                # Extract data from the batch
                combined_embeds = batch["combined_embedding"].to(weight_dtype).to(accelerator.device)
                image_embeds = batch["image_embedding"].to(weight_dtype).to(accelerator.device)
                image_embeds = select_random_dimension(image_embeds)
                bs = image_embeds.shape[0]

                # 1. randomly replacing combined_embeds to None
                if torch.rand(1) < config.cfg_drop_rate:
                    combined_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(image_embeds).to(weight_dtype).to(accelerator.device)

                # 3. sample timestep
                timesteps = torch.randint(
                    0, config.num_train_timesteps, (bs,), 
                    device=accelerator.device,
                    dtype=torch.int64
                )

                # 4. add noise to image_embedding
                perturbed_image_embeds = noise_scheduler.add_noise(
                    image_embeds,
                    noise,
                    timesteps
                ) # (batch_size, embed_dim), (batch_size, )

                # 5. predict noise
                noise_pred = diffusion_prior(perturbed_image_embeds, timesteps, combined_embeds).to(weight_dtype)

                # 6. loss function weighted by sigma
                loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
                loss = loss.mean()

                # 7. update parameters
                optimizer.zero_grad()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(diffusion_prior.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(config.dataloader_kwargs.dataloader.batch_size)).mean()
                train_loss = avg_loss.item() / config.gradient_accumulation_steps

                # Log training metrics
                logs = {'train_loss': train_loss, 'lr': scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({'Train Loss': train_loss}, step=global_step)
                    accelerator.log({'Learning Rate': scheduler.get_last_lr()[0]}, step=global_step)

                    # Save checkpoints periodically
                    if global_step % config.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)

            # Break the loop if the maximum number of training steps is reached
            if global_step >= config.max_train_steps:
                break

    # End training
    accelerator.end_training()
