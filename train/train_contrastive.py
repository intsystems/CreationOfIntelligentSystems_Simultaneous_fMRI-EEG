import argparse
import os
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.optimization import get_scheduler

# our scripts
import sys
sys.path.append('../src')
from dataset import build_dataloaders, select_random_dimension
from brain_encoder import BrainEncoder

import wandb

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config.yaml", 
                        required=False, help="Path to the configuration file.")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    os.environ["WANDB_API_KEY"] = config.wandb_api_key
    
    logging_dir = os.path.join(config.output_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)
    
    project_config = ProjectConfiguration(
        project_dir=config.output_dir, 
        logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.log_with,
        project_config=project_config,
    )
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model = BrainEncoder(
        fmri_masks_path="/home/jovyan/shares/SR004.nfs2/nkiselev/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/natview/fmri_masks",
        input_length=525, 
        num_channels=61, 
        latent_dim=4096, 
        embed_dim=1024, 
        patch_size=16, 
        in_chans=1
    ).to(accelerator.device).to(weight_dtype)
    
    train_dataloader, val_dataloader = build_dataloaders(
        dataset_json="/home/jovyan/shares/SR004.nfs2/nkiselev/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/dataset.json", 
        batch_size=16, 
        train_ratio=0.9)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    scheduler = get_scheduler(
        name=config.scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_epochs * config.gradient_accumulation_steps,
    )
    
    model, optimizer, scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, val_dataloader
    )
    
    if accelerator.is_main_process:
        tracker_config = dict(config)
        accelerator.init_trackers(config.experiment_name, tracker_config)
        
    global_step = 0
    first_epoch = 0
                
    if config.resume_from_checkpoint:
        
        if config.resume_from_checkpoint == 'latest':
            # the most recent checkpoint
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

    progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(first_epoch, config.max_train_epochs):
        
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                sub_ids = batch["id"]
                batch_eeg = batch["eeg"]
                batch_fmri = batch["fmri"]
                image_features = batch["frames"]

                image_features = select_random_dimension(image_features)
                optimizer.zero_grad()

                batch_size = batch_eeg.shape[0]
                brain_features = model(sub_ids, batch_eeg, batch_fmri)
                logit_scale = accelerator.unwrap_model(model).logit_scale
                
                loss = accelerator.unwrap_model(model).loss_func(
                    image_features=image_features, 
                    brain_features=brain_features, 
                    logit_scale=logit_scale
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                train_loss += avg_loss.item() / config.gradient_accumulation_steps

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Compute the corresponding logits and calculate accuracy
                logits = logit_scale * brain_features @ image_features.T
                predicted = torch.argmax(logits, dim=1) # (n_batch, ) in {0, 1, ..., n_cls-1}
                labels = torch.arange(batch_size)
                correct = (predicted == labels).sum().item()
                
                logs = {'loss': loss.detach().item(), 'accuracy': correct / batch_size, 'lr': scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                
                if accelerator.sync_gradients:
                    
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({'train_loss': train_loss}, step=global_step)
                    accelerator.log({'lr': scheduler.get_last_lr()[0]}, step=global_step)
                    train_loss = 0

                    if global_step % config.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logging.info(f"Saved state to {save_path}")

            if global_step >= config.max_train_steps:
                break
            
    accelerator.end_training()