import argparse
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from omegaconf import OmegaConf
from diffusers.optimization import get_scheduler
from torch.utils.tensorboard import SummaryWriter
import sys
import warnings
warnings.filterwarnings('ignore')

# Import custom scripts
sys.path.append('../src')
from dataset import BrainStimuliDataset, BrainStimuliDataLoader, select_random_dimension
from brain_encoder import fMRIBrainEncoder, EEGBrainEncoder, BrainEncoder

# /home/jovyan/.mlspace/envs/genimages

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/fmri-monkeys.yaml",
                        required=False, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load configuration file
    config = OmegaConf.load(args.config_path)

    # Set device from config
    device = torch.device(config['device'])

    # Initialize TensorBoard writer
    log_dir = os.path.join(config['output_dir'], 'tensorboard_logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # Check training mode
    assert config['mode'] in ['fmri', 'eeg', 'both', 'none']

    # Choose corresponding model class
    if config['mode'] == 'fmri':
        Model = fMRIBrainEncoder
    elif config['mode'] == 'eeg':
        Model = EEGBrainEncoder
    elif config['mode'] == 'both':
        Model = BrainEncoder

    # Build dataloader
    dataset = BrainStimuliDataset(**config['dataloader_kwargs']['dataset'], mode=config['mode'])
    dataloader = BrainStimuliDataLoader(dataset, **config['dataloader_kwargs']['dataloader'], mode=config['mode'])

    # Initialize the model and move it to the appropriate device
    model = Model(**config['model_kwargs'], num_subs=dataset.num_subs).to(device)

    # Set weight data type based on config
    weight_dtype = torch.float32
    if config['mixed_precision'] == "fp16":
        weight_dtype = torch.float16
    elif config['mixed_precision'] == "bf16":
        weight_dtype = torch.bfloat16
    model = model.to(dtype=weight_dtype)

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), **config['optimizer_kwargs'])

    # Initialize the learning rate scheduler
    scheduler = get_scheduler(
        name=config.scheduler_kwargs.name,
        optimizer=optimizer,
        num_warmup_steps=config.scheduler_kwargs.warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_epochs * config.gradient_accumulation_steps,
    )

    # Initialize training variables
    global_step = 0
    first_epoch = 0

    # Handle resuming from a checkpoint
    os.makedirs(os.path.join(config['output_dir'], "checkpoints"), exist_ok=True)
    if config.get('resume_from_checkpoint'):
        if config['resume_from_checkpoint'] == 'latest':
            # Find the most recent checkpoint
            files = os.listdir(os.path.join(config['output_dir'], "checkpoints"))
            files = sorted(files, key=lambda x: int(x.split("-")[1].split('.')[0]))
            path = files[-1] if len(files) > 0 else None
        else:
            path = os.path.basename(config['resume_from_checkpoint'])

        if path is not None:
            checkpoint = torch.load(os.path.join(config['output_dir'], 'checkpoints', path), map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            global_step = checkpoint['global_step']
            first_epoch = checkpoint['epoch']
            print(f"Resuming from checkpoint {path} (global step: {global_step})")
        else:
            print(f"Checkpoint '{config['resume_from_checkpoint']}' does not exist. Starting a new training run.")

    # Initialize the progress bar
    progress_bar = tqdm(range(global_step, config['max_train_steps']), disable=False)

    # Training loop
    for epoch in range(first_epoch, config['max_train_epochs']):
        model.train()

        for step, batch in enumerate(dataloader):
            # Extract data from the batch
            sub_ids = batch["id"].to(device)
            if batch["eeg"] is not None:
                batch_eeg = batch["eeg"].to(dtype=weight_dtype).to(device)
            else:
                batch_eeg = None
            if batch["fmri"] is not None:
                batch_fmri = batch["fmri"].to(dtype=weight_dtype).to(device)
            else:
                batch_fmri = None
            image_features = batch["frames"].to(dtype=weight_dtype).to(device)
            image_features = select_random_dimension(image_features)

            # Compute the loss
            output = model.loss(
                sub_ids=sub_ids,
                batch_eeg=batch_eeg,
                batch_fmri=batch_fmri,
                image_features=image_features
            )
            loss = output['loss']
            logits = output['logits_per_brain'].detach()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Compute accuracy
            predicted = torch.argmax(logits, dim=1)
            labels = torch.arange(config['dataloader_kwargs']['dataloader']['batch_size'], device=device)
            accuracy = (predicted == labels).float().mean().item()

            # Log training metrics
            train_loss = loss.item()

            logs = {'train_loss': train_loss, 'accuracy': accuracy, 'lr': scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Log to TensorBoard
            writer.add_scalar('Train/Loss', train_loss, global_step)
            writer.add_scalar('Train/Accuracy', accuracy, global_step)
            writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], global_step)

            progress_bar.update(1)
            global_step += 1

            # Save checkpoints periodically
            if global_step % config['checkpointing_steps'] == 0:
                save_dir = os.path.join(config['output_dir'], "checkpoints")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"checkpoint-{global_step}.pth")
                torch.save({
                    'epoch': torch.tensor(epoch),
                    'global_step': torch.tensor(global_step),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, save_path)

            # Break the loop if the maximum number of training steps is reached
            if global_step >= config['max_train_steps']:
                break

    # Close TensorBoard writer
    writer.close()

if __name__ == '__main__':
    main()
