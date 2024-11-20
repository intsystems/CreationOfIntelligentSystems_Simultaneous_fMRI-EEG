from pathlib import Path
from datetime import datetime
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from src.egg_encoder import EEG_Encoder
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from src.dataset import build_dataloaders
from optimizer import get_optimizer


# load config
with open("config.yaml", "r") as f:
    config: dict = yaml.full_load(f)

# create current launch dir
launch_dir  = Path("launch/" + str(datetime.now()))
launch_dir.mkdir(parents=True)

# make Tensorboard writer
writer_log_dir = launch_dir / "runs"
writer_log_dir.mkdir()
writer = SummaryWriter(writer_log_dir)

# define device
device = torch.device(config["device"])

# build dataloaders
train_dataloader, test_dataloader = build_dataloaders(
    config["dataset_json"],
    config["batch_size"], 
    config["train_data_ratio"]
)

# create encoder
eeg_encoder = EEG_Encoder(**config["eeg_encoder"]).to(device)

# load CLIP model
clip_vision = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)

# define optimizer
optimizer = get_optimizer(eeg_encoder)

# counter for overall step in training process
global_step = 0

for epoch in tqdm(range(config["num_epochs"]), desc="Epoch"):
    eeg_encoder.train()

    for batch in train_dataloader:
        optimizer.zero_grad()

        eeg_signal = batch["eeg"].to(device)
        imgs = batch["frames"].to(device)

        # get mean CLIP embedding out of given frames
        num_frames = imgs.shape[1]
        frames_embeds = []
        for frame_num in range(num_frames):
            frames_embeds.append(
                clip_vision(imgs[:, frame_num, ...]).image_embeds / num_frames
            )
        clip_emb = sum(frames_embeds)

        model_emb = eeg_encoder(eeg_signal)
        loss = F.mse_loss(model_emb, clip_emb)

        loss.backward()
        optimizer.step()

        writer.add_scalar("Train/MSE", loss.item(), global_step)
        global_step += 1

    # model's evalutation
    eeg_encoder.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            eeg_signal = batch["eeg"].to(device)
            imgs = batch["frames"].to(device)

            # get mean CLIP embedding out of given frames
            num_frames = imgs.shape[1]
            frames_embeds = []
            for frame_num in range(num_frames):
                frames_embeds.append(
                    clip_vision(imgs[:, frame_num, ...]).image_embeds / num_frames
                )
            clip_emb = sum(frames_embeds)

            model_emb = eeg_encoder(eeg_signal)
            loss = F.mse_loss(model_emb, clip_emb)

            writer.add_scalar("Test/MSE", loss.item(), global_step)
            global_step += 1

    # save model
    torch.save(eeg_encoder.state_dict(), launch_dir / "model.pt")

writer.close()
