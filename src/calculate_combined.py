import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from safetensors.torch import load_model
from accelerate import Accelerator

from brain_encoder import BrainEncoder
from dataset import BrainStimuliDataset, select_random_dimension


def main(args):
    
    # load training configuration file
    config_path = os.path.join(args.config_dir, args.config_name)
    config = OmegaConf.load(config_path)

    # set up accelerator to process in parallel
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    
    # determine device and weight data type based on mixed precision settings
    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # initialize model and load weights from checkpoint
    model = BrainEncoder(**config.model_kwargs).to(device).to(weight_dtype)
    ckpt_path = os.path.join(config.output_dir, f'checkpoint-{args.steps}', 'model.safetensors')
    load_model(model, ckpt_path)

    # initialize dataset
    dataset = BrainStimuliDataset(**config.dataloader_kwargs.dataset)

    # split indices between processes
    indices = [_ for _ in range(len(dataset))]
    if accelerator.is_main_process:
        pbar = tqdm(total=len(indices))

    # calculate combined embeddings for each data tripled (fMRI, EEG, image)
    # NOTE: image is chosen randomly from the provided chunk 
    model.eval()
    with accelerator.split_between_processes(indices) as chunked_indices:
        for idx in chunked_indices:
            
            # take (fMRI, EEG, image) triplet from dataset
            x = dataset[idx]
            sub_id = torch.tensor([x['id']]).to(device)
            fmri_embeds = x['fmri'].unsqueeze(0).to(device).to(weight_dtype)
            eeg_embeds = x['eeg'].unsqueeze(0).to(device).to(weight_dtype)

            # calculate combined embeddings via BrainEncoder
            with torch.no_grad():
                combined_embeds = model(sub_id, eeg_embeds, fmri_embeds).squeeze(0)
                
            # save combined embeds
            combined_dir = os.path.join(
                args.save_dir,
                x['index']['key'],
                x['index']['sub'],
                x['index']['ses'],
                x['index']['run']
            )
            os.makedirs(combined_dir, exist_ok=True)
            combined_path = os.path.join(combined_dir, f"chunk-{x['index']['chunk']}.pt")
            torch.save(combined_embeds, combined_path)
            
            # update progress bar
            if accelerator.is_main_process:
                pbar.update(accelerator.num_processes)
                
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculating combined embeddings via BrainEncoder")
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='/home/jovyan/shares/SR004.nfs2/nkiselev/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/combined'
    )
    parser.add_argument(
        '--config_dir', 
        type=str, 
        default='/home/jovyan/shares/SR004.nfs2/nkiselev/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/train/configs'
    )
    parser.add_argument(
        '--config_name', 
        type=str, 
        default='improved-dataloader.yaml'
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=31000
    )
    args = parser.parse_args()
    main(args)