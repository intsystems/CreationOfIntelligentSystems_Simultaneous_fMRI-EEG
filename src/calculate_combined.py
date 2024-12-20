import os
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
from safetensors.torch import load_model
from accelerate import Accelerator

from brain_encoder import BrainEncoder, fMRIBrainEncoder, EEGBrainEncoder
from dataset import BrainStimuliDataset, select_random_dimension


def main(args):
    
    # load training configuration file
    config_path = os.path.join(args.config_dir, args.config_name)
    config_name = os.path.splitext(args.config_name)[0]
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
    
    # Check training mode
    # 1) 'fmri' - we train only fMRI encoder
    # 2) 'eeg' - we train only EEG encoder
    # 3) 'both' - we train the model on both fMRI and EEG data
    # 4) 'none' - we do not include fMRI and EEG in the data
    assert config.mode in ['fmri', 'eeg', 'both', 'none']
    
    # initialize dataset
    dataset = BrainStimuliDataset(**config.dataloader_kwargs.dataset, mode=config.mode)

    # choose corresponding model class
    if config.mode == 'fmri':
        Model = fMRIBrainEncoder
    elif config.mode == 'eeg':
        Model = EEGBrainEncoder
    elif config.mode == 'both':
        Model = BrainEncoder
        
    # initialize model and load weights from checkpoint
    model = Model(**config.model_kwargs, num_subs=dataset.num_subs).to(accelerator.device).to(weight_dtype)
    ckpt_path = os.path.join(config.output_dir, f'checkpoint-{args.steps}', 'model.safetensors')
    load_model(model, ckpt_path)

    # split indices between processes
    indices = [_ for _ in range(len(dataset))]
    if accelerator.is_main_process:
        pbar = tqdm(total=len(indices), desc='Processed combined embeds')

    # calculate combined embeddings for each data tripled (fMRI, EEG, image)
    # NOTE: image is chosen randomly from the provided chunk 
    model.eval()
    with accelerator.split_between_processes(indices) as chunked_indices:
        for idx in chunked_indices:
            
            # take (fMRI, EEG, image) triplet from dataset
            x = dataset[idx]
            sub_id = torch.tensor([x['id']]).to(device)
            if x['fmri'] is not None:
                fmri_embeds = x['fmri'].unsqueeze(0).to(device).to(weight_dtype)
            else:
                fmri_embeds = None
            if x['eeg'] is not None:
                eeg_embeds = x['eeg'].unsqueeze(0).to(device).to(weight_dtype)
            else:
                eeg_embeds = None

            # calculate combined embeddings via BrainEncoder
            with torch.no_grad():
                combined_embeds = model(
                    sub_ids=sub_id,
                    batch_eeg=eeg_embeds,
                    batch_fmri=fmri_embeds
                ).squeeze(0)
                
            # save combined embeds
            combined_dir = os.path.join(
                args.save_dir,
                config_name,
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
        default='fmri-monkeys.yaml'
    )
    parser.add_argument(
        '--steps', 
        type=int, 
        default=8000
    )
    args = parser.parse_args()
    main(args)