import os
import json
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


def clip_score(combined_embeds: torch.Tensor, image_embeds: torch.Tensor) -> float:
    """
    Calculate the CLIP-Score between two CLIP embeddings.

    Parameters:
    combined_embeds (torch.Tensor): The first CLIP embedding.
    image_embeds (torch.Tensor): The second CLIP embedding.

    Returns:
    float: The CLIP-Score (cosine similarity) between the two embeddings.
    """
    
    # normalize combined embeds
    combined_embeds = combined_embeds / combined_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # normalize image embeds
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    
    # cosine similarity between feature vectors
    score = (combined_embeds * image_embeds).sum(axis=-1)

    return score


class AutoVivification(dict):
    """
    Implementation of perl's autovivification feature.
    This is very convenient way to use nested dicts, adding new elements.
    """
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


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

    # initialize dataset
    dataset = BrainStimuliDataset(**config.dataloader_kwargs.dataset, mode='none')

    # initialize dict for CLIP-Score
    scores_dict = {key: {sub: [] for sub in list(dataset.data_dict[key].keys())[1:]} for key in dataset.data_dict.keys()}

    # split indices between processes
    indices = [_ for _ in range(len(dataset))]
    if accelerator.is_main_process:
        pbar = tqdm(total=len(indices), desc='Processed chunks')

    # calculate CLIP-Score for each combined embedding
    # NOTE: image is chosen randomly from the provided chunk 
    with accelerator.split_between_processes(indices) as chunked_indices:
        for idx in chunked_indices:
            
            # get data sample
            x = dataset[idx]
            
            # get combined embeds path
            combined_dir = os.path.join(
                args.combined_dir,
                config_name,
                x['index']['key'],
                x['index']['sub'],
                x['index']['ses'],
                x['index']['run']
            )
            combined_path = os.path.join(combined_dir, f"chunk-{x['index']['chunk']}.pt")
            
            # load combine embeds
            combined_embeds = torch.load(combined_path, map_location="cpu").to(device).to(weight_dtype)
            
            # load image embeds
            frames = x['frames'].to(device).to(weight_dtype).unsqueeze(0)
            image_embeds = select_random_dimension(frames)

            # calculate CLIP-Score
            score = clip_score(combined_embeds, image_embeds).item()
            
            # update scores dict
            scores_dict[x['index']['key']][x['index']['sub']].append(score)
            
            # update progress bar
            if accelerator.is_main_process:
                pbar.update(accelerator.num_processes)
                
    # save scores into {process.index}.json file
    metrics = AutoVivification()
    for key in scores_dict.keys():
        for sub, scores in scores_dict[key].items():
            metrics[config_name][key][sub] = scores
    process_index = accelerator.process_index
    tmp_dir = os.path.join(args.metrics_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, f'{process_index}.json'), 'w') as outfile:
        json.dump(metrics, outfile, indent=4)
        
    # merge .json files into the one and save it
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        
        metrics_path = os.path.join(args.metrics_dir, 'metrics.json')

        # if we already have metrics.json file
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                merged_metrics = json.load(f)
            merged_metrics = AutoVivification(merged_metrics)
        else:
            merged_metrics = AutoVivification()
            
        # initialize empty lists for all the videos and subs
        merged_metrics[config_name] = {key: {sub: [] for sub in list(dataset.data_dict[key].keys())[1:]} for key in dataset.data_dict.keys()}
        
        # extend these lists for each process index
        for idx in range(accelerator.num_processes):
            json_path = os.path.join(tmp_dir, f'{idx}.json')
            with open(json_path, 'r') as f:
                tmp_metrics = json.load(f)
            for key in tmp_metrics[config_name].keys():
                for sub in tmp_metrics[config_name][key].keys():
                    merged_metrics[config_name][key][sub].extend(tmp_metrics[config_name][key][sub])
                
        # dump it to json file
        with open(metrics_path, 'w') as outfile:
            json.dump(merged_metrics, outfile, indent=4)
                
            
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Calculating CLIP-Score for combined embeddings.")
    parser.add_argument(
        '--metrics_dir', 
        type=str, 
        default='/home/jovyan/shares/SR004.nfs2/nkiselev/visual_stimuli_reconstruction/CreationOfIntelligentSystems_Simultaneous_fMRI-EEG/code/data/metrics'
    )
    parser.add_argument(
        '--combined_dir', 
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
    args = parser.parse_args()
    main(args)