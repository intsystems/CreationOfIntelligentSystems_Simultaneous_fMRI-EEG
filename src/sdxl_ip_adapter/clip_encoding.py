import json
import os
import torch
import glob
import open_clip
from diffusers.utils import load_image
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.image_paths = []
        for key in data_dict.keys():
            frames_dir = data_dict[key]['frames_dir']
            self.image_paths.extend(glob.glob(f'{frames_dir}/*.jpg'))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = load_image(image_path)
        return image, image_path

def main():
    json_path = '/home/jovyan/shares/SR004.nfs2/nkiselev/visual_stimuli_reconstruction/dataset.json'
    with open(json_path, "r") as file:
        data_dict = json.load(file)

    accelerator = Accelerator()
    device = accelerator.device

    image_encoder, _, feature_extractor = open_clip.create_model_and_transforms(
        'ViT-H-14',
        pretrained='laion2b_s32b_b79k',
        precision='fp16',
        device=device
    )

    dataset = ImageDataset(data_dict)

    image_encoder.eval()
    with torch.no_grad():
        for image, image_path in tqdm(dataset, desc="Processing images"):
            image_processed = feature_extractor(image)[None, ...].to(device, dtype=torch.float16)
            image_embeds = image_encoder.encode_image(image_processed)[0]
            basename = image_path.replace('.jpg', '')
            embed_path = basename + '.pt'
            torch.save(image_embeds, embed_path)

if __name__ == "__main__":
    main()
