# Encode images with CLIP-ViT-H-14 into vector of dim 1024

import open_clip
from diffusers.utils import load_image

image_encoder, _, feature_extractor = open_clip.create_model_and_transforms(
    'ViT-H-14', pretrained='laion2b_s32b_b79k', precision='fp16', device='cuda')

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")

image_processed = feature_extractor(image)[None, ...].to("cuda", dtype=torch.float16)
image_embeds = image_encoder.encode_image(image_processed)