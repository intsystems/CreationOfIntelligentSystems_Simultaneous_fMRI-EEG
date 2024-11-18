from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl_vit-h.safetensors", torch_dtype=torch.float16)
pipe.set_ip_adapter_scale(1)

image_embeds = ... # vector of dim 1024

ip_adapter_image_embeds = torch.cat([
    torch.zeros_like(image_embeds),
    image_embeds
]).unsqueeze(1)
ip_adapter_image_embeds.shape

image = pipe(
    prompt='', 
    ip_adapter_image_embeds=[ip_adapter_image_embeds], 
    num_inference_steps=30,
    guidance_scale=5.0,
    # generator=generator,
).images[0]