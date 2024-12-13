import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset


class DiffusionPrior:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device='cuda'):
        self.diffusion_prior = diffusion_prior.to(device)
        
        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
        else:
            self.scheduler = scheduler
            
        self.device = device
        
    def train(self, dataloader, num_epochs=10, learning_rate=1e-4):
        self.diffusion_prior.train()
        device = self.device
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                combined_embeds = batch['combined_embedding'].to(device) if 'combined_embedding' in batch.keys() else None
                image_embeds = batch['image_embedding'].to(device)
                N = image_embeds.shape[0]

                # 1. randomly replecing combined_embeds to None
                if torch.rand(1) < 0.1:
                    combined_embeds = None

                # 2. Generate noisy embeddings as input
                noise = torch.randn_like(image_embeds)

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to image_embedding
                perturbed_image_embeds = self.scheduler.add_noise(
                    image_embeds,
                    noise,
                    timesteps
                ) # (batch_size, embed_dim), (batch_size, )

                # 5. predict noise
                noise_pred = self.diffusion_prior(perturbed_image_embeds, timesteps, combined_embeds)
                
                # 6. loss function weighted by sigma
                loss = criterion(noise_pred, noise) # (batch_size,)
                loss = loss.mean()
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            print(f'epoch: {epoch}, loss: {loss_epoch}')

    def generate(
            self, 
            combined_embeds=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None
        ):
        # combined_embeds (batch_size, cond_dim)
        self.diffusion_prior = self.diffusion_prior.eval().float()
        N = combined_embeds.shape[0] if combined_embeds is not None else 1

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        print('num_inference_steps:', num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(scheduler=self.scheduler, num_inference_steps=num_inference_steps, device=self.device, timesteps=timesteps)

        # 2. Prepare combined_embeds
        if combined_embeds is not None:
            combined_embeds = combined_embeds.to(self.device).float()

        # 3. Prepare noise
        clip_t = torch.randn(
            N, self.diffusion_prior.embed_dim, 
            generator=generator, device=self.device, dtype=combined_embeds.dtype
        )

        # 4. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t = torch.ones(clip_t.shape[0], dtype=combined_embeds.dtype, device=self.device) * t
            
            # 4.1 noise prediction
            if guidance_scale == 0 or combined_embeds is None:
                noise_pred = self.diffusion_prior(clip_t, t)
            else:
                noise_pred_cond = self.diffusion_prior(clip_t, t, combined_embeds)
                noise_pred_uncond = self.diffusion_prior(clip_t, t)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample clip_t -> clip_{t-1}
            clip_t = self.scheduler.step(noise_pred, t.long().item(), clip_t, generator=generator).prev_sample
        
        return clip_t