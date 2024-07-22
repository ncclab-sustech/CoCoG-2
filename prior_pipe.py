import numpy as np
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
from diffusers.utils.torch_utils import randn_tensor

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class PriorPipe:
    
    def __init__(self, diffusion_prior=None, scheduler=None, device=torch.device('cuda')):
        self.diffusion_prior = diffusion_prior.to(device)
        self.ema = deepcopy(self.diffusion_prior).to(device)  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)
        update_ema(self.ema, self.diffusion_prior, decay=0)  # Ensure EMA is initialized with synced weights
        self.ema.eval()  # EMA model should always be in eval mode

        if scheduler is None:
            from diffusers.schedulers import DDPMScheduler
            self.scheduler = DDPMScheduler() 
        else:
            self.scheduler = scheduler
            
        self.device = device
        
    def train(self, dataloader, path=None, num_epochs=10, learning_rate=1e-4, uncondition_rate=0.1):

        device = self.device
        self.diffusion_prior.train()

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(self.diffusion_prior.parameters(), lr=learning_rate)
        from diffusers.optimization import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=(len(dataloader) * num_epochs),
        )

        num_train_timesteps = self.scheduler.config.num_train_timesteps
        optimal_loss = 1e6
        for epoch in range(num_epochs):
            loss_sum = 0
            for batch in dataloader:
                h_embeds = batch[0].to(device)
                c_embeds = batch[1].to(device) if len(batch) > 1 else None
                N = h_embeds.shape[0]

                # 1. randomly replece c_embeds by None
                if torch.rand(1) < uncondition_rate:
                    c_embeds = None

                # 2. generate noisy embeddings as input
                noise = torch.randn_like(h_embeds)

                # 3. sample timestep
                timesteps = torch.randint(0, num_train_timesteps, (N,), device=device)

                # 4. add noise to h_embedding
                perturbed_h_embeds = self.scheduler.add_noise(
                    h_embeds,
                    noise,
                    timesteps
                ) # (batch_size, embed_dim)

                # 5. predict noise
                noise_pre = self.diffusion_prior(perturbed_h_embeds, timesteps, c_embeds)
                
                # 6. loss function weighted by sigma
                loss = criterion(noise_pre, noise) # (batch_size,)
                loss = loss.mean()
                            
                # 7. update parameters
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), 1.0)

                lr_scheduler.step()
                optimizer.step()
                update_ema(self.ema, self.diffusion_prior)

                loss_sum += loss.item()

            loss_epoch = loss_sum / len(dataloader)
            if loss_epoch < optimal_loss:
                optimal_loss = loss_epoch
                if path is not None:
                    torch.save(self.diffusion_prior.state_dict(), f'{path}.pt')
                    torch.save(self.ema.state_dict(), f'{path}_ema.pt')
            print(f'epoch: {epoch}, loss: {loss_epoch}, lr: {optimizer.param_groups[0]["lr"]}')
            # lr_scheduler.step(loss)
        

    @torch.no_grad()
    def generate(
            self, 
            c_embeds=None, 
            num_inference_steps=50, 
            timesteps=None,
            guidance_scale=5.0,
            generator=None,
            shape=None,
            N=1,
            use_ema=False,
        ):
        model = self.ema if use_ema else self.diffusion_prior
        model.eval()

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare c_embeds
        if c_embeds is not None:
            c_embeds = c_embeds.to(self.device)
            c_embeds = c_embeds.repeat_interleave(N, dim=0) # (n_cond*N, )
            N = c_embeds.shape[0] # n_cond * N

        # 3. Prepare noise
        if shape is None:
            shape = (self.diffusion_prior.clip_dim, )
        h_t = randn_tensor((N, *shape), generator=generator, device=self.device)
        # scale the initial noise by the standard deviation required by the scheduler
        h_t = h_t * self.scheduler.init_noise_sigma

        # 4. denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_tensor = torch.ones(N, dtype=torch.float, device=self.device) * t
            # 4.1 noise prediction
            if guidance_scale == 0 or c_embeds is None:
                noise_pred = model(h_t, t_tensor)
            else:
                noise_pred_cond = model(h_t, t_tensor, c_embeds)
                noise_pred_uncond = model(h_t, t_tensor)
                # perform classifier-free guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # 4.2 compute the previous noisy sample h_t -> h_{t-1}
            h_t = self.scheduler.step(noise_pred, t, h_t, generator=generator).prev_sample
        
        return h_t

    @torch.no_grad()
    def generate_guidance(
        self,
        loss_fn=None,
        N=1,
        num_inference_steps=50, 
        timesteps=None,
        guidance_scale=1,
        generator=None,
        shape=None,
        num_resampling_steps=1,
        latent=None,
        strength=1,
        use_ema=True,
    ):
        """
        References:
        [Guidance with Spherical Gaussian Constraint for Conditional Diffusion]
        [Understanding Training-free Diffusion Guidance: Mechanisms and Limitations]
        """
        if latent is not None:
            assert N == latent.shape[0] 
        model = self.ema if use_ema else self.diffusion_prior
        model.eval()

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare noise
        if latent is not None:
            denoise_start = int(strength * num_inference_steps)
            if denoise_start > 0:
                x_t = self.scheduler.add_noise(
                    latent, 
                    randn_tensor((N, *shape), generator=generator, device=self.device), 
                    torch.tensor(N * [timesteps[num_inference_steps-denoise_start-1]])
                )
            else:
                x_t = latent
            timesteps = timesteps[num_inference_steps-denoise_start:]
        else:
            x_t = randn_tensor((N, *shape), generator=generator, device=self.device)
        

        # 3. Denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_tensor = torch.ones(N, dtype=torch.float, device=self.device) * t
            for s in range(num_resampling_steps):
                with torch.enable_grad():
                    x_t.requires_grad_(True)
                    noise_pred = model(x_t, t_tensor)

                    # 3.1 Unconditional sampling x_t -> x_{t-1}
                    output = self.scheduler.step(noise_pred, t, x_t, generator=generator)
                    x_t_uncond, x_0t = output.prev_sample, output.pred_original_sample

                    # 3.2 Posterior sampling
                    sigma_t = self._get_variance(t) ** 0.5
                    sqrt_n_shape = np.prod(shape) ** 0.5
                    shape_multiplier = [1] * len(shape)

                    grad = torch.autograd.grad(loss_fn(x_0t), x_t)[0]
                    norm = torch.linalg.norm(grad.view(N, -1), dim=1).view(-1, *shape_multiplier)
                    grad = sqrt_n_shape * sigma_t * grad / norm
                    x_t.requires_grad_(False)

                # Apply classifier guidance
                x_t = x_t_uncond - guidance_scale * grad

                # 3.3 Resampling trick / time travel
                # resampling for s-1 times
                if s < num_resampling_steps - 1:
                    x_t = self._forward_one_step(x_t, t, generator) # q(h_t | h_{t-1})
                x_t = x_t.detach()  
            
        return x_t
    
    def _previous_timestep(self, timestep):
        num_inference_steps = self.scheduler.num_inference_steps
        prev_t = timestep - self.scheduler.config.num_train_timesteps // num_inference_steps
        return prev_t   
     
    def _get_variance(self, t):

        # get beta_t
        prev_t = self._previous_timestep(t)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def _forward_one_step(self, x_t, t, generator=None):

        # get beta_t
        prev_t = self._previous_timestep(t)
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0)
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # q(x_t | x_{t-1}):
        # DDPM: x_t = sqrt(1 - \beta_t) * x_{t-1} + \beta_t * N(0, 1)
        noise = randn_tensor(x_t.shape, generator=generator, device=x_t.device) 
        a, b = torch.sqrt(1 - current_beta_t), current_beta_t
        x_t = a * x_t + b * noise 
        
        return x_t

    def get_inversion(
            self,
            x,
            num_inference_steps=50, 
    ):
        from diffusers.schedulers import DDIMInverseScheduler
        self.inverse_scheduler = DDIMInverseScheduler()
        model = self.ema
        model.eval()

        # 1. Prepare timesteps
        from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, self.device, timesteps)

        # 2. Prepare noise
        x_t = x
        N = x.shape[0]
        shape = x.shape[1:]

        # 3. Denoising loop
        for _, t in tqdm(enumerate(timesteps)):
            t_tensor = torch.ones(N, dtype=torch.float, device=self.device) * t
            noise_pred = model(x_t, t_tensor)
            # 3.1 Unconditional sampling x_t -> x_{t-1}
            x_t = self.inverse_scheduler.step(noise_pred, t, x_t).prev_sample
            x_t = x_t.detach()  
        return x_t