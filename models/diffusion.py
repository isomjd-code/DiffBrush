import torch
from tqdm import tqdm

class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by 
    computing a moving average of the parameters, which can help to reduce 
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0


    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


    def step_ema(self, ema_model, model, step_start_ema=20000000000000000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1


    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Diffusion:
    def __init__(self, noise_steps=1000, noise_offset=0, beta_start=1e-4, beta_end=0.02, device=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_offset = noise_offset
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
        self.device = device


    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)


    def predict_start_from_noise(self, x, t, noise):
        alpha_hat = self.alpha_hat[t][:, None, None, None]
        x_start = (x - (1 - alpha_hat).sqrt()*noise) / (alpha_hat.sqrt())
        return x_start

    def predict_t_minus_one_from_t(self, x_t, t, predicted_noise):
        # alpha = self.alpha[t][:, None, None, None]
        # alpha_hat = self.alpha_hat[t][:, None, None, None]
        # beta = self.beta[t][:, None, None, None]
        # noise = torch.zeros_like(x_t)
        # for i in range(len(t)):
        #     if t[i] > 0:
        #         noise[i] = torch.randn_like(x_t[i])
        # x = 1 / torch.sqrt(alpha) * (x_t - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        beta = self.beta[t][:, None, None, None]
        x = (x_t - beta.sqrt() * predicted_noise) / (1 - beta).sqrt()
        
        return x

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x) + self.noise_offset*torch.randn(x.shape[0], x.shape[1], 1, 1).to(self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


    def sample_timesteps(self, n, finetune=False):
        if finetune:
            return torch.randint(low=6, high=self.noise_steps, size=(n,))
        else:
            return torch.randint(low=0, high=self.noise_steps, size=(n,))


    def train_ddim(self, model, x, styles, content, total_t, wid, sampling_timesteps=6, eta=0):
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = [-1] + [i/sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x_start = None
        noise_list = []
        # for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
        for time, time_next in time_pairs:
            tmp_time, tmp_time_next = time, time_next
            batch_size = x.shape[0]
            # Clamp time values to valid range [0, total_timesteps-1]
            time_val = max(0, min(total_timesteps - 1, int(total_timesteps * time)))
            time_next_val = max(0, min(total_timesteps - 1, int(total_timesteps * time_next)))
            time = (torch.ones(batch_size) * time_val).long().to(self.device)
            time_next = (torch.ones(batch_size) * time_next_val).long().to(self.device)
            
            predicted_noise, vertical_proxy_loss, horizontal_proxy_loss = model(x, time, styles, content, wid, tag='train')
            noise_list.append(predicted_noise)
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            # denoise to approximately x0
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
            # Check if we're at the final timestep (time_next_val == 0 after clamping)
            if time_next_val == 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                  c * predicted_noise + \
                  sigma * noise # re-noising to x_t from approximately x0
            time_pairs.remove((tmp_time, tmp_time_next))
            break
        
        for time, time_next in time_pairs:
            batch_size = x.shape[0]
            # Clamp time values to valid range [0, total_timesteps-1]
            time_val = max(0, min(total_timesteps - 1, int(total_timesteps * time)))
            time_next_val = max(0, min(total_timesteps - 1, int(total_timesteps * time_next)))
            time = (torch.ones(batch_size) * time_val).long().to(self.device)
            time_next = (torch.ones(batch_size) * time_next_val).long().to(self.device)
            predicted_noise = model(x, time, styles, content)
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            # denoise to approximately x0
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
            # Check if we're at the final timestep (time_next_val == 0 after clamping)
            if time_next_val == 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                c * predicted_noise + \
                sigma * noise # re-noising to x_t from approximately x0
        
        return noise_list[0], vertical_proxy_loss, horizontal_proxy_loss


    def train_ddim_wo_nce(self, model, x, styles, content, total_t, sampling_timesteps=6, eta=0):
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = [-1] + [i/sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x_start = None
        noise_list = []
        # for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
        for time, time_next in time_pairs:
            batch_size = x.shape[0]
            # Clamp time values to valid range [0, total_timesteps-1]
            time_val = max(0, min(total_timesteps - 1, int(total_timesteps * time)))
            time_next_val = max(0, min(total_timesteps - 1, int(total_timesteps * time_next)))
            time = (torch.ones(batch_size) * time_val).long().to(self.device)
            time_next = (torch.ones(batch_size) * time_next_val).long().to(self.device)
            
            predicted_noise = model(x, time_next, styles, content)
            noise_list.append(predicted_noise)
            
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            
            # denoise to approximately x0
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
            # Check if we're at the final timestep (time_next_val == 0 after clamping)
            if time_next_val == 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            x = x_start * alpha_hat_next.sqrt() + \
                c * predicted_noise + \
                sigma * noise # re-noising to x_t from approximately x0
        
        return x, noise_list[0]


    def gernerate_ddim(self, model, x, styles, content, total_t, sampling_timesteps=6, eta=0):
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = [-1] + [i/sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x_start = None
        # for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
        for time, time_next in time_pairs:
            batch_size = x.shape[0]
            # Clamp time values to valid range [0, total_timesteps-1]
            time_val = max(0, min(total_timesteps - 1, int(total_timesteps * time)))
            time_next_val = max(0, min(total_timesteps - 1, int(total_timesteps * time_next)))
            time = (torch.ones(batch_size) * time_val).long().to(self.device)
            time_next = (torch.ones(batch_size) * time_next_val).long().to(self.device)
            
            predicted_noise = model(x, time, styles, content)
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            # denoise to approximately x0
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
            # Check if we're at the final timestep (time_next_val == 0 after clamping)
            if time_next_val == 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                  c * predicted_noise + \
                  sigma * noise # re-noising to x_t from approximately x0
        
        return x


    @torch.no_grad()
    def ddim_sample(self, model, vae, n, x, styles, content, sampling_timesteps=50, eta=0):
        model.eval()
        
        total_timesteps = self.noise_steps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # Scale initial noise to match training space (model expects scaled latents)
        # Training: latents = vae_latents * 0.18215
        # Inference: We start with unit Gaussian noise, so scale it to match training space
        # Now x is in the same space as training latents (scaled space)
        x = x * 0.18215
        
        for step_idx, (time, time_next) in enumerate(time_pairs):
            t = (torch.ones(n) * time).long().to(self.device)
            t_next = (torch.ones(n) * time_next).long().to(self.device)
            
            # During inference, use tag='test' (default) and no wid
            # UNet will use mix_net.generate() instead of mix_net() which doesn't need wid
            # Handle return value: tag='test' returns single tensor, tag='train' returns tuple
            result = model(x, t, styles, content, wid=None, tag='test')
            if isinstance(result, tuple):
                predicted_noise = result[0]  # Extract first element if tuple
            else:
                predicted_noise = result  # Single tensor
            
            # Debug: Check if model is predicting all zeros (model collapse)
            if step_idx == 0:
                pred_mean = predicted_noise.mean().item()
                pred_std = predicted_noise.std().item()
                if abs(pred_mean) < 1e-6 and pred_std < 1e-6:
                    print(f"Warning: Model appears to be predicting all zeros! This suggests model collapse.")
                elif abs(pred_mean) > 10 or pred_std > 10:
                    print(f"Warning: Model predictions are very large! mean={pred_mean:.4f}, std={pred_std:.4f}")
            
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            alpha_hat_next = self.alpha_hat[t_next][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            # Predict x0
            x_start = (x - (1 - alpha_hat).sqrt() * predicted_noise) / alpha_hat.sqrt()
            
            # Clamp x_start to reasonable range to prevent variance explosion
            # Training latents were in roughly [-3, 3] range in scaled space
            x_start = x_start.clamp(-3, 3)
            
            if time_next < 0:
                x = x_start
                break
                
            # DDIM step
            sigma = eta * ((1 - alpha_hat_next) / (1 - alpha_hat) * beta).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            
            x = alpha_hat_next.sqrt() * x_start + c * predicted_noise + sigma * noise
        
        model.train()
        
        # After DDIM loop, x is the predicted x0 in scaled, centered space
        # Training: latents = (vae_latents - mean) * 0.18215 (centered, then scaled)
        # Generation: vae_latents = (latents / 0.18215) + mean (unscale, then un-center)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN or Inf in final latents! Replacing with zeros.")
            x = torch.zeros_like(x)
        
        # Debug: Check latents after DDIM loop
        print(f"Debug: After DDIM loop - x mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        # Normalize output to match training distribution
        # Training scaled latents had std ≈ 1.1 (from training logs)
        target_std = 1.1
        current_std = x.std()
        current_mean = x.mean()
        
        if current_std > 0.01:  # Avoid division by zero
            # Center and rescale to target distribution
            x = (x - current_mean) / current_std * target_std
        
        print(f"Debug: After normalization - x mean={x.mean().item():.4f}, std={x.std().item():.4f}")
        
        # Unscale: convert from scaled space to normalized space
        latents = x / 0.18215
        print(f"Debug: After unscale - latents mean={latents.mean().item():.4f}, std={latents.std().item():.4f}")
        
        # Compute VAE expected mean dynamically by encoding a reference image
        # This ensures we use the correct offset for this specific VAE
        with torch.no_grad():
            # Encode a mid-gray image to get typical latent mean
            # Use a value that represents typical document images (mostly white)
            ref_img = torch.ones(1, 3, 64, 64).to(x.device) * 0.8  # 80% white in [-1,1] space
            ref_latent_dist = vae.encode(ref_img).latent_dist
            ref_latent_mean = ref_latent_dist.mean.mean().item()
            print(f"Debug: VAE expected latent mean (from ref image): {ref_latent_mean:.4f}")
        
        # Un-center: Add back the mean that was removed during training
        # Use the dynamically computed mean from VAE
        latent_mean_offset = ref_latent_mean
        latents = latents + latent_mean_offset
        print(f"Debug: Final latents for VAE - mean={latents.mean().item():.4f}, std={latents.std().item():.4f}, "
              f"range=[{latents.min().item():.4f}, {latents.max().item():.4f}]")
        
        # Decode
        image = vae.decode(latents).sample
        print(f"Debug: VAE output - mean={image.mean().item():.4f}, std={image.std().item():.4f}, "
              f"range=[{image.min().item():.4f}, {image.max().item():.4f}]")
        
        image = (image / 2 + 0.5).clamp(0, 1)
        print(f"Debug: Final image - mean={image.mean().item():.4f}, std={image.std().item():.4f}, "
              f"range=[{image.min().item():.4f}, {image.max().item():.4f}]")
        
        return image


    @torch.no_grad()
    def ddim_sample_latent(self, model, n, x, styles, laplace, content, sampling_timesteps=5, eta=0):
        model.eval()
        
        total_timesteps, sampling_timesteps = self.noise_steps, sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x_start = None
        
        # for time, time_next in tqdm(time_pairs, position=1, leave=False, desc='sampling'):
        for time, time_next in time_pairs:
            time = (torch.ones(n) * time).long().to(self.device)
            time_next = (torch.ones(n) * time_next).long().to(self.device)
            predicted_noise = model(x, time, styles, laplace, content)
            
            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            # denoise to approximately x0
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
            
            if time_next[0] < 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)
            
            x = x_start * alpha_hat_next.sqrt() + \
                  c * predicted_noise + \
                  sigma * noise # re-noising to x_t from approximately x0
        
        model.train()
        
        return x


    @torch.no_grad()
    def ddpm_sample(self, model, vae, n, x, styles, laplace, content):
        model.eval()
        
        # Scale initial noise to match training space (model expects scaled latents)
        # Check if input is unscaled (std ≈ 1.0) and scale if needed
        if x.std().item() > 0.5:  # Likely unscaled
            x = x * 0.18215
        
        # for i in tqdm(reversed(range(0, self.noise_steps)), position=1, leave=False, desc='sampling'):
        for i in reversed(range(0, self.noise_steps)):
            time = (torch.ones(n) * i).long().to(self.device)
            predicted_noise = model(x, time, styles, laplace, content)
            alpha = self.alpha[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            beta = self.beta[time][:, None, None, None]
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            # Clamp to prevent variance explosion
            x = x.clamp(-3, 3)
        
        model.train()
        
        # Normalize output to match training distribution
        # Training scaled latents had std ≈ 1.1
        target_std = 1.1
        current_std = x.std()
        current_mean = x.mean()
        
        if current_std > 0.01:  # Avoid division by zero
            # Center and rescale to target distribution
            x = (x - current_mean) / current_std * target_std
        
        # Reverse the scaling and centering applied during training
        # Training: latents = (vae_latents - mean) * 0.18215 (centered, then scaled)
        # Generation: vae_latents = (latents / 0.18215) + mean (unscale, then un-center)
        latents = x / 0.18215
        
        # Compute VAE expected mean dynamically
        with torch.no_grad():
            ref_img = torch.ones(1, 3, 64, 64).to(x.device) * 0.8  # 80% white
            ref_latent_dist = vae.encode(ref_img).latent_dist
            ref_latent_mean = ref_latent_dist.mean.mean().item()
        
        # Un-center: Add back the mean that was removed during training
        latent_mean_offset = ref_latent_mean
        latents = latents + latent_mean_offset
        
        image = vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).contiguous().numpy()

        image = torch.from_numpy(image)
        x = image.permute(0, 3, 1, 2).contiguous()
        
        return x
    
    @torch.no_grad()
    def ddim_sample_t(self, model, vae, x, styles, laplace, content, total_t, sampling_timesteps=50, eta=0):
        total_timesteps, sampling_timesteps = total_t, sampling_timesteps
        times = [-1] + [i/sampling_timesteps for i in range(1, sampling_timesteps + 1)]
        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))
        x_start = None
        
        # Scale initial noise to match training space (model expects scaled latents)
        # Check if input is unscaled (std ≈ 1.0) and scale if needed
        if x.std().item() > 0.5:  # Likely unscaled
            x = x * 0.18215
        
        for time, time_next in time_pairs:
            batch_size = x.shape[0]
            # Clamp time values to valid range [0, total_timesteps-1]
            time_val = max(0, min(total_timesteps - 1, int(total_timesteps * time)))
            time_next_val = max(0, min(total_timesteps - 1, int(total_timesteps * time_next)))
            time = (torch.ones(batch_size) * time_val).long().to(self.device)
            time_next = (torch.ones(batch_size) * time_next_val).long().to(self.device)
            predicted_noise = model(x, time, styles, laplace, content)

            beta = self.beta[time][:, None, None, None]
            alpha_hat = self.alpha_hat[time][:, None, None, None]
            alpha_hat_next = self.alpha_hat[time_next][:, None, None, None]
            
            x_start = (x - (1 - alpha_hat).sqrt()*predicted_noise) / (alpha_hat.sqrt())
            
            # Clamp x_start to reasonable range to prevent variance explosion
            x_start = x_start.clamp(-3, 3)
            
            # Check if we're at the final timestep (time_next_val == 0 after clamping)
            if time_next_val == 0:
                x = x_start
                continue
            
            sigma = eta * (beta * (1 - alpha_hat_next) / (1 - alpha_hat)).sqrt()
            c = (1 - alpha_hat_next - sigma ** 2).sqrt()
            
            noise = torch.randn_like(x)

            x = x_start * alpha_hat_next.sqrt() + \
                  c * predicted_noise + \
                  sigma * noise
        
        # Normalize output to match training distribution
        # Training scaled latents had std ≈ 1.1
        target_std = 1.1
        current_std = x.std()
        current_mean = x.mean()
        
        if current_std > 0.01:  # Avoid division by zero
            # Center and rescale to target distribution
            x = (x - current_mean) / current_std * target_std
        
        # Reverse the scaling and centering applied during training
        # Training: latents = (vae_latents - mean) * 0.18215 (centered, then scaled)
        # Generation: vae_latents = (latents / 0.18215) + mean (unscale, then un-center)
        latents = x / 0.18215
        
        # Compute VAE expected mean dynamically
        with torch.no_grad():
            ref_img = torch.ones(1, 3, 64, 64).to(x.device) * 0.8  # 80% white
            ref_latent_dist = vae.encode(ref_img).latent_dist
            ref_latent_mean = ref_latent_dist.mean.mean().item()
        
        # Un-center: Add back the mean that was removed during training
        latent_mean_offset = ref_latent_mean
        latents = latents + latent_mean_offset
        
        image = vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)   # 将数值范围从[-1,1]缩放至[0, 1]
        image = image.cpu().permute(0, 2, 3, 1).contiguous().numpy()

        image = torch.from_numpy(image)
        x = image.permute(0, 3, 1, 2).contiguous()
        
        return x
        