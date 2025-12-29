import os
import argparse
import random
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
import torch.nn as nn
from data_loader.IAMDataset import IAMDataset
from data_loader.LatinBHODataset import LatinBHODataset
from models.unet import UNetModel
from diffusers import AutoencoderKL
from models.diffusion import Diffusion, EMA
import torch.distributed as dist
from tqdm import tqdm
from utils.util import fix_seed
import numpy as np
from torch.nn.utils import clip_grad_norm_

# WRITER_NUMS will be determined based on dataset

def main(args):
    """ load config file into cfg"""
    cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    
    """ set mulit-gpu """
    # Check if distributed training is actually available (environment variables set)
    use_distributed = args.local_rank is not None and os.environ.get('RANK') is not None
    if use_distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = args.local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()
    else:
        # Single GPU training
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        world_size = 1
    
    """Determine number of writers based on dataset"""
    # Determine writer_nums based on config file name or DATASET config
    if 'IAM' in args.cfg_file or getattr(cfg, 'DATASET', None) == 'IAM':
        writer_nums = 496
    elif 'LatinBHO' in args.cfg_file or getattr(cfg, 'DATASET', None) == 'LatinBHO':
        writer_nums = 1
    else:
        writer_nums = 496  # default
    
    """build model architecture"""
    diffusion = Diffusion(noise_steps=1000, device=device)
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM, nb_classes=writer_nums).to(device)
    
    """Load and Freeze VAE Encoder"""
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    """Create EMA model"""
    ema_unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                         out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                         attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                         context_dim=cfg.MODEL.EMB_DIM, nb_classes=writer_nums).to(device)
    ema_unet.load_state_dict(unet.state_dict())
    ema_unet.eval()
    ema = EMA(0.9999)
    
    """Setup optimizer"""
    optimizer = torch.optim.AdamW(unet.parameters(), lr=cfg.SOLVER.BASE_LR)
    
    """Load checkpoint if exists"""
    start_epoch = 0
    start_iter = 0
    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            unet.load_state_dict(checkpoint['model'])
            ema_unet.load_state_dict(checkpoint['ema_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint.get('epoch', 0)
            start_iter = checkpoint.get('iter', 0)
            print(f'Resumed from epoch {start_epoch}, iter {start_iter}')
    
    """Setup dataset and dataloader"""
    if 'IAM' in args.cfg_file or getattr(cfg, 'DATASET', None) == 'IAM':
        dataset = IAMDataset(
            image_path=cfg.TRAIN.IMAGE_PATH,
            style_path=cfg.TRAIN.STYLE_PATH,
            text_path=cfg.TRAIN.LABEL_PATH,
            type=cfg.TRAIN.TYPE,
            content_type='unifont'
        )
    elif 'LatinBHO' in args.cfg_file or getattr(cfg, 'DATASET', None) == 'LatinBHO':
        dataset = LatinBHODataset(
            image_path=cfg.TRAIN.IMAGE_PATH,
            style_path=cfg.TRAIN.STYLE_PATH,
            text_path=cfg.TRAIN.LABEL_PATH,
            type=cfg.TRAIN.TYPE,
            content_type='unifont'
        )
    else:
        raise IOError(f'Unknown dataset. Config file: {args.cfg_file}, DATASET: {getattr(cfg, "DATASET", "not set")}')
    
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=local_rank, shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.TRAIN.IMS_PER_BATCH, 
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.DATA_LOADER.NUM_THREADS, 
        pin_memory=True,
        drop_last=True,
        collate_fn=dataset.collate_fn_
    )
    
    """Create output directory"""
    if local_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    """Training loop"""
    total_iters = start_iter
    num_epochs = cfg.SOLVER.EPOCHS
    
    for epoch in range(start_epoch, num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        unet.train()
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}/{num_epochs}', disable=(local_rank != 0))
        
        for batch_idx, data in enumerate(progress_bar):
            img = data['img'].to(device)  # [B, 3, H, W] - target image
            style = data['style'].to(device)  # [B, 1, H, W] - style reference
            content = data['content'].to(device)  # [B, T, 16, 16] - content glyphs
            wid = data['wid'].to(device)  # writer IDs [B]
            
            # Convert RGB image to grayscale for VAE (VAE expects 3 channels)
            if img.shape[1] == 3:
                # Convert to grayscale but keep 3 channels for VAE
                img_for_vae = img  # VAE can handle RGB
            else:
                img_for_vae = img.repeat(1, 3, 1, 1)
            
            # Encode images to latent space
            with torch.no_grad():
                # Debug: Check image values before VAE encode (first batch only, every 1000 iters)
                if total_iters == 0 or (total_iters % 1000 == 0 and batch_idx == 0):
                    img_min = img_for_vae.min().item()
                    img_max = img_for_vae.max().item()
                    img_mean = img_for_vae.mean().item()
                    print(f"\n[Debug] Image stats before VAE encode - min={img_min:.4f}, max={img_max:.4f}, mean={img_mean:.4f}")
                    print(f"  Expected range: [-1, 1] for Stable Diffusion VAE")
                    if img_min < -1.1 or img_max > 1.1:
                        print(f"  ⚠ WARNING: Image values outside expected range!")
                
                # Encode target image to latent space
                latents = vae.encode(img_for_vae).latent_dist.sample()
                latents = latents * 0.18215  # scale factor
            
            # Sample timesteps
            t = diffusion.sample_timesteps(latents.shape[0], finetune=False).to(device)
            
            # Add noise
            x, noise = diffusion.noise_images(latents, t)
            
            # Forward pass - predict noise directly instead of using DDIM sampling
            # The model should predict the noise at timestep t
            predicted_noise, vertical_proxy_loss, horizontal_proxy_loss = unet(x, t, style, content, wid, tag='train')
            
            # Calculate loss between predicted noise and actual noise
            noise_loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Handle NaN/Inf in proxy losses (expected with single writer class)
            # Suppress repeated warnings - only print once per 1000 iterations
            if torch.isnan(vertical_proxy_loss) or torch.isinf(vertical_proxy_loss):
                if local_rank == 0 and total_iters % 1000 == 0:
                    print(f"\nNote: vertical_proxy_loss is NaN/Inf (expected with single writer), setting to 0")
                vertical_proxy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            if torch.isnan(horizontal_proxy_loss) or torch.isinf(horizontal_proxy_loss):
                if local_rank == 0 and total_iters % 1000 == 0:
                    print(f"\nNote: horizontal_proxy_loss is NaN/Inf (expected with single writer), setting to 0")
                horizontal_proxy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Total loss
            total_loss = noise_loss + 0.1 * vertical_proxy_loss + 0.1 * horizontal_proxy_loss
            
            # Check for NaN/Inf in total loss before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                if local_rank == 0:
                    print(f"\nWarning: NaN/Inf detected in total_loss at iter {total_iters}. Skipping this batch.")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping and compute gradient norm
            if cfg.SOLVER.GRAD_L2_CLIP > 0:
                grad_norm = clip_grad_norm_(unet.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
                # Check if gradients exploded after clipping
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    if local_rank == 0:
                        print(f"\nWarning: NaN/Inf in grad_norm at iter {total_iters}. Skipping optimizer step.")
                    continue
            else:
                # Calculate grad norm even if not clipping
                parameters = [p for p in unet.parameters() if p.grad is not None]
                if parameters:
                    grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters]))
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if local_rank == 0:
                            print(f"\nWarning: NaN/Inf in grad_norm at iter {total_iters}. Skipping optimizer step.")
                        continue
                else:
                    grad_norm = 0.0
            
            # Log statistics periodically to diagnose training
            if local_rank == 0 and total_iters % 100 == 0:
                with torch.no_grad():
                    pred_mean = predicted_noise.mean().item()
                    pred_std = predicted_noise.std().item()
                    noise_mean = noise.mean().item()
                    noise_std = noise.std().item()
                    # Calculate L2 distance between predicted and actual noise
                    l2_diff = torch.norm(predicted_noise - noise).item() / (predicted_noise.numel() ** 0.5)
                    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                    print(f"\n[Iter {total_iters}] Loss: {noise_loss.item():.4f} | "
                          f"Pred noise: mean={pred_mean:.4f}, std={pred_std:.4f} | "
                          f"Actual noise: mean={noise_mean:.4f}, std={noise_std:.4f} | "
                          f"L2 diff: {l2_diff:.4f} | Grad norm: {grad_norm_val:.4f} | "
                          f"Timestep range: {t.min().item()}-{t.max().item()}")
            
            optimizer.step()
            
            # Update EMA
            ema.step_ema(ema_unet, unet, step_start_ema=2000)
            
            total_iters += 1
            
            # Update progress bar
            if local_rank == 0:
                # Show more precision for proxy losses to see if they're just very small
                v_proxy_val = vertical_proxy_loss.item()
                h_proxy_val = horizontal_proxy_loss.item()
                progress_bar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'noise': f'{noise_loss.item():.4f}',
                    'v_proxy': f'{v_proxy_val:.6f}' if v_proxy_val > 0 else '0.000000',
                    'h_proxy': f'{h_proxy_val:.6f}' if h_proxy_val > 0 else '0.000000'
                })
            
            # Save checkpoint
            if total_iters % cfg.TRAIN.SNAPSHOT_ITERS == 0 and local_rank == 0:
                checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_{total_iters}.pth')
                torch.save({
                    'model': unet.state_dict(),
                    'ema_model': ema_unet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'iter': total_iters,
                }, checkpoint_path)
                
                # Also save as latest
                latest_path = os.path.join(args.output_dir, 'latest.pth')
                torch.save({
                    'model': unet.state_dict(),
                    'ema_model': ema_unet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'iter': total_iters,
                }, latest_path)
                print(f'Saved checkpoint at iter {total_iters}')
        
        if local_rank == 0:
            print(f'Epoch {epoch+1} completed')


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', dest='cfg_file', default='configs/IAM.yml',
                        help='Config file for training')
    parser.add_argument('--output_dir', dest='output_dir', default='outputs',
                        help='Output directory for checkpoints')
    parser.add_argument('--stable_dif_path', dest='stable_dif_path', default='model_zoo',
                        help='Path to stable diffusion VAE')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    main(args)

