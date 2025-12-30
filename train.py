import os
import argparse
import random
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader.IAMDataset import IAMDataset
from data_loader.LatinBHODataset import LatinBHODataset, LatinBHOGenerateDataset
from models.unet import UNetModel
from diffusers import AutoencoderKL
from models.diffusion import Diffusion, EMA
from models.pylaia_crnn import create_pylaia_supervisor, SymbolTable
import torch.distributed as dist
from tqdm import tqdm
from utils.util import fix_seed
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torchvision
from PIL import Image, ImageOps

# WRITER_NUMS will be determined based on dataset

def safe_cuda_clear():
    """Safely clear CUDA cache and synchronize, handling errors gracefully."""
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        # CUDA context may be corrupted, ignore
        pass
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        # CUDA context may be corrupted, ignore
        pass

def get_gpu_memory_usage():
    """Get current GPU memory usage as a fraction (0-1). Returns None if not available."""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            return allocated / total, reserved / total
    except Exception:
        pass
    return None, None

def should_skip_heavy_operations(threshold=0.85):
    """Check if GPU memory is too high to safely run heavy operations like PyLaia CTC."""
    try:
        allocated_frac, reserved_frac = get_gpu_memory_usage()
        if allocated_frac is not None and allocated_frac > threshold:
            return True
    except Exception:
        pass
    return False

def check_model_parameters_for_nan(model, model_name="model", local_rank=0):
    """Check if any model parameters contain NaN/Inf values."""
    has_nan = False
    has_inf = False
    nan_params = []
    inf_params = []
    cuda_error = False
    
    try:
        for name, param in model.named_parameters():
            try:
                # Move to CPU first to avoid CUDA errors affecting the check
                param_cpu = param.detach().cpu()
                if torch.isnan(param_cpu).any():
                    has_nan = True
                    nan_count = torch.isnan(param_cpu).sum().item()
                    nan_params.append((name, nan_count))
                if torch.isinf(param_cpu).any():
                    has_inf = True
                    inf_count = torch.isinf(param_cpu).sum().item()
                    inf_params.append((name, inf_count))
            except RuntimeError as e:
                if "CUDA" in str(e):
                    cuda_error = True
                    if local_rank == 0:
                        print(f"  ⚠️ CUDA error while checking {name}: {e}")
                    break
                raise
    except RuntimeError as e:
        if "CUDA" in str(e):
            cuda_error = True
            if local_rank == 0:
                print(f"  ⚠️ CUDA error during parameter check: {e}")
    
    if cuda_error and local_rank == 0:
        print(f"\n" + "="*70)
        print(f"🚨 CUDA CONTEXT CORRUPTED - UNRECOVERABLE ERROR")
        print(f"="*70)
        print(f"The CUDA context has entered an error state that cannot be recovered")
        print(f"within this process. You must restart training:")
        print(f"")
        print(f"  1. Press Ctrl+C to stop this training")
        print(f"  2. Run: python train.py --resume [other args]")
        print(f"")
        print(f"Training will resume from the last saved checkpoint.")
        print(f"="*70 + "\n")
        return True, True, [], []  # Return as if corrupted to trigger exit
    
    if (has_nan or has_inf) and local_rank == 0:
        print(f"  [{model_name}] Found NaN/Inf in parameters:")
        if nan_params:
            for name, count in nan_params[:10]:  # Show first 10
                print(f"    NaN in {name}: {count} values")
        if inf_params:
            for name, count in inf_params[:10]:
                print(f"    Inf in {name}: {count} values")
    
    return has_nan, has_inf, nan_params, inf_params

def reload_from_checkpoint(unet, ema_unet, optimizer, output_dir, device, local_rank):
    """Reload model from latest checkpoint to recover from corruption."""
    checkpoint_path = os.path.join(output_dir, 'latest.pth')
    
    if not os.path.exists(checkpoint_path):
        if local_rank == 0:
            print(f"\n⚠️ ERROR: Cannot recover - no checkpoint found at {checkpoint_path}")
        return False, 0, 0
    
    try:
        if local_rank == 0:
            print(f"\n🔄 RECOVERY: Reloading from checkpoint {checkpoint_path}...")
        
        # Load to CPU first to avoid CUDA errors
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        unet.load_state_dict(checkpoint['model'], strict=False)
        ema_unet.load_state_dict(checkpoint['ema_model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        start_epoch = checkpoint.get('epoch', 0)
        start_iter = checkpoint.get('iter', 0)
        
        if local_rank == 0:
            print(f"✓ Successfully reloaded from checkpoint at epoch {start_epoch}, iter {start_iter}")
            print(f"  Model parameters have been restored. Continuing training...")
        
        # Clear CUDA cache after reload
        safe_cuda_clear()
        
        return True, start_epoch, start_iter
    except RuntimeError as e:
        if "CUDA" in str(e) and local_rank == 0:
            print(f"\n" + "="*70)
            print(f"🚨 CUDA CONTEXT CORRUPTED - CANNOT RECOVER IN-PROCESS")
            print(f"="*70)
            print(f"Error: {e}")
            print(f"")
            print(f"The CUDA context is in an unrecoverable state.")
            print(f"You must restart training to recover:")
            print(f"")
            print(f"  1. Press Ctrl+C to stop this training")
            print(f"  2. Run: python train.py --resume [other args]")
            print(f"")
            print(f"Training will resume from iter {checkpoint.get('iter', 'unknown')}.")
            print(f"="*70 + "\n")
        return False, 0, 0
    except Exception as e:
        if local_rank == 0:
            print(f"\n⚠️ ERROR: Failed to reload checkpoint: {e}")
        return False, 0, 0

def generate_test_image(unet, vae, diffusion, device, test_text, style_path, output_dir, iter_num, dataset_name='LatinBHO'):
    """
    Generate a test image to track training progress.
    
    Args:
        unet: The UNet model (use EMA model for better results)
        vae: The VAE encoder/decoder
        diffusion: Diffusion model
        device: Device to run on
        test_text: Text string to generate
        style_path: Path to style images
        output_dir: Output directory for saving images
        iter_num: Current iteration number
        dataset_name: Name of dataset ('LatinBHO' or 'IAM')
    """
    try:
        # Create progress tracking directory
        progress_dir = os.path.join(output_dir, 'progress_images')
        os.makedirs(progress_dir, exist_ok=True)
        
        # Set model to eval mode
        unet.eval()
        vae.eval()
        
        # Create generate dataset
        if dataset_name == 'LatinBHO':
            gen_dataset = LatinBHOGenerateDataset(
                style_path=style_path,
                type='test',
                ref_num=1,
                content_type='unifont'
            )
        else:
            # For IAM, import and use IAMGenerateDataset
            from data_loader.IAMDataset import IAMGenerateDataset
            gen_dataset = IAMGenerateDataset(
                style_path=style_path,
                type='test',
                ref_num=1
            )
        
        # Get style reference - only use FIRST writer for quick progress check
        data_loader = torch.utils.data.DataLoader(
            gen_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        data = next(iter(data_loader))
        
        # Only take the first style (first writer) instead of all writers
        # This makes progress image generation much faster
        # DataLoader adds batch dim: data['style'] is [1, num_writers, 1, H, W]
        # We want just first writer: [1, 1, H, W]
        style_ref = data['style'][0, 0:1]  # Remove batch dim, take first writer -> [1, 1, H, W]
        style_input = style_ref.to(device)
        
        # Get content glyphs for the text
        text_ref = gen_dataset.get_content(test_text)  # [1, T, 16, 16]
        text_ref = text_ref.to(device)  # [1, T, 16, 16] - no repeat needed for single style
        
        # Initialize random noise in latent space
        # NO scaling here - diffusion process expects unit Gaussian at t=T
        # The model will handle scaling internally during the diffusion process
        latent_h = style_ref.shape[2] // 8
        latent_w = gen_dataset.fixed_len // 8
        x = torch.randn((1, 4, latent_h, latent_w)).to(device)  # Single sample
        
        # Generate image using DDIM sampling
        with torch.no_grad():
            generated_images = diffusion.ddim_sample(
                unet,
                vae,
                style_input.shape[0],
                x,
                style_input,
                text_ref,
                sampling_timesteps=50,  # Use fewer steps for faster generation during training
                eta=0.0
            )
        
        # Save generated image
        for idx, img_tensor in enumerate(generated_images):
            img_tensor = img_tensor.clamp(0, 1)
            
            # Convert to 3-channel if needed
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            elif img_tensor.shape[0] != 3:
                print(f"Warning: Unexpected number of channels: {img_tensor.shape[0]}")
                continue
            
            # Convert to PIL Image
            im = torchvision.transforms.ToPILImage()(img_tensor)
            image = im.convert("L")
            
            # Debug: Check image before any inversion
            img_array = np.array(image)
            img_mean = img_array.mean()
            print(f"  Image before inversion: mean={img_mean:.2f} (0-255 scale)")
            
            # Only invert if image is extremely bright (mean > 240) and has low variance
            # This prevents inverting images that are legitimately bright but have content
            if img_mean > 240 and img_array.std() < 10:
                print(f"  Inverting: image is very bright (mean={img_mean:.2f}) with low variance (std={img_array.std():.2f})")
                image = ImageOps.invert(image)
                print(f"  Image after inversion: mean={np.array(image).mean():.2f}")
            else:
                print(f"  Not inverting: mean={img_mean:.2f}, std={img_array.std():.2f}")
            
            # Save image
            safe_text = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in test_text)[:50]
            filename = f"iter_{iter_num:06d}_{safe_text}.png"
            filepath = os.path.join(progress_dir, filename)
            image.save(filepath)
            print(f'Generated test image: {filepath}')
        
        # Set model back to train mode
        unet.train()
        
    except Exception as e:
        print(f'Warning: Failed to generate test image at iter {iter_num}: {e}')
        # Set model back to train mode even if generation failed
        unet.train()


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
        # Count unique writers from training data
        label_path = cfg.TRAIN.LABEL_PATH
        if os.path.exists(label_path):
            writers = set()
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Format: writer_id,image_name transcription
                        parts = line.split(',', 1)
                        if len(parts) >= 1:
                            writers.add(parts[0])
            writer_nums = len(writers)
            if local_rank == 0:
                print(f"Detected {writer_nums} unique writers from {label_path}")
        else:
            writer_nums = 1  # fallback
            if local_rank == 0:
                print(f"Warning: Could not find {label_path}, using default writer_nums=1")
    else:
        writer_nums = 496  # default
    
    """build model architecture"""
    # Verify noise_offset is 0 (default) - non-zero can cause train/inference mismatch
    noise_offset = getattr(cfg.SOLVER, 'NOISE_OFFSET', 0)
    if noise_offset != 0 and local_rank == 0:
        print(f"Warning: noise_offset={noise_offset} is non-zero. This may cause train/inference mismatch.")
    diffusion = Diffusion(noise_steps=1000, noise_offset=noise_offset, device=device)
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM, nb_classes=writer_nums).to(device)
    
    """Load and Freeze VAE Encoder"""
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()
    
    """Verify VAE encode/decode cycle"""
    if local_rank == 0:
        print("\n[VAE Test] Testing VAE encode/decode cycle...")
        with torch.no_grad():
            # Create test image in [-1, 1] range (VAE expects this)
            test_img = torch.randn(1, 3, 64, 64).to(device).clamp(-1, 1)
            
            # Encode
            latent_dist = vae.encode(test_img).latent_dist
            latent = latent_dist.sample()
            print(f"  VAE latent: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}, "
                  f"range=[{latent.min().item():.4f}, {latent.max().item():.4f}]")
            
            # Decode
            decoded = vae.decode(latent).sample
            print(f"  VAE decoded: mean={decoded.mean().item():.4f}, std={decoded.std().item():.4f}, "
                  f"range=[{decoded.min().item():.4f}, {decoded.max().item():.4f}]")
            
            # The decoded should be roughly in [-1, 1] (VAE outputs in this range)
            if decoded.min().item() < -2 or decoded.max().item() > 2:
                print(f"  ⚠️ WARNING: VAE output range unexpected! Expected roughly [-1, 1]")
            else:
                print(f"  ✓ VAE output range looks correct")
    
    """Load PyLaia readability supervisor (if enabled)"""
    pylaia_model = None
    pylaia_symbol_table = None
    pylaia_ctc_loss = None
    use_pylaia = getattr(cfg, 'PYLAIA', None) is not None and getattr(cfg.PYLAIA, 'ENABLED', False)
    
    if use_pylaia:
        pylaia_cfg = cfg.PYLAIA
        pylaia_checkpoint = pylaia_cfg.CHECKPOINT
        pylaia_syms = pylaia_cfg.SYMS_PATH
        
        # Check if files exist
        if not os.path.exists(pylaia_checkpoint):
            print(f"⚠️ WARNING: PyLaia checkpoint not found at {pylaia_checkpoint}")
            print("  Disabling PyLaia readability supervision.")
            use_pylaia = False
        elif not os.path.exists(pylaia_syms):
            print(f"⚠️ WARNING: PyLaia symbols file not found at {pylaia_syms}")
            print("  Disabling PyLaia readability supervision.")
            use_pylaia = False
        else:
            try:
                pylaia_model, pylaia_symbol_table = create_pylaia_supervisor(
                    checkpoint_path=pylaia_checkpoint,
                    syms_path=pylaia_syms,
                    device=device
                )
                # CTC Loss with blank=0 (standard CTC)
                pylaia_ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
                if local_rank == 0:
                    print(f"\n[PyLaia] Readability supervisor loaded successfully")
                    print(f"  Checkpoint: {pylaia_checkpoint}")
                    print(f"  Weight: {pylaia_cfg.WEIGHT}")
                    print(f"  Start iter: {pylaia_cfg.START_ITER}")
                    print(f"  Apply every: {pylaia_cfg.APPLY_EVERY} batches")
            except Exception as e:
                print(f"⚠️ ERROR loading PyLaia model: {e}")
                print("  Disabling PyLaia readability supervision.")
                use_pylaia = False
    
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
    if args.checkpoint:
        # Resume from specific checkpoint
        if os.path.isabs(args.checkpoint) or '/' in args.checkpoint or '\\' in args.checkpoint:
            # Full path provided
            checkpoint_path = args.checkpoint
        else:
            # Just filename, look in checkpoints directory
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', args.checkpoint)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            try:
                # Try to load state dict - use strict=False to handle writer_nums mismatch
                unet_result = unet.load_state_dict(checkpoint['model'], strict=False)
                ema_unet_result = ema_unet.load_state_dict(checkpoint['ema_model'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint.get('epoch', 0)
                start_iter = checkpoint.get('iter', 0)
                print(f'Resumed from checkpoint {checkpoint_path} at epoch {start_epoch}, iter {start_iter}')
                # Check if proxy anchors were missing (indicates writer_nums change)
                proxy_missing = any('proxy' in key.lower() for key in unet_result.missing_keys)
                if proxy_missing and local_rank == 0:
                    print('Note: Using strict=False - proxy anchors reinitialized due to writer_nums change')
            except RuntimeError as e:
                if 'size mismatch' in str(e) and 'proxy' in str(e):
                    print(f'⚠️ ERROR: Checkpoint has different number of writers than current model!')
                    print(f'  Checkpoint was saved with different writer_nums. Starting from scratch.')
                    print(f'  Delete old checkpoints to start fresh training.')
                else:
                    raise
        else:
            print(f'Warning: Checkpoint not found at {checkpoint_path}. Starting from scratch.')
    elif args.resume:
        # Resume from latest checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'latest.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            try:
                # Try to load state dict - use strict=False to handle writer_nums mismatch
                unet_result = unet.load_state_dict(checkpoint['model'], strict=False)
                ema_unet_result = ema_unet.load_state_dict(checkpoint['ema_model'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint.get('epoch', 0)
                start_iter = checkpoint.get('iter', 0)
                print(f'Resumed from latest checkpoint at epoch {start_epoch}, iter {start_iter}')
                # Check if proxy anchors were missing (indicates writer_nums change)
                proxy_missing = any('proxy' in key.lower() for key in unet_result.missing_keys)
                if proxy_missing and local_rank == 0:
                    print('Note: Using strict=False - proxy anchors reinitialized due to writer_nums change')
            except RuntimeError as e:
                if 'size mismatch' in str(e) and 'proxy' in str(e):
                    print(f'⚠️ ERROR: Checkpoint has different number of writers than current model!')
                    print(f'  Checkpoint was saved with different writer_nums. Starting from scratch.')
                    print(f'  Delete old checkpoints to start fresh training.')
                else:
                    raise
        else:
            print(f'Warning: Latest checkpoint not found at {checkpoint_path}. Starting from scratch.')
    
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
    
    """Setup learning rate scheduler to prevent collapse"""
    # Cosine annealing: gradually reduce LR to help stabilize training
    # Use a longer schedule - reduce LR more slowly to avoid interfering with learning
    total_iters_per_epoch = len(data_loader)
    total_training_iters = total_iters_per_epoch * cfg.SOLVER.EPOCHS
    # Only reduce LR to 50% of original (not 10%) to maintain learning capacity
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_training_iters, eta_min=cfg.SOLVER.BASE_LR * 0.5
    )
    # Fast-forward scheduler to current iteration if resuming
    if start_iter > 0:
        for _ in range(start_iter):
            scheduler.step()
    
    """Training loop"""
    total_iters = start_iter
    num_epochs = cfg.SOLVER.EPOCHS
    skip_count = 0  # Track total skipped batches
    consecutive_nan_batches = 0  # Track consecutive NaN/Inf batches for recovery
    max_consecutive_nan = 100  # Reload checkpoint if this many consecutive NaN batches
    recovery_failures = 0  # Track failed recovery attempts
    max_recovery_failures = 3  # Exit after this many failed recoveries
    pylaia_skip_count = 0  # Track PyLaia skips due to memory/stability concerns
    bad_images = {}  # Track images that cause NaN/Inf issues {image_name: count}
    
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
            
            # Check for NaN/Inf in raw img FIRST (before any processing)
            if torch.isnan(img).any() or torch.isinf(img).any():
                skip_count += 1
                consecutive_nan_batches += 1
                img_names = data.get('image_name', ['unknown'] * len(img))
                
                # Track which specific images have NaN/Inf
                for i in range(img.shape[0]):
                    if torch.isnan(img[i]).any() or torch.isinf(img[i]).any():
                        img_name = img_names[i] if i < len(img_names) else f'unknown_{i}'
                        bad_images[img_name] = bad_images.get(img_name, 0) + 1
                
                if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                    nan_count = torch.isnan(img).sum().item()
                    inf_count = torch.isinf(img).sum().item()
                    print(f"\n⚠️ ERROR: NaN/Inf detected in raw img at iter {total_iters}, batch_idx {batch_idx}")
                    print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                    print(f"  Image names: {img_names}")
                    # Show which specific images in batch have issues
                    bad_in_batch = [img_names[i] for i in range(img.shape[0]) 
                                   if torch.isnan(img[i]).any() or torch.isinf(img[i]).any()]
                    print(f"  Problematic images: {bad_in_batch}")
                    print(f"  Image shape: {img.shape}")
                    print(f"  Skipping this batch. (Total skipped: {skip_count}, Consecutive: {consecutive_nan_batches})")
                    
                    # Periodically report most problematic images
                    if len(bad_images) > 0 and skip_count % 100 == 0:
                        sorted_bad = sorted(bad_images.items(), key=lambda x: x[1], reverse=True)[:10]
                        print(f"  Top problematic images: {sorted_bad}")
                
                # Check if we need to recover from checkpoint
                if consecutive_nan_batches >= max_consecutive_nan:
                    if local_rank == 0:
                        print(f"\n🚨 CRITICAL: {consecutive_nan_batches} consecutive NaN batches detected!")
                        print(f"  Checking model parameters for corruption...")
                    has_nan, has_inf, nan_params, inf_params = check_model_parameters_for_nan(unet, "unet", local_rank)
                    if has_nan or has_inf:
                        if local_rank == 0:
                            print(f"  Model parameters are corrupted! Attempting recovery...")
                        success, recovered_epoch, recovered_iter = reload_from_checkpoint(
                            unet, ema_unet, optimizer, args.output_dir, device, local_rank
                        )
                        if success:
                            consecutive_nan_batches = 0  # Reset counter after recovery
                            recovery_failures = 0  # Reset failure counter on success
                            # Update epoch/iter if we went back
                            if recovered_epoch < epoch or (recovered_epoch == epoch and recovered_iter < total_iters):
                                if local_rank == 0:
                                    print(f"  Note: Training position reset to epoch {recovered_epoch}, iter {recovered_iter}")
                        else:
                            recovery_failures += 1
                            if local_rank == 0:
                                print(f"  ⚠️ Recovery failed ({recovery_failures}/{max_recovery_failures})")
                            if recovery_failures >= max_recovery_failures:
                                if local_rank == 0:
                                    print(f"\n" + "="*70)
                                    print(f"🛑 TRAINING STOPPED - UNRECOVERABLE CUDA ERROR")
                                    print(f"="*70)
                                    print(f"CUDA context is corrupted and cannot be recovered in-process.")
                                    print(f"")
                                    print(f"To resume training, run:")
                                    print(f"  python train.py --resume [your other args]")
                                    print(f"")
                                    print(f"Training will continue from the last checkpoint.")
                                    print(f"="*70 + "\n")
                                import sys
                                sys.exit(1)
                    else:
                        if local_rank == 0:
                            print(f"  Model parameters are clean. Issue is likely in input data.")
                
                optimizer.zero_grad()
                safe_cuda_clear()
                total_iters += 1  # Increment even when skipping
                continue
            
            # Input range diagnostics (first batch only, every 1000 iters)
            if total_iters == 0 or (total_iters % 1000 == 0 and batch_idx == 0):
                print(f"\n[Input Diagnostics] Raw input ranges:")
                print(f"  Image: [{img.min().item():.3f}, {img.max().item():.3f}] (expected: [-1, 1])")
                print(f"  Style: [{style.min().item():.3f}, {style.max().item():.3f}]")
                print(f"  Content: [{content.min().item():.3f}, {content.max().item():.3f}]")
                if img.min().item() < -1.1 or img.max().item() > 1.1:
                    print(f"  ⚠ WARNING: Image values outside expected range!")
            
            # Convert RGB image to grayscale for VAE (VAE expects 3 channels)
            if img.shape[1] == 3:
                # Convert to grayscale but keep 3 channels for VAE
                img_for_vae = img  # VAE can handle RGB
            else:
                img_for_vae = img.repeat(1, 3, 1, 1)
            
            # Double-check for NaN/Inf after creating img_for_vae (shouldn't happen if img is clean)
            if torch.isnan(img_for_vae).any() or torch.isinf(img_for_vae).any():
                skip_count += 1
                consecutive_nan_batches += 1
                if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                    nan_count = torch.isnan(img_for_vae).sum().item()
                    inf_count = torch.isinf(img_for_vae).sum().item()
                    img_names = data.get('image_name', ['unknown'] * len(img))
                    print(f"\n⚠️ ERROR: NaN/Inf detected in img_for_vae at iter {total_iters}, batch_idx {batch_idx}")
                    print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                    print(f"  Image names: {img_names}")
                    print(f"  This should not happen if raw img was clean! Check repeat() operation.")
                    print(f"  Skipping this batch. (Total skipped: {skip_count}, Consecutive: {consecutive_nan_batches})")
                
                # Check if we need to recover from checkpoint
                if consecutive_nan_batches >= max_consecutive_nan:
                    if local_rank == 0:
                        print(f"\n🚨 CRITICAL: {consecutive_nan_batches} consecutive NaN batches detected!")
                        has_nan, has_inf, nan_params, inf_params = check_model_parameters_for_nan(unet, "unet", local_rank)
                        if has_nan or has_inf:
                            print(f"  Model parameters are corrupted! Attempting recovery...")
                            success, recovered_epoch, recovered_iter = reload_from_checkpoint(
                                unet, ema_unet, optimizer, args.output_dir, device, local_rank
                            )
                            if success:
                                consecutive_nan_batches = 0
                                recovery_failures = 0
                            else:
                                recovery_failures += 1
                                if recovery_failures >= max_recovery_failures:
                                    print(f"\n🛑 TRAINING STOPPED - Restart with: python train.py --resume")
                                    import sys
                                    sys.exit(1)
                
                optimizer.zero_grad()
                safe_cuda_clear()
                total_iters += 1  # Increment even when skipping
                continue
            
            # Encode images to latent space
            try:
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
                    # Use sample() for stochasticity (helps prevent collapse)
                    latent_dist = vae.encode(img_for_vae).latent_dist
                    latents = latent_dist.sample()  # Use sample, not mean
                    
                    # Debug: Check raw VAE encoded latents
                    if total_iters == 0 or (total_iters % 1000 == 0 and batch_idx == 0):
                        raw_latent_mean = latents.mean().item()
                        raw_latent_std = latents.std().item()
                        raw_latent_min = latents.min().item()
                        raw_latent_max = latents.max().item()
                        print(f"  [Debug] Raw VAE encoded latents - mean={raw_latent_mean:.4f}, std={raw_latent_std:.4f}, min={raw_latent_min:.4f}, max={raw_latent_max:.4f}")
                    
                    # Check for NaN/Inf in latents after VAE encoding
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        skip_count += 1
                        if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                            print(f"\nWarning: NaN/Inf detected in VAE latents at iter {total_iters}, batch_idx {batch_idx}. Skipping this batch. (Total skipped: {skip_count})")
                        optimizer.zero_grad()
                        safe_cuda_clear()
                        total_iters += 1  # Increment even when skipping
                        continue
                    
                    # Standard Stable Diffusion scaling only (no custom normalization)
                    latents = latents * 0.18215
                    
                    # Center latents to zero mean (VAE outputs non-zero mean, but model expects zero-mean inputs)
                    # This ensures the model learns on centered latents, matching standard SD behavior
                    latents = latents - latents.mean(dim=[1,2,3], keepdim=True)  # Center per-sample
                    
                    # Debug: Check scaled latents (what model trains on)
                    if total_iters == 0 or (total_iters % 1000 == 0 and batch_idx == 0):
                        scaled_latent_mean = latents.mean().item()
                        scaled_latent_std = latents.std().item()
                        scaled_latent_min = latents.min().item()
                        scaled_latent_max = latents.max().item()
                        print(f"  [Debug] Scaled latents (for training) - mean={scaled_latent_mean:.4f}, std={scaled_latent_std:.4f}, min={scaled_latent_min:.4f}, max={scaled_latent_max:.4f}")
                        print(f"  Expected: std≈0.18215 * original_std (standard SD scaling)")
            except RuntimeError as e:
                if "CUDNN" in str(e) or "CUDA" in str(e) or "out of memory" in str(e).lower():
                    skip_count += 1
                    if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                        print(f"\nError during VAE encoding at iter {total_iters}, batch_idx {batch_idx}: {e}")
                        print(f"Clearing CUDA cache and skipping this batch... (Total skipped: {skip_count})")
                    optimizer.zero_grad()
                    safe_cuda_clear()
                    total_iters += 1  # Increment even when skipping
                    continue
                else:
                    raise
            
            # Sample timesteps
            t = diffusion.sample_timesteps(latents.shape[0], finetune=False).to(device)
            
            # Add noise
            x, noise = diffusion.noise_images(latents, t)
            
            # Forward pass - predict noise directly instead of using DDIM sampling
            # The model should predict the noise at timestep t
            predicted_noise, vertical_proxy_loss, horizontal_proxy_loss = unet(x, t, style, content, wid, tag='train')
            
            # Check for NaN/Inf in model output immediately after forward pass
            if torch.isnan(predicted_noise).any() or torch.isinf(predicted_noise).any():
                skip_count += 1
                consecutive_nan_batches += 1
                if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                    nan_count = torch.isnan(predicted_noise).sum().item()
                    inf_count = torch.isinf(predicted_noise).sum().item()
                    print(f"\n⚠️ ERROR: NaN/Inf detected in predicted_noise at iter {total_iters}, batch_idx {batch_idx}")
                    print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
                    print(f"  This indicates model corruption! (Consecutive: {consecutive_nan_batches})")
                
                # Check model parameters and recover if needed
                if consecutive_nan_batches >= max_consecutive_nan or consecutive_nan_batches % 50 == 0:
                    if local_rank == 0:
                        print(f"\n🔍 Checking model parameters for corruption...")
                    has_nan, has_inf, nan_params, inf_params = check_model_parameters_for_nan(unet, "unet", local_rank)
                    if has_nan or has_inf:
                        if local_rank == 0:
                            print(f"  🚨 Model parameters ARE corrupted!")
                            if nan_params:
                                print(f"  NaN in parameters: {nan_params[:5]}...")  # Show first 5
                            if inf_params:
                                print(f"  Inf in parameters: {inf_params[:5]}...")
                            print(f"  Attempting recovery from checkpoint...")
                        success, recovered_epoch, recovered_iter = reload_from_checkpoint(
                            unet, ema_unet, optimizer, args.output_dir, device, local_rank
                        )
                        if success:
                            consecutive_nan_batches = 0  # Reset after recovery
                            recovery_failures = 0
                        else:
                            recovery_failures += 1
                            if local_rank == 0:
                                print(f"  ⚠️ Recovery failed ({recovery_failures}/{max_recovery_failures})")
                            if recovery_failures >= max_recovery_failures:
                                if local_rank == 0:
                                    print(f"\n🛑 TRAINING STOPPED - Restart with: python train.py --resume")
                                import sys
                                sys.exit(1)
                    else:
                        if local_rank == 0 and consecutive_nan_batches % 50 == 0:
                            print(f"  ✓ Model parameters are clean. Issue may be in forward pass or inputs.")
                
                optimizer.zero_grad()
                safe_cuda_clear()
                total_iters += 1
                continue
            
            # Collapse monitoring - check if model predictions are collapsing
            if local_rank == 0 and total_iters % 500 == 0:
                with torch.no_grad():
                    # Per-sample prediction std (should be high, ~1.0 for noise)
                    pred_per_sample_std = predicted_noise.std(dim=[1,2,3])  # Std per sample
                    
                    # Check if all samples are becoming identical (collapse indicator)
                    if predicted_noise.shape[0] > 1:
                        # Cosine similarity between first two samples
                        pred_flat_0 = predicted_noise[0].flatten().unsqueeze(0)
                        pred_flat_1 = predicted_noise[1].flatten().unsqueeze(0)
                        sample_sim = torch.nn.functional.cosine_similarity(
                            pred_flat_0, pred_flat_1, dim=1
                        ).item()
                        
                        # If similarity > 0.95, model is likely collapsing
                        if sample_sim > 0.95:
                            print(f"\n⚠️ WARNING: High sample similarity ({sample_sim:.4f}) - possible collapse!")
                    else:
                        sample_sim = 0.0
                    
                    # Check if per-sample std is too low (collapse indicator)
                    min_std = pred_per_sample_std.min().item()
                    max_std = pred_per_sample_std.max().item()
                    mean_std = pred_per_sample_std.mean().item()
                    
                    # Expected std for noise predictions should be around 0.5-1.0
                    # Very low std (< 0.01) suggests model is predicting near-constant values
                    if min_std < 0.01:
                        print(f"\n⚠️ WARNING: Very low per-sample std ({min_std:.6f}) - model may be collapsing!")
                        print(f"  This is normal at iter 0, but should increase quickly. Monitor closely.")
                    
                    print(f"[Collapse Monitor] Per-sample std: "
                          f"min={min_std:.4f}, max={max_std:.4f}, mean={mean_std:.4f}, "
                          f"sample_similarity={sample_sim:.4f}")
            
            # Calculate loss between predicted noise and actual noise
            noise_loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Check for NaN/Inf in noise_loss early
            if torch.isnan(noise_loss) or torch.isinf(noise_loss):
                skip_count += 1
                consecutive_nan_batches += 1
                if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                    print(f"\n⚠️ ERROR: NaN/Inf detected in noise_loss at iter {total_iters}")
                    print(f"  pred_noise range: [{predicted_noise.min().item():.4f}, {predicted_noise.max().item():.4f}]")
                    print(f"  noise range: [{noise.min().item():.4f}, {noise.max().item():.4f}]")
                optimizer.zero_grad()
                safe_cuda_clear()
                total_iters += 1
                continue
            
            # Disable proxy losses for single writer class (they're degenerate)
            if writer_nums == 1:
                vertical_proxy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                horizontal_proxy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # Handle NaN/Inf in proxy losses
                if torch.isnan(vertical_proxy_loss) or torch.isinf(vertical_proxy_loss):
                    if local_rank == 0 and total_iters % 1000 == 0:
                        print(f"\nNote: vertical_proxy_loss is NaN/Inf, setting to 0")
                    vertical_proxy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                if torch.isnan(horizontal_proxy_loss) or torch.isinf(horizontal_proxy_loss):
                    if local_rank == 0 and total_iters % 1000 == 0:
                        print(f"\nNote: horizontal_proxy_loss is NaN/Inf, setting to 0")
                    horizontal_proxy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Output diversity regularization to prevent mode collapse
            diversity_loss = torch.tensor(0.0, device=device, requires_grad=True)
            diversity_weight = getattr(cfg.SOLVER, 'DIVERSITY_WEIGHT', 0.0)
            diversity_start_iter = getattr(cfg.SOLVER, 'DIVERSITY_START_ITER', 0)
            
            # Adaptive diversity: only apply after warmup period, then gradually increase
            if diversity_weight > 0 and predicted_noise.shape[0] > 1 and total_iters >= diversity_start_iter:
                # Gradually ramp up diversity weight from 0 to full weight over 5000 iterations
                if total_iters < diversity_start_iter + 5000:
                    ramp_factor = (total_iters - diversity_start_iter) / 5000.0
                    current_diversity_weight = diversity_weight * ramp_factor
                else:
                    current_diversity_weight = diversity_weight
            else:
                current_diversity_weight = 0.0
            
            if current_diversity_weight > 0 and predicted_noise.shape[0] > 1:  # Need batch size > 1
                # Simpler diversity regularization: encourage batch variance
                pred_flat = predicted_noise.view(predicted_noise.shape[0], -1)  # [B, C*H*W]
                batch_var = pred_flat.var(dim=0).mean()  # Average variance across features
                # Target variance of 0.5 - penalize if below this
                diversity_loss = torch.clamp(0.5 - batch_var, min=0.0)
                
                # Log diversity loss periodically
                if local_rank == 0 and total_iters % 500 == 0:
                    print(f"\n[Diversity] weight={current_diversity_weight:.4f}, batch_var={batch_var.item():.6f}, "
                          f"diversity_loss={diversity_loss.item():.6f}")
            
            # PyLaia CTC Loss for readability supervision
            ctc_loss = torch.tensor(0.0, device=device, requires_grad=True)
            ctc_weight = 0.0
            
            if use_pylaia and pylaia_model is not None:
                pylaia_cfg = cfg.PYLAIA
                pylaia_start_iter = getattr(pylaia_cfg, 'START_ITER', 10000)
                pylaia_apply_every = getattr(pylaia_cfg, 'APPLY_EVERY', 4)
                
                # Skip PyLaia if we've had recent NaN issues or memory is too high
                skip_pylaia_this_iter = consecutive_nan_batches > 0
                
                # Also skip if GPU memory is critically high (>85%)
                if not skip_pylaia_this_iter and should_skip_heavy_operations(threshold=0.85):
                    skip_pylaia_this_iter = True
                    pylaia_skip_count += 1
                    if local_rank == 0 and pylaia_skip_count <= 5:
                        print(f"\n[PyLaia] Skipping due to high GPU memory usage")
                
                # Only apply after start_iter and every N batches (for efficiency)
                if total_iters >= pylaia_start_iter and batch_idx % pylaia_apply_every == 0 and not skip_pylaia_this_iter:
                    ctc_weight = pylaia_cfg.WEIGHT
                    
                    try:
                        # Get transcriptions from batch
                        transcriptions = data.get('transcr', None)
                        img_names = data.get('image_name', None)
                        
                        if transcriptions is not None:
                            # Use reduced batch size for PyLaia CTC to save memory
                            # Default to 2 samples (configurable via PYLAIA.BATCH_SIZE)
                            pylaia_batch_size = getattr(pylaia_cfg, 'BATCH_SIZE', 2)
                            full_batch_size = x.shape[0]
                            
                            # Select a subset of samples for CTC computation
                            if pylaia_batch_size < full_batch_size:
                                # Randomly select indices for this iteration
                                indices = torch.randperm(full_batch_size)[:pylaia_batch_size]
                                x_subset = x[indices]
                                t_subset = t[indices]
                                predicted_noise_subset = predicted_noise[indices]
                                transcriptions_subset = [transcriptions[i] for i in indices.tolist()]
                                if img_names is not None:
                                    img_names_subset = [img_names[i] for i in indices.tolist()]
                                else:
                                    img_names_subset = None
                            else:
                                x_subset = x
                                t_subset = t
                                predicted_noise_subset = predicted_noise
                                transcriptions_subset = transcriptions
                                img_names_subset = img_names
                            
                            # Calculate predicted x_0 (clean latent estimate) from noisy x and predicted noise
                            # x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * noise
                            # => x_0 = (x_t - sqrt(1 - alpha_hat) * noise) / sqrt(alpha_hat)
                            alpha_hat = diffusion.alpha_hat[t_subset][:, None, None, None]
                            
                            # Safety: clamp alpha_hat to avoid division by very small numbers
                            alpha_hat_safe = alpha_hat.clamp(min=1e-6)
                            
                            x_0_pred = (x_subset - (1 - alpha_hat).sqrt() * predicted_noise_subset) / alpha_hat_safe.sqrt()
                            
                            # Clamp x_0 to reasonable range (tighter than before)
                            x_0_pred = x_0_pred.clamp(-3, 3)
                            
                            # Check for NaN before proceeding (early exit)
                            if torch.isnan(x_0_pred).any() or torch.isinf(x_0_pred).any():
                                if local_rank == 0 and total_iters % 500 == 0:
                                    print(f"\n[PyLaia] Skipping CTC loss - NaN/Inf in x_0_pred")
                                    if img_names_subset:
                                        print(f"  Affected images: {img_names_subset}")
                                ctc_weight = 0.0
                                ctc_loss = torch.tensor(0.0, device=device, requires_grad=True)
                            else:
                                # Unscale and un-center latents for VAE
                                latents_for_vae = x_0_pred / 0.18215
                                
                                # Clamp latents to prevent VAE explosion
                                latents_for_vae = latents_for_vae.clamp(-20, 20)
                                
                                # Free memory before VAE decode (most memory-intensive operation)
                                safe_cuda_clear()
                                
                                # Decode through VAE to get predicted image
                                # Note: We need gradients to flow, so don't use torch.no_grad()
                                decoded_img = vae.decode(latents_for_vae).sample
                                
                                # Check for NaN in VAE output
                                if torch.isnan(decoded_img).any() or torch.isinf(decoded_img).any():
                                    if local_rank == 0 and total_iters % 500 == 0:
                                        print(f"\n[PyLaia] Skipping CTC loss - NaN/Inf in VAE decoded image")
                                    ctc_weight = 0.0
                                    ctc_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                else:
                                    # Convert to grayscale [B, 3, H, W] -> [B, 1, H, W]
                                    decoded_gray = 0.299 * decoded_img[:, 0:1, :, :] + \
                                                  0.587 * decoded_img[:, 1:2, :, :] + \
                                                  0.114 * decoded_img[:, 2:3, :, :]
                                    
                                    # UPSCALE from DiffBrush (64px) to PyLaia (128px) height
                                    # DiffBrush uses 64px height, PyLaia expects 128px
                                    # Scale factor: 128/64 = 2x
                                    pylaia_height = getattr(pylaia_cfg, 'INPUT_HEIGHT', 128)
                                    current_height = decoded_gray.shape[2]
                                    current_width = decoded_gray.shape[3]
                                    scale_factor = pylaia_height / current_height
                                    new_width = int(current_width * scale_factor)
                                    
                                    # Log resize info periodically
                                    if local_rank == 0 and total_iters % 2000 == 0:
                                        print(f"\n[PyLaia] Upscaling image: {current_height}x{current_width} -> {pylaia_height}x{new_width} (scale={scale_factor:.2f}x)")
                                    
                                    decoded_resized = F.interpolate(
                                        decoded_gray, 
                                        size=(pylaia_height, new_width), 
                                        mode='bilinear', 
                                        align_corners=False
                                    )
                                    
                                    # Normalize for PyLaia: convert VAE output [-1, 1] to [0, 1]
                                    # PyLaia was trained on images normalized to [0, 1]
                                    decoded_normalized = (decoded_resized + 1) / 2
                                    decoded_normalized = decoded_normalized.clamp(0, 1)
                                    
                                    # Normalization sanity check (first activation only)
                                    if total_iters == pylaia_start_iter and local_rank == 0:
                                        print(f"\n[PyLaia Normalization Check]")
                                        print(f"  VAE output (decoded_img): range=[{decoded_img.min().item():.3f}, {decoded_img.max().item():.3f}], mean={decoded_img.mean().item():.3f}")
                                        print(f"  After grayscale: range=[{decoded_gray.min().item():.3f}, {decoded_gray.max().item():.3f}]")
                                        print(f"  After resize ({decoded_gray.shape[2]}→{decoded_resized.shape[2]}px): range=[{decoded_resized.min().item():.3f}, {decoded_resized.max().item():.3f}]")
                                        print(f"  After [0,1] normalize: range=[{decoded_normalized.min().item():.3f}, {decoded_normalized.max().item():.3f}], mean={decoded_normalized.mean().item():.3f}")
                                        print(f"  Expected for PyLaia: [0, 1] with mean ~0.5 for typical document images")
                                    
                                    # Pass through frozen PyLaia model
                                    # PyLaia returns log probabilities [T, B, num_classes]
                                    pylaia_output = pylaia_model(decoded_normalized)
                                    
                                    # Free memory after PyLaia forward pass
                                    safe_cuda_clear()
                                    
                                    # Prepare targets for CTC loss (using subset transcriptions)
                                    ctc_batch_size = len(transcriptions_subset)
                                    target_indices = []
                                    target_lengths = []
                                    
                                    # Log batch size info periodically
                                    if local_rank == 0 and total_iters == pylaia_start_iter:
                                        print(f"\n[PyLaia] Using batch_size={ctc_batch_size} for CTC (full batch={full_batch_size})")
                                    
                                    for transcr in transcriptions_subset:
                                        indices = pylaia_symbol_table.encode(transcr)
                                        target_indices.extend(indices)
                                        target_lengths.append(len(indices))
                                    
                                    targets = torch.tensor(target_indices, dtype=torch.long, device=device)
                                    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
                                    
                                    # Input lengths (time steps from PyLaia output)
                                    input_lengths = torch.full(
                                        (ctc_batch_size,), 
                                        pylaia_output.shape[0], 
                                        dtype=torch.long, 
                                        device=device
                                    )
                                    
                                    # Calculate CTC loss
                                    ctc_loss = pylaia_ctc_loss(
                                        pylaia_output,  # [T, B, C] log probs
                                        targets,        # [sum(target_lengths)]
                                        input_lengths,  # [B]
                                        target_lengths  # [B]
                                    )
                                    
                                    # Handle NaN/Inf
                                    if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                                        ctc_loss = torch.tensor(0.0, device=device, requires_grad=True)
                                        if local_rank == 0 and total_iters % 500 == 0:
                                            print(f"\n[PyLaia] CTC loss was NaN/Inf, setting to 0")
                                    
                                    # ========================================
                                    # GRADIENT VERIFICATION (Critical Sanity Check)
                                    # Run once when PyLaia first activates to verify gradients flow
                                    # ========================================
                                    if total_iters == pylaia_start_iter and local_rank == 0:
                                        print(f"\n{'='*60}")
                                        print(f"[PyLaia Gradient Check] First activation at iter {total_iters}")
                                        print(f"  CTC Loss: {ctc_loss.item():.4f}")
                                        print(f"  x_0_pred requires_grad: {x_0_pred.requires_grad}")
                                        print(f"  latents_for_vae requires_grad: {latents_for_vae.requires_grad}")
                                        print(f"  decoded_img requires_grad: {decoded_img.requires_grad}")
                                        print(f"  pylaia_output requires_grad: {pylaia_output.requires_grad}")
                                        
                                        # Test gradient flow by doing a backward pass on CTC loss alone
                                        if ctc_loss.requires_grad and x_0_pred.requires_grad:
                                            # Compute gradient of ctc_loss w.r.t. x_0_pred
                                            test_grad = torch.autograd.grad(
                                                ctc_loss, x_0_pred, 
                                                retain_graph=True, 
                                                allow_unused=True
                                            )[0]
                                            if test_grad is not None:
                                                grad_magnitude = test_grad.abs().mean().item()
                                                print(f"  ✓ Gradient flows! x_0_pred grad magnitude: {grad_magnitude:.6f}")
                                                if grad_magnitude < 1e-10:
                                                    print(f"    ⚠️ WARNING: Gradient is very small!")
                                            else:
                                                print(f"  ❌ ERROR: No gradient from CTC loss to x_0_pred!")
                                                print(f"     The VAE decoder may be breaking the gradient graph.")
                                        else:
                                            print(f"  ❌ ERROR: CTC loss or x_0_pred doesn't require grad!")
                                        print(f"{'='*60}\n")
                                    
                                    # Log CTC loss periodically
                                    if local_rank == 0 and total_iters % 500 == 0:
                                        print(f"\n[PyLaia CTC] loss={ctc_loss.item():.4f}, weight={ctc_weight:.4f}")
                    
                    except Exception as e:
                        if local_rank == 0 and total_iters % 1000 == 0:
                            print(f"\n[PyLaia] Error calculating CTC loss: {e}")
                        ctc_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Total loss - skip proxy losses for single writer
            if writer_nums == 1:
                total_loss = noise_loss + current_diversity_weight * diversity_loss + ctc_weight * ctc_loss
            else:
                total_loss = noise_loss + 0.1 * vertical_proxy_loss + 0.1 * horizontal_proxy_loss + current_diversity_weight * diversity_loss + ctc_weight * ctc_loss
            
            # Check for NaN/Inf in total loss before backward pass
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                skip_count += 1
                if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                    print(f"\nWarning: NaN/Inf detected in total_loss at iter {total_iters}, batch_idx {batch_idx}. Skipping this batch. (Total skipped: {skip_count})")
                optimizer.zero_grad()
                # Clear CUDA cache and synchronize to prevent GPU state corruption
                safe_cuda_clear()
                total_iters += 1  # Increment even when skipping
                continue
            
            # Backward pass with error handling for cuDNN/CUDA errors
            optimizer.zero_grad()
            try:
                total_loss.backward()
            except RuntimeError as e:
                if "CUDNN" in str(e) or "CUDA" in str(e) or "out of memory" in str(e).lower():
                    skip_count += 1
                    if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                        print(f"\nError during backward pass at iter {total_iters}, batch_idx {batch_idx}: {e}")
                        print(f"Clearing CUDA cache and skipping this batch... (Total skipped: {skip_count})")
                    optimizer.zero_grad()
                    safe_cuda_clear()
                    total_iters += 1  # Increment even when skipping
                    continue
                else:
                    raise
            
            # Free GPU memory after backward pass (especially after CTC loss)
            if ctc_weight > 0:
                safe_cuda_clear()
            
            # Gradient check after first backward pass (verify output layer gets gradients)
            if total_iters == 0 and local_rank == 0:
                print("\n[Gradient Check] Checking if gradients flow to output layer...")
                output_layer_has_grad = False
                for name, param in unet.named_parameters():
                    if 'out.2' in name and param.grad is not None:  # The conv layer in self.out
                        grad_norm = param.grad.norm().item()
                        param_norm = param.norm().item()
                        print(f"  {name}: param_norm={param_norm:.6f}, grad_norm={grad_norm:.6f}")
                        if grad_norm < 1e-10:
                            print(f"    ⚠️ WARNING: Very small gradient! This may indicate a problem.")
                        else:
                            output_layer_has_grad = True
                
                if not output_layer_has_grad:
                    print("  ⚠️ WARNING: No gradients found in output layer! Check model architecture.")
                else:
                    print("  ✓ Gradients are flowing to output layer")
            
            # Gradient clipping and compute gradient norm
            if cfg.SOLVER.GRAD_L2_CLIP > 0:
                grad_norm = clip_grad_norm_(unet.parameters(), cfg.SOLVER.GRAD_L2_CLIP)
                # Check if gradients exploded after clipping
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    skip_count += 1
                    if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                        print(f"\nWarning: NaN/Inf in grad_norm at iter {total_iters}, batch_idx {batch_idx}. Skipping optimizer step. (Total skipped: {skip_count})")
                    total_iters += 1  # Increment even when skipping optimizer step
                    continue
            else:
                # Calculate grad norm even if not clipping
                parameters = [p for p in unet.parameters() if p.grad is not None]
                if parameters:
                    grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters]))
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        skip_count += 1
                        if local_rank == 0 and (skip_count <= 5 or skip_count % 50 == 0):
                            print(f"\nWarning: NaN/Inf in grad_norm at iter {total_iters}, batch_idx {batch_idx}. Skipping optimizer step. (Total skipped: {skip_count})")
                        total_iters += 1  # Increment even when skipping optimizer step
                        continue
                else:
                    grad_norm = 0.0
            
            # Log statistics periodically to diagnose training
            if local_rank == 0 and total_iters % 500 == 0:
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
            
            # Reset consecutive NaN counter on successful batch
            consecutive_nan_batches = 0
            
            # Periodic memory cleanup every 50 iterations to prevent buildup
            if total_iters % 50 == 0:
                safe_cuda_clear()
            
            # Update learning rate scheduler
            scheduler.step()
            
            # Update EMA
            ema.step_ema(ema_unet, unet, step_start_ema=2000)
            
            # Successfully processed batch - reset skip counter
            skip_count = 0
            total_iters += 1
            
            # Update progress bar
            if local_rank == 0:
                # Show more precision for proxy losses to see if they're just very small
                v_proxy_val = vertical_proxy_loss.item()
                h_proxy_val = horizontal_proxy_loss.item()
                ctc_loss_val = ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else ctc_loss
                postfix_dict = {
                    'loss': f'{total_loss.item():.4f}',
                    'noise': f'{noise_loss.item():.4f}',
                    'v_proxy': f'{v_proxy_val:.6f}' if v_proxy_val > 0 else '0.000000',
                    'h_proxy': f'{h_proxy_val:.6f}' if h_proxy_val > 0 else '0.000000'
                }
                # Add CTC loss to progress bar when using PyLaia
                if use_pylaia and ctc_weight > 0:
                    postfix_dict['ctc'] = f'{ctc_loss_val:.4f}'
                progress_bar.set_postfix(postfix_dict)
            
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
                
                # Report problematic images if any
                if len(bad_images) > 0:
                    sorted_bad = sorted(bad_images.items(), key=lambda x: x[1], reverse=True)
                    print(f'\n[Bad Images Report] {len(bad_images)} unique images with NaN/Inf:')
                    for img_name, count in sorted_bad[:20]:  # Show top 20
                        print(f'  {img_name}: {count} occurrences')
                    
                    # Save bad images list to file
                    bad_images_path = os.path.join(args.output_dir, 'bad_images.txt')
                    with open(bad_images_path, 'w') as f:
                        f.write(f'# Bad images report at iter {total_iters}\n')
                        f.write(f'# Total unique: {len(bad_images)}, Total skipped batches: {skip_count}\n')
                        for img_name, count in sorted_bad:
                            f.write(f'{img_name}\t{count}\n')
                    print(f'  Saved full list to {bad_images_path}')
                
                # Clean up old checkpoints, keeping only the 3 most recent
                checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
                checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith('checkpoint_') and f.endswith('.pth')]
                
                # Extract iteration numbers and sort
                checkpoint_iters = []
                for f in checkpoint_files:
                    try:
                        iter_num = int(f.replace('checkpoint_', '').replace('.pth', ''))
                        checkpoint_iters.append((iter_num, f))
                    except ValueError:
                        continue
                
                # Sort by iteration number (descending) and keep only the 3 most recent
                checkpoint_iters.sort(reverse=True)
                if len(checkpoint_iters) > 3:
                    # Delete older checkpoints
                    for iter_num, filename in checkpoint_iters[3:]:
                        old_checkpoint_path = os.path.join(checkpoints_dir, filename)
                        try:
                            os.remove(old_checkpoint_path)
                            print(f'Deleted old checkpoint: {filename}')
                        except OSError as e:
                            print(f'Warning: Could not delete old checkpoint {filename}: {e}')
            
            # Generate test image every 200 iterations (separate from checkpoint saving)
            if total_iters % 200 == 0 and local_rank == 0:
                if 'LatinBHO' in args.cfg_file or getattr(cfg, 'DATASET', None) == 'LatinBHO':
                    dataset_name = 'LatinBHO'
                    test_text = "Et p'd'cus Ioh'es Knyght p' Will'm Byngham attorn' suum uen' & defend' uim & iniur' quando &c Et dicit q'd p'd'ci"
                else:
                    dataset_name = 'IAM'
                    test_text = "The quick brown fox jumps over the lazy dog"
                
                generate_test_image(
                    unet=ema_unet,
                    vae=vae,
                    diffusion=diffusion,
                    device=device,
                    test_text=test_text,
                    style_path=cfg.TRAIN.STYLE_PATH,
                    output_dir=args.output_dir,
                    iter_num=total_iters,
                    dataset_name=dataset_name
                )
        
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
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Resume from specific checkpoint file (e.g., checkpoint_1000.pth or full path)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with anomaly detection (slower but helps find NaN sources)')
    args = parser.parse_args()
    
    # Enable anomaly detection in debug mode
    if args.debug:
        torch.autograd.set_detect_anomaly(True)
        print("⚠️ DEBUG MODE: Anomaly detection enabled (training will be slower)")
    
    main(args)

