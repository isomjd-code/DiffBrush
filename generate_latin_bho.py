import os
import argparse
import random
from parse_config import cfg, cfg_from_file, assert_and_infer_cfg
import torch
from data_loader.LatinBHODataset import LatinBHOGenerateDataset
from models.unet import UNetModel
from diffusers import AutoencoderKL
from models.diffusion import Diffusion
import torchvision
from tqdm import tqdm
from utils.util import fix_seed
import numpy as np
from PIL import Image, ImageOps

# For LatinBHO, we have 1 writer
WRITER_NUMS = 1

def main(args):
    """ load config file into cfg"""
    cfg_from_file(args.cfg_file)
    assert_and_infer_cfg()
    """fix the random seed"""
    fix_seed(cfg.TRAIN.SEED)
    
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    """build model architecture"""
    diffusion = Diffusion(noise_steps=1000, device=device)
    unet = UNetModel(in_channels=cfg.MODEL.IN_CHANNELS, model_channels=cfg.MODEL.EMB_DIM, 
                     out_channels=cfg.MODEL.OUT_CHANNELS, num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS, 
                     attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=cfg.MODEL.NUM_HEADS, 
                     context_dim=cfg.MODEL.EMB_DIM, nb_classes=WRITER_NUMS).to(device)
    
    """load pretrained model"""
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        checkpoint = torch.load(args.pretrained_model, map_location=device)
        # Handle both full checkpoint dict and state_dict
        if isinstance(checkpoint, dict):
            # Use EMA model if available (better for generation), otherwise use regular model
            if 'ema_model' in checkpoint:
                unet.load_state_dict(checkpoint['ema_model'])
                print(f'Loaded EMA model from {args.pretrained_model}')
                if 'iter' in checkpoint:
                    print(f'Checkpoint is from iteration {checkpoint["iter"]}')
            elif 'model' in checkpoint:
                unet.load_state_dict(checkpoint['model'])
                print(f'Loaded regular model from {args.pretrained_model}')
                if 'iter' in checkpoint:
                    print(f'Checkpoint is from iteration {checkpoint["iter"]}')
            else:
                raise ValueError("Checkpoint dict must contain 'model' or 'ema_model' key")
        else:
            unet.load_state_dict(checkpoint)
            print(f'Loaded model from {args.pretrained_model}')
    else:
        raise IOError(f'Checkpoint not found: {args.pretrained_model}')
    unet.eval()

    """Load and Freeze VAE Encoder"""
    if os.path.isdir(args.stable_dif_path):
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae", local_files_only=True)
    else:
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    vae = vae.to(device)
    vae.requires_grad_(False)
    vae.eval()

    """Prepare dataset for generation"""
    dataset = LatinBHOGenerateDataset(
        style_path=cfg.TRAIN.STYLE_PATH,
        type='test',
        ref_num=1,
        content_type='unifont'
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    """Generate images"""
    # If text file is provided, read texts from file
    if args.text_file:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
    # Otherwise use the provided text
    elif args.text:
        texts = [args.text]
    else:
        raise ValueError("Either --text or --text_file must be provided")

    with torch.no_grad():
        for text in tqdm(texts, desc="Generating images"):
            # Get style reference from dataset
            data = next(iter(data_loader))
            style_ref = data['style'][0]  # [B, 1, H, W] - batch of style images from collate
            wid_list = data['wid']
            
            style_input = style_ref.to(device)  # [B, 1, H, W]
            
            # Get content glyphs for the text
            text_ref = dataset.get_content(text)  # [1, T, 16, 16]
            text_ref = text_ref.to(device).repeat(style_input.shape[0], 1, 1, 1)  # [B, T, 16, 16]
            
            # Initialize random noise in latent space
            # Latent space is 8x smaller than image space
            # style_ref shape from GenerateDataset collate: [B, C, H, W] where C=1
            # shape[0] = B, shape[1] = C (1), shape[2] = H (height), shape[3] = W (width)
            # Match the original generate.py which uses shape[2] for height
            latent_h = style_ref.shape[2] // 8  # Height dimension (H)
            latent_w = dataset.fixed_len // 8   # Width dimension (fixed_len = 2048, so 2048//8 = 256)
            print(f"Debug: style_ref.shape = {style_ref.shape}, latent_h = {latent_h}, latent_w = {latent_w}")
            print(f"Debug: text length = {len(text)}, text_ref.shape = {text_ref.shape}")
            # Initialize noise - NO scaling here, diffusion process expects unit Gaussian at t=T
            # The model will handle scaling internally during the diffusion process
            x = torch.randn((style_input.shape[0], 4, latent_h, latent_w)).to(device)
            
            # Generate image using DDIM sampling
            # Note: ddim_sample expects: model, vae, n (batch size), x (latents), styles, content, sampling_timesteps, eta
            print(f"Debug: Starting DDIM sampling with {args.sampling_timesteps} steps")
            print(f"Debug: Initial noise stats - mean={x.mean().item():.4f}, std={x.std().item():.4f}, min={x.min().item():.4f}, max={x.max().item():.4f}")
            
            generated_images = diffusion.ddim_sample(
                unet, 
                vae, 
                style_input.shape[0],  # batch size
                x,  # initial noise latents
                style_input,  # style images [B, 1, H, W]
                text_ref,  # content glyphs [B, T, 16, 16]
                sampling_timesteps=args.sampling_timesteps, 
                eta=args.eta
            )
            
            print(f"Debug: Generated image stats - mean={generated_images.mean().item():.4f}, std={generated_images.std().item():.4f}, min={generated_images.min().item():.4f}, max={generated_images.max().item():.4f}")
            
            # Save generated images
            for idx, img_tensor in enumerate(generated_images):
                # Debug: Check image tensor stats before conversion
                print(f"Debug: Image tensor stats - mean={img_tensor.mean().item():.4f}, std={img_tensor.std().item():.4f}, min={img_tensor.min().item():.4f}, max={img_tensor.max().item():.4f}")
                print(f"Debug: Image tensor shape = {img_tensor.shape}")
                
                # Check if image is all black (or very dark)
                if img_tensor.mean().item() < 0.01:
                    print(f"Warning: Generated image is very dark (mean={img_tensor.mean().item():.4f}). This suggests model collapse or VAE decode issue.")
                
                # Convert tensor to PIL Image
                img_tensor = img_tensor.clamp(0, 1)
                
                # If image is grayscale (single channel), convert to 3-channel for ToPILImage
                if img_tensor.shape[0] == 1:
                    img_tensor = img_tensor.repeat(3, 1, 1)
                elif img_tensor.shape[0] == 3:
                    pass  # Already RGB
                else:
                    print(f"Warning: Unexpected number of channels: {img_tensor.shape[0]}")
                
                im = torchvision.transforms.ToPILImage()(img_tensor)
                # Convert to grayscale
                image = im.convert("L")
                
                # Debug: Check PIL image stats
                import numpy as np
                img_array = np.array(image)
                print(f"Debug: PIL image stats - mean={img_array.mean():.2f}, std={img_array.std():.2f}, min={img_array.min()}, max={img_array.max()}")
                
                # Check if image is mostly white (mean > 200) - might need inversion
                if img_array.mean() > 200:
                    print(f"Note: Image is very bright (mean={img_array.mean():.2f}). Text might be white on black. Trying inversion...")
                    from PIL import ImageOps
                    image_inverted = ImageOps.invert(image)
                    # Save both versions
                    safe_text = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text)[:50]
                    filename_inverted = f"generated_{safe_text}_{idx}_inverted.png"
                    filepath_inverted = os.path.join(args.save_dir, filename_inverted)
                    image_inverted.save(filepath_inverted)
                    print(f"Also saved inverted version: {filepath_inverted}")
                
                # Save image
                # Sanitize text for filename
                safe_text = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in text)[:50]
                filename = f"generated_{safe_text}_{idx}.png"
                filepath = os.path.join(args.save_dir, filename)
                image.save(filepath)
                print(f"Saved: {filepath}")


if __name__ == '__main__':
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Generate handwritten text images using trained DiffBrush model')
    parser.add_argument('--cfg_file', dest='cfg_file', default='configs/LatinBHO.yml',
                        help='Config file for the model')
    parser.add_argument('--save_dir', dest='save_dir', default='generated_images',
                        help='Directory to save generated images')
    parser.add_argument('--pretrained_model', dest='pretrained_model', required=True,
                        help='Path to trained model checkpoint (e.g., outputs/LatinBHO/checkpoints/checkpoint_3000.pth or outputs/LatinBHO/latest.pth)')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to generate (e.g., "Hello world")')
    parser.add_argument('--text_file', type=str, default=None,
                        help='Path to text file with one text per line')
    parser.add_argument('--device', type=str, default='cuda', help='Device for generation')
    parser.add_argument('--stable_dif_path', type=str, default='model_zoo',
                        help='Path to Stable Diffusion VAE model directory')
    parser.add_argument('--sampling_timesteps', type=int, default=50,
                        help='Number of DDIM sampling steps (more steps = higher quality, slower)')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0.0 = deterministic DDIM, 1.0 = DDPM)')
    args = parser.parse_args()
    
    main(args)

