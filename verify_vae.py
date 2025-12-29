#!/usr/bin/env python3
"""
Verify that the VAE is working correctly by encoding and decoding a test image.
"""

import torch
from diffusers import AutoencoderKL
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def test_vae(vae_path="model_zoo", device="cuda"):
    """Test VAE encoding and decoding."""
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    vae = vae.to(device)
    vae.eval()
    
    # Create a simple test image (white square with black border)
    test_image = torch.ones(1, 3, 64, 64) * 0.5  # Gray image in [-1, 1] range
    test_image = test_image.to(device)
    
    print(f"\nTest image stats: mean={test_image.mean().item():.4f}, std={test_image.std().item():.4f}")
    print(f"  Expected: mean≈0, std≈0 (for 0.5 in [-1,1] range)")
    
    # Encode
    print("\nEncoding image...")
    with torch.no_grad():
        latent_dist = vae.encode(test_image).latent_dist
        
        # Check distribution parameters
        dist_mean = latent_dist.mean
        dist_logvar = latent_dist.logvar
        dist_std = torch.exp(0.5 * dist_logvar)
        
        print(f"VAE latent distribution parameters:")
        print(f"  Mean: mean={dist_mean.mean().item():.4f}, std={dist_mean.std().item():.4f}")
        print(f"  Std: mean={dist_std.mean().item():.4f}, std={dist_std.std().item():.4f}")
        
        # Try both mean and sample
        latents_mean = latent_dist.mean
        latents_sample = latent_dist.sample()
        
        print(f"\nUsing .mean():")
        print(f"  Latents: mean={latents_mean.mean().item():.4f}, std={latents_mean.std().item():.4f}")
        print(f"  Min={latents_mean.min().item():.4f}, Max={latents_mean.max().item():.4f}")
        
        print(f"\nUsing .sample():")
        print(f"  Latents: mean={latents_sample.mean().item():.4f}, std={latents_sample.std().item():.4f}")
        print(f"  Min={latents_sample.min().item():.4f}, Max={latents_sample.max().item():.4f}")
        
        # Scale and decode
        latents_scaled = latents_mean * 0.18215
        print(f"\nScaled latents (mean):")
        print(f"  mean={latents_scaled.mean().item():.4f}, std={latents_scaled.std().item():.4f}")
        
        # Decode
        latents_for_decode = 1 / 0.18215 * latents_scaled
        decoded = vae.decode(latents_for_decode).sample
        
        print(f"\nDecoded image:")
        print(f"  Raw: mean={decoded.mean().item():.4f}, std={decoded.std().item():.4f}")
        
        decoded_normalized = (decoded / 2 + 0.5).clamp(0, 1)
        print(f"  Normalized: mean={decoded_normalized.mean().item():.4f}, std={decoded_normalized.std().item():.4f}")
        
        # Check reconstruction quality
        mse = torch.nn.functional.mse_loss(decoded_normalized, test_image / 2 + 0.5)
        print(f"\nReconstruction MSE: {mse.item():.6f}")
        print(f"  (Lower is better, should be < 0.1 for good VAE)")
    
    print("\n✓ VAE test complete")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_path", type=str, default="model_zoo", help="Path to VAE")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()
    
    test_vae(args.vae_path, args.device)

