#!/bin/bash
# Start fresh training for Latin BHO dataset with stable config
# This script starts training from scratch (no resume) with all fixes applied

cd "$(dirname "$0")"  # Change to diffbrush directory

# Remove existing checkpoints and progress images to start completely fresh
echo "Removing existing checkpoints and progress images to start fresh..."
rm -rf outputs/LatinBHO/checkpoints/*.pth 
rm -f outputs/LatinBHO/latest.pth 
rm -rf outputs/LatinBHO/progress_images/*.png

echo "Starting fresh training with all fixes applied..."
echo "  - Standard SD scaling (no double normalization)"
echo "  - Consistent train/inference scaling"
echo "  - Stochastic VAE encoding (sample instead of mean)"
echo "  - Disabled proxy losses for single writer"
echo "  - Simplified diversity regularization"
echo "  - Collapse monitoring enabled"
echo ""

# Start training with stable config
python train.py \
    --cfg_file configs/LatinBHO_stable.yml \
    --output_dir outputs/LatinBHO \
    --stable_dif_path model_zoo \
    --device cuda

