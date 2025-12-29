#!/bin/bash
# Start fresh training for Latin BHO dataset with stable config
# This script starts training from scratch (no resume)

cd "$(dirname "$0")"  # Change to diffbrush directory

# Remove existing checkpoints and progress images to start completely fresh
echo "Removing existing checkpoints and progress images to start fresh..."
rm -rf outputs/LatinBHO/checkpoints/*.pth outputs/LatinBHO/latest.pth outputs/LatinBHO/progress_images/*.png

# Start training with stable config (batch size 8, checkpoint every 500 iters)
python train.py \
    --cfg_file configs/LatinBHO_stable.yml \
    --output_dir outputs/LatinBHO \
    --stable_dif_path model_zoo \
    --device cuda

