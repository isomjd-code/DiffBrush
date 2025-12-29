#!/bin/bash
# Start fresh training for Latin BHO dataset with stable config
# This script starts training from scratch (no resume)

cd "$(dirname "$0")"  # Change to diffbrush directory

# Optional: Remove existing checkpoints to start completely fresh
# Uncomment the next two lines if you want to delete existing checkpoints
echo "Removing existing checkpoints to start fresh..."
rm -rf outputs/LatinBHO/checkpoints/*.pth outputs/LatinBHO/latest.pth

# Start training with stable config (lower LR, batch size 4, tighter gradient clipping)
python train.py \
    --cfg_file configs/LatinBHO_stable.yml \
    --output_dir outputs/LatinBHO \
    --stable_dif_path model_zoo \
    --device cuda

