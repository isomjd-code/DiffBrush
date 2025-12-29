#!/usr/bin/env python3
"""Quick script to delete old writer_0 directories."""

import os
import shutil

output_data_dir = "diffbrush/data/LatinBHO"
writer_0_dirs = [
    os.path.join(output_data_dir, 'images', 'writer_0'),
    os.path.join(output_data_dir, 'style_images', 'writer_0')
]

print("Cleaning up old writer_0 directories...")
for dir_path in writer_0_dirs:
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"  ✓ Deleted: {dir_path}")
        except Exception as e:
            print(f"  ✗ Error deleting {dir_path}: {e}")
    else:
        print(f"  - Not found (already deleted): {dir_path}")

print("\nDone!")





