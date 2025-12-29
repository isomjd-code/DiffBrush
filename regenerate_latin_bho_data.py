#!/usr/bin/env python3
"""
Regenerate Latin BHO training data for diffbrush with CP40-XXX groups as writers.

This script:
1. Reads all metadata.json files from corrected_lines directories
2. Extracts CP40-XXX group name from directory name
3. Maps line_ids to train/val/test splits
4. Generates new train.txt, val.txt, test.txt files with format: CP40-XXX,image_name transcription
"""

import os
import json
import re
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def extract_cp40_roll_number(directory_name: str) -> str:
    """
    Extract CP40-XXX roll number from directory name (just the 3 digits after CP40-).
    
    Examples:
    - "CP 40-559 055-a" -> "CP40-559"
    - "CP40-562 307a" -> "CP40-562"
    - "CP 40-559 116d-a" -> "CP40-559"
    - "CP40-559055-a" -> "CP40-559"
    """
    # Remove spaces and normalize
    name = directory_name.replace(" ", "")
    
    # Ensure CP40 format (handle "CP40" and "CP 40")
    name = re.sub(r'CP\s*40', 'CP40', name)
    
    # Extract roll number: CP40-XXX where XXX is 3 digits
    match = re.search(r'CP40-(\d{3})', name)
    if match:
        roll_number = match.group(1)
        return f"CP40-{roll_number}"
    else:
        # Fallback: try to find any CP40-XXX pattern
        match = re.search(r'CP40-(\d+)', name)
        if match:
            # Take first 3 digits
            roll_digits = match.group(1)[:3]
            return f"CP40-{roll_digits}"
        else:
            # If no match, return original (shouldn't happen)
            return name


def load_split_ids(split_ids_file: str) -> set:
    """Load line IDs for a given split."""
    if not os.path.exists(split_ids_file):
        return set()
    
    with open(split_ids_file, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f if line.strip()}


def process_metadata_file(metadata_path: str, group_name: str, split_ids_map: dict) -> list:
    """
    Process a single metadata.json file and return list of (line_id, split, transcription) tuples.
    
    Returns:
        List of tuples: (line_id, split, corrected_text)
    """
    results = []
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        lines = metadata.get('lines', [])
        
        for line_data in lines:
            line_id = line_data.get('line_id')
            corrected_text = line_data.get('corrected_text', '').strip()
            
            if not line_id or not corrected_text:
                continue
            
            # Determine which split this line belongs to
            split = None
            for split_name, split_ids in split_ids_map.items():
                if line_id in split_ids:
                    split = split_name
                    break
            
            if split:
                results.append((line_id, split, corrected_text))
    
    except Exception as e:
        print(f"Error processing {metadata_path}: {e}")
    
    return results


def organize_images_by_writer(dataset_dir: str, split_data: dict, output_data_dir: str):
    """
    Organize images into writer directories in the diffbrush data directory.
    Expected structure: 
    - Source: bootstrap_training_data/datasets/dataset_v22/images/{split}/{line_id}.png
    - Destination: diffbrush/data/LatinBHO/images/{writer_id}/{line_id}.png
    """
    # Destination: diffbrush/data/LatinBHO/images
    dest_images_dir = os.path.join(output_data_dir, 'images')
    os.makedirs(dest_images_dir, exist_ok=True)
    
    # Source: bootstrap_training_data/datasets/dataset_v22/images
    source_images_base = os.path.join(dataset_dir, 'images')
    
    print("\nOrganizing images by writer...")
    
    # Collect all writers and their images across all splits
    all_writer_images = defaultdict(set)  # writer_id -> set of image_names
    
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(source_images_base, split)
        if not os.path.exists(split_images_dir):
            print(f"  Warning: {split_images_dir} does not exist, skipping...")
            continue
        
        for writer_id, line_id, _ in split_data[split]:
            image_name = f"{line_id}.png"
            all_writer_images[writer_id].add((split, image_name))
    
    # Copy images to writer directories
    total_copied = 0
    for writer_id, image_info in tqdm(all_writer_images.items(), desc="Organizing images"):
        writer_dir = os.path.join(dest_images_dir, writer_id)
        os.makedirs(writer_dir, exist_ok=True)
        
        for split, image_name in image_info:
            src_path = os.path.join(source_images_base, split, image_name)
            dst_path = os.path.join(writer_dir, image_name)
            
            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    total_copied += 1
                else:
                    total_copied += 1  # Count as already organized
    
    print(f"  Organized {total_copied} images into {len(all_writer_images)} writer directories")


def organize_style_images_by_writer(dataset_dir: str, split_data: dict, output_data_dir: str, num_style_per_writer: int = 15):
    """
    Organize style images into writer directories.
    Style images are sampled from training images (typically a subset for style reference).
    Expected structure: 
    - Source: bootstrap_training_data/datasets/dataset_v22/images/{split}/{line_id}.png
    - Destination: diffbrush/data/LatinBHO/style_images/{writer_id}/{line_id}.png
    """
    import random
    random.seed(42)  # For reproducibility
    
    # Destination: diffbrush/data/LatinBHO/style_images
    dest_style_dir = os.path.join(output_data_dir, 'style_images')
    os.makedirs(dest_style_dir, exist_ok=True)
    
    # Source: bootstrap_training_data/datasets/dataset_v22/images
    source_images_base = os.path.join(dataset_dir, 'images')
    
    print("\nOrganizing style images by writer...")
    
    # Collect all writers and their images (prefer train split for style images)
    writer_images = defaultdict(list)  # writer_id -> list of (split, image_name)
    
    # First, collect from train split (preferred for style images)
    train_images_dir = os.path.join(source_images_base, 'train')
    if os.path.exists(train_images_dir):
        for writer_id, line_id, _ in split_data['train']:
            image_name = f"{line_id}.png"
            writer_images[writer_id].append(('train', image_name))
    
    # If not enough images from train, supplement with val
    val_images_dir = os.path.join(source_images_base, 'val')
    if os.path.exists(val_images_dir):
        for writer_id, line_id, _ in split_data['val']:
            image_name = f"{line_id}.png"
            # Only add if we don't have enough from train
            if len([x for x in writer_images[writer_id] if x[0] == 'train']) < num_style_per_writer:
                writer_images[writer_id].append(('val', image_name))
    
    # Copy style images to writer directories (sample up to num_style_per_writer per writer)
    total_copied = 0
    for writer_id, image_list in tqdm(writer_images.items(), desc="Organizing style images"):
        writer_style_dir = os.path.join(dest_style_dir, writer_id)
        os.makedirs(writer_style_dir, exist_ok=True)
        
        # Sample up to num_style_per_writer images per writer
        if len(image_list) > num_style_per_writer:
            sampled_images = random.sample(image_list, num_style_per_writer)
        else:
            sampled_images = image_list
        
        for split, image_name in sampled_images:
            src_path = os.path.join(source_images_base, split, image_name)
            dst_path = os.path.join(writer_style_dir, image_name)
            
            if os.path.exists(src_path):
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                    total_copied += 1
                else:
                    total_copied += 1  # Count as already organized
    
    print(f"  Organized {total_copied} style images into {len(writer_images)} writer directories")


def cleanup_old_writer_directories(output_data_dir: str):
    """Remove old writer_0 directories if they exist."""
    writer_0_dirs = [
        os.path.join(output_data_dir, 'images', 'writer_0'),
        os.path.join(output_data_dir, 'style_images', 'writer_0')
    ]
    
    print("\nCleaning up old writer_0 directories...")
    for dir_path in writer_0_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"  Deleted: {dir_path}")
            except Exception as e:
                print(f"  Warning: Could not delete {dir_path}: {e}")
        else:
            print(f"  (Not found, skipping: {dir_path})")


def main():
    # Paths
    corrected_lines_dir = "bootstrap_training_data/corrected_lines"
    dataset_dir = "bootstrap_training_data/datasets/dataset_v22"
    # Output to diffbrush/data/LatinBHO
    output_data_dir = "diffbrush/data/LatinBHO"
    os.makedirs(output_data_dir, exist_ok=True)
    
    # Clean up old writer_0 directories first
    cleanup_old_writer_directories(output_data_dir)
    
    # Load split IDs
    print("Loading split IDs...")
    split_ids_map = {
        'train': load_split_ids(os.path.join(dataset_dir, 'train_ids.txt')),
        'val': load_split_ids(os.path.join(dataset_dir, 'val_ids.txt')),
        'test': load_split_ids(os.path.join(dataset_dir, 'test_ids.txt'))
    }
    
    print(f"Loaded {len(split_ids_map['train'])} train, {len(split_ids_map['val'])} val, {len(split_ids_map['test'])} test line IDs")
    
    # Collect all data by split
    split_data = defaultdict(list)  # split -> list of (group_name, line_id, transcription)
    
    # Process all metadata.json files
    print("\nProcessing metadata.json files...")
    metadata_files = list(Path(corrected_lines_dir).rglob('metadata.json'))
    
    for metadata_path in tqdm(metadata_files, desc="Processing metadata"):
        # Extract roll number (writer ID) from parent directory
        parent_dir = metadata_path.parent.name
        writer_id = extract_cp40_roll_number(parent_dir)
        
        # Process metadata file
        results = process_metadata_file(str(metadata_path), writer_id, split_ids_map)
        
        for line_id, split, transcription in results:
            split_data[split].append((writer_id, line_id, transcription))
    
    # Write output files to diffbrush/data/LatinBHO
    print("\nWriting output files...")
    for split in ['train', 'val', 'test']:
        output_file = os.path.join(output_data_dir, f'{split}.txt')
        data = split_data[split]
        
        print(f"Writing {split}.txt with {len(data)} entries...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for writer_id, line_id, transcription in sorted(data):
                # Format: writer_id,image_name transcription
                # Note: load_data() adds .png automatically, so we write just the line_id
                # The actual image file is {line_id}.png in the writer directory
                f.write(f"{writer_id},{line_id} {transcription}\n")
        
        print(f"  Wrote {len(data)} entries to {output_file}")
    
    # Organize images by writer in diffbrush/data/LatinBHO/images
    organize_images_by_writer(dataset_dir, split_data, output_data_dir)
    
    # Organize style images by writer in diffbrush/data/LatinBHO/style_images
    organize_style_images_by_writer(dataset_dir, split_data, output_data_dir, num_style_per_writer=15)
    
    # Print statistics
    print("\n=== Statistics ===")
    total_writers = set()
    for split in ['train', 'val', 'test']:
        writers = {writer for writer, _, _ in split_data[split]}
        total_writers.update(writers)
        print(f"{split}: {len(split_data[split])} lines, {len(writers)} unique writers")
    
    print(f"\nTotal unique CP40 writers (rolls): {len(total_writers)}")
    print(f"\nSample writers: {sorted(list(total_writers))[:10]}")


if __name__ == '__main__':
    main()

