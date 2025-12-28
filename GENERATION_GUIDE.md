# Generating Handwritten Text with DiffBrush

This guide explains how to use your trained DiffBrush model to generate synthetic handwriting images.

## Prerequisites

1. **Trained Model Checkpoint**: You need a trained model checkpoint (saved during training)
   - Checkpoints are saved in `outputs/LatinBHO/checkpoints/checkpoint_XXXX.pth`
   - The latest checkpoint is also saved as `outputs/LatinBHO/latest.pth`

2. **VAE Model**: The Stable Diffusion VAE should be in `model_zoo/` (already set up)

3. **Style Images**: Your style reference images should be in `data/LatinBHO/style_images/0/`

## Basic Usage

### Generate a single text

```bash
cd diffbrush
conda activate Diffusion_Brush

python generate_latin_bho.py \
    --cfg_file configs/LatinBHO.yml \
    --pretrained_model outputs/LatinBHO/latest.pth \
    --text "Hello world" \
    --save_dir generated_images \
    --stable_dif_path model_zoo
```

### Generate from a text file

Create a text file `texts.txt` with one text per line:

```
Hello world
Lorem ipsum dolor sit amet
The quick brown fox jumps over the lazy dog
```

Then run:

```bash
python generate_latin_bho.py \
    --cfg_file configs/LatinBHO.yml \
    --pretrained_model outputs/LatinBHO/latest.pth \
    --text_file texts.txt \
    --save_dir generated_images \
    --stable_dif_path model_zoo
```

## Advanced Options

### Sampling Parameters

- `--sampling_timesteps`: Number of DDIM sampling steps (default: 50)
  - More steps = higher quality but slower (try 20-100)
  - Recommended: 50 for good balance
  
- `--eta`: DDIM eta parameter (default: 0.0)
  - `0.0` = deterministic DDIM (consistent results)
  - `1.0` = stochastic DDPM (more diverse, slower)
  - Recommended: 0.0 for most cases

### Example with custom sampling

```bash
python generate_latin_bho.py \
    --cfg_file configs/LatinBHO.yml \
    --pretrained_model outputs/LatinBHO/checkpoints/checkpoint_3000.pth \
    --text "Custom text here" \
    --save_dir generated_images \
    --sampling_timesteps 100 \
    --eta 0.0
```

## How It Works

1. **Style Reference**: The model uses style images from `data/LatinBHO/style_images/0/` to learn the handwriting style
2. **Content Glyphs**: The text is converted to glyph images using Unifont
3. **Generation**: The diffusion model generates a handwritten version of the text in the learned style
4. **Output**: Generated images are saved as grayscale PNG files

## Output Format

Generated images are saved as:
- Filename: `generated_<sanitized_text>_<index>.png`
- Format: Grayscale PNG
- Location: Specified by `--save_dir`

## Tips

1. **Model Quality**: Better trained models (more iterations) produce better results
   - Check your training loss - lower loss generally means better generation
   - Try different checkpoints to see which works best

2. **Text Length**: The model works best with text lengths similar to your training data
   - Your training data appears to have variable lengths
   - Very short or very long texts may have mixed results

3. **Style Consistency**: Since you have a single writer, all generated images will be in the same style
   - To get variety, you could train separate models on different writers
   - Or use different style reference images during generation

4. **Troubleshooting**:
   - If generation is too slow, reduce `--sampling_timesteps` to 20-30
   - If results are blurry, increase `--sampling_timesteps` to 50-100
   - If you get memory errors, ensure you're using the correct GPU

## Notes

- The model generates images in the style learned from your training data
- Text is converted using Unifont glyphs - ensure your text only contains characters present in the training data
- Generated images are grayscale (black text on white background)

