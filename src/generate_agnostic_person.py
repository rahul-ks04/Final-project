import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def generate_agnostic(img_path, parse_path, output_dir, warped_mask_path=None, dilation_kernel_size=15, smoothing_sigma=5):
    """
    Generates agnostic image.
    Following user intuition: Mask the OLD shirt and the NEW shirt footprint.
    Any skin outside these regions stays untouched.
    """
    agnostic_img_dir = os.path.join(output_dir, "img")
    agnostic_parse_dir = os.path.join(output_dir, "parse")
    agnostic_mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(agnostic_img_dir, exist_ok=True)
    os.makedirs(agnostic_parse_dir, exist_ok=True)
    os.makedirs(agnostic_mask_dir, exist_ok=True)

    # Load image and parsing
    image = np.array(Image.open(img_path).convert("RGB"))
    parse = np.array(Image.open(parse_path))

    # 1. Identify ORIGINAL clothing area (LIP Labels: 5:Upper, 6:Dress, 7:Coat, 10:Jumpsuit, 11:Scarf)
    clothing_mask = np.isin(parse, [5, 6, 7, 10, 11]).astype(np.uint8)
    
    # 2. Identify NEW clothing footprint (if provided)
    w_mask = np.zeros_like(clothing_mask)
    if warped_mask_path:
        w_mask_img = Image.open(warped_mask_path).convert("L").resize((image.shape[1], image.shape[0]), Image.NEAREST)
        w_mask = (np.array(w_mask_img) > 127).astype(np.uint8)
    
    # 3. Final Agnostic Mask = (Old Clothing Area OR New Clothing Footprint)
    # Preservation labels: 1:Hat, 2:Hair, 4:Sunglasses, 13:Face (NEVER MASK THESE)
    preserve_mask = np.isin(parse, [1, 2, 4, 13]).astype(np.uint8)
    
    combined_mask = (clothing_mask | w_mask)
    
    # Refine with Dilation
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=1)
    
    # Ensure face/hair are never masked
    final_agnostic_mask = (dilated_mask == 1) & (preserve_mask == 0)

    # 4. Smoothing for seamless blending
    mask_float = final_agnostic_mask.astype(float)
    if smoothing_sigma > 0:
        mask_float = cv2.GaussianBlur(mask_float, (0, 0), smoothing_sigma)

    # 5. Create Agnostic Image (Gray out the masked area)
    gray_img = np.full_like(image, 128, dtype=np.uint8)
    mask_float_3d = mask_float[:, :, None]
    
    # result = original * (1 - mask) + gray * mask
    agnostic_img_float = (1.0 - mask_float_3d) * image.astype(float) + mask_float_3d * gray_img.astype(float)
    agnostic_img = np.clip(agnostic_img_float, 0, 255).astype(np.uint8)

    # 6. Save results
    img_name = Path(img_path).name
    Image.fromarray(agnostic_img).save(os.path.join(agnostic_img_dir, img_name))
    
    # Save binary mask
    Image.fromarray((final_agnostic_mask * 255).astype(np.uint8)).save(os.path.join(agnostic_mask_dir, img_name))
    
    # Save parsing (clear out the masked areas)
    agnostic_parse = parse.copy()
    agnostic_parse[final_agnostic_mask] = 0
    parse_name = Path(parse_path).name
    Image.fromarray(agnostic_parse.astype(np.uint8)).save(os.path.join(agnostic_parse_dir, parse_name))

    print(f"Generated agnostic person (Intuitive Layering Mode): {os.path.join(output_dir, img_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Agnostic Person for VITON")
    parser.add_argument("--image", required=True)
    parser.add_argument("--parse", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dilation", type=int, default=15)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--warped_mask", help="Path to warped garment mask")
    parser.add_argument("--preserve_arms", action="store_true") # Kept for CLI compatibility, but logic is now automatic

    args = parser.parse_args()
    generate_agnostic(args.image, args.parse, args.output_dir, args.warped_mask, args.dilation, args.smoothing)
