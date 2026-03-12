#!/usr/bin/env python3
"""
Quick garment preprocessing: Extract garment RGB and mask using rembg.
Creates cloth.png (garment RGB) and cloth_mask.png (binary mask).
"""

import os
import sys
import numpy as np
import argparse
from PIL import Image
import cv2

def preprocess_flat_garment(garment_path, output_dir):
    """Preprocess flat garment: remove background using rembg."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Loading garment: {garment_path}")
    img_pil = Image.open(garment_path).convert("RGB")
    
    # Resize to standard model size
    img_pil = img_pil.resize((768, 1024))
    
    print("[*] Removing background with rembg...")
    try:
        from rembg import remove
        img_rgba = remove(img_pil)  # Returns RGBA PIL image
    except ImportError:
        print("[ERROR] rembg not installed. Install with: pip install rembg")
        return False
    
    # Extract RGB and alpha
    img_rgba_np = np.array(img_rgba)
    img_rgb_np = img_rgba_np[:, :, :3]
    alpha = img_rgba_np[:, :, 3]
    
    # Create binary mask (threshold alpha channel)
    mask = np.where(alpha > 10, 255, 0).astype(np.uint8)
    
    # Save cloth RGB
    cloth_rgb = Image.fromarray(img_rgb_np, mode='RGB')
    cloth_rgb_path = os.path.join(output_dir, "cloth.png")
    cloth_rgb.save(cloth_rgb_path)
    print(f"[OK] Saved garment RGB: {cloth_rgb_path}")
    
    # Save cloth mask
    cloth_mask = Image.fromarray(mask, mode='L')
    cloth_mask_path = os.path.join(output_dir, "cloth_mask.png")
    cloth_mask.save(cloth_mask_path)
    print(f"[OK] Saved garment mask: {cloth_mask_path}")
    
    # Also save as torch tensor file for compatibility (optional try)
    try:
        import torch
        mask_tensor = torch.from_numpy(mask).long()
        torch.save(mask_tensor, os.path.join(output_dir, "source_parsing.pt"))
        print(f"[OK] Saved source_parsing.pt")
    except ImportError:
        print(f"[SKIP] torch not available, skipping .pt save")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Preprocess flat garment with rembg")
    parser.add_argument("--input", required=True, help="Path to garment image")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        return 1
    
    if not preprocess_flat_garment(args.input, args.output_dir):
        return 1
    
    print(f"\n[SUCCESS] Garment preprocessing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
