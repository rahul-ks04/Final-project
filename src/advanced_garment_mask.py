#!/usr/bin/env python3
"""
Advanced Garment Mask Extraction
Combines rembg with morphological operations and optional semantic refinement
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import argparse

def create_clean_garment_mask(garment_path, output_dir, method="morphological", debug=False):
    """
    Create clean garment mask using multiple refinement techniques.
    
    Methods:
    - 'morphological': Uses morphological operations to clean up rembg mask
    - 'contour': Detects contour and fills interior
    - 'hybrid': Combines multiple methods
    """
    
    print(f"[*] Loading garment: {garment_path}")
    garment_rgb = cv2.imread(garment_path)
    if garment_rgb is None:
        print(f"[ERROR] Could not load image")
        return False
    
    # Resize to standard size
    garment_rgb = cv2.resize(garment_rgb, (768, 1024))
    h, w = garment_rgb.shape[:2]
    
    print(f"[*] Image size: {w}×{h}")
    
    # Step 1: Get initial mask from rembg or manual approach
    print("[*] Creating initial mask...")
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(garment_rgb, cv2.COLOR_BGR2GRAY)
    
    # Try rembg if available, otherwise use threshold-based approach
    try:
        from rembg import remove
        print("  Using rembg for background removal...")
        garment_pil = Image.fromarray(cv2.cvtColor(garment_rgb, cv2.COLOR_BGR2RGB))
        garment_rgba = remove(garment_pil)
        garment_rgba_np = np.array(garment_rgba)
        initial_mask = garment_rgba_np[:, :, 3]  # Alpha channel
        initial_mask = cv2.resize(initial_mask, (w, h))
        _, initial_mask = cv2.threshold(initial_mask, 10, 255, cv2.THRESH_BINARY)
    except:
        print("  rembg not available, using threshold approach...")
        # Threshold bright pixels (white/light garments) or use Otsu's
        _, initial_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        # Invert if needed (assume garment is darker than background)
        if np.sum(initial_mask) < (w * h * 0.2):  # If less than 20% is white
            initial_mask = cv2.bitwise_not(initial_mask)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "01_initial_mask.png"), initial_mask)
    
    # Step 2: Morphological cleanup
    print("[*] Cleaning mask with morphological operations...")
    
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_cleaned = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "02_after_open.png"), mask_cleaned)
    
    # Fill small holes in the garment
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "03_after_close.png"), mask_cleaned)
    
    # Step 3: Contour-based shape refinement
    print("[*] Refining with contour detection...")
    
    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (the garment)
        largest_contour = max(contours, key=cv2.contourArea)
        garment_area = cv2.contourArea(largest_contour)
        
        if garment_area > (w * h * 0.05):  # At least 5% of image
            # Create mask from largest contour
            mask_contour = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask_contour, [largest_contour], 0, 255, -1)
            
            # Smooth the boundaries
            mask_contour = cv2.GaussianBlur(mask_contour, (5, 5), 0)
            _, mask_contour = cv2.threshold(mask_contour, 127, 255, cv2.THRESH_BINARY)
            
            mask_cleaned = mask_contour
            
            if debug:
                cv2.imwrite(os.path.join(output_dir, "04_after_contour.png"), mask_cleaned)
    
    # Step 4: Edge-aware refinement
    print("[*] Final edge refinement...")
    
    # Dilate slightly to ensure full coverage
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.dilate(mask_cleaned, kernel_dilate, iterations=1)
    
    if debug:
        cv2.imwrite(os.path.join(output_dir, "05_final_mask.png"), mask_final)
    
    # Save outputs
    print("[*] Saving outputs...")
    
    # Save cleaned mask
    mask_path = os.path.join(output_dir, "cloth_mask.png")
    cv2.imwrite(mask_path, mask_final)
    print(f"  ✓ Saved: {mask_path}")
    
    # Save mask visualization (RGB)
    mask_rgb = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR)
    mask_rgb_path = os.path.join(output_dir, "cloth_mask_rgb.png")
    cv2.imwrite(mask_rgb_path, mask_rgb)
    print(f"  ✓ Saved: {mask_rgb_path}")
    
    # Apply mask to garment RGB (clean garment without noise)
    mask_3ch = cv2.cvtColor(mask_final, cv2.COLOR_GRAY2BGR) / 255.0
    garment_masked = (garment_rgb * mask_3ch).astype(np.uint8)
    
    cloth_path = os.path.join(output_dir, "cloth.png")
    cv2.imwrite(cloth_path, garment_masked)
    print(f"  ✓ Saved: {cloth_path}")
    
    # Statistics
    mask_area = np.sum(mask_final > 0)
    coverage = (mask_area / (w * h)) * 100
    print(f"\n[INFO] Mask coverage: {coverage:.1f}% of image")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Advanced Garment Mask Extraction")
    parser.add_argument("--input", required=True, help="Input garment image")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--method", choices=["morphological", "contour", "hybrid"], 
                       default="hybrid", help="Extraction method")
    parser.add_argument("--debug", action="store_true", help="Save intermediate steps")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not create_clean_garment_mask(args.input, args.output_dir, args.method, args.debug):
        return 1
    
    print("\n[SUCCESS] Garment mask extraction complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
