import os
import cv2
import numpy as np
import argparse
import torch
from PIL import Image

def preprocess_garment(garment_path, garment_type, output_dir, schp_mask_path=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load garment
    img = cv2.imread(garment_path)
    if img is None:
        print(f"Error: Could not read garment image at {garment_path}")
        return

    # Resize to standard size (e.g., 768x1024 for VITON-HD)
    img = cv2.resize(img, (768, 1024))
    
    # 1. Generate Garment Mask
    if garment_type == "flat":
        print("Using rembg for flat garment masking...")
        from rembg import remove
        
        # rembg works on PIL images
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        rembg_img = remove(pil_img)
        
        # Mask is the alpha channel
        mask = np.array(rembg_img)[:, :, 3]
        
        # Ensure mask is binary
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    else:
        # For worn garments, we use the parsing mask from SCHP
        if schp_mask_path and os.path.exists(schp_mask_path):
            # MUST use PIL to read SCHP output — it's a palette PNG.
            # cv2.imread in grayscale converts palette colors to grayscale
            # values, losing the actual label indices.
            parse_img = Image.open(schp_mask_path)
            parse_mask = np.array(parse_img)
            parse_mask = cv2.resize(parse_mask, (768, 1024), interpolation=cv2.INTER_NEAREST)
            # LIP/SCHP labels: 5 = Upper-clothes, 6 = Dress, 7 = Coat
            mask = np.where(np.isin(parse_mask, [5, 6, 7]), 255, 0).astype(np.uint8)
        else:
            print("Warning: Worn garment type selected but no SCHP mask provided. Using threshold fallback.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 2. Save Results
    cv2.imwrite(os.path.join(output_dir, "cloth.png"), img)
    cv2.imwrite(os.path.join(output_dir, "cloth_mask.png"), mask)
    
    # 3. Generate source_parsing.pt (Tensor expected by some flow renderers)
    # This is a dummy/simplified version - actual requirements vary by model
    # We'll save the mask as a long tensor
    mask_tensor = torch.from_numpy(mask).long()
    torch.save(mask_tensor, os.path.join(output_dir, "source_parsing.pt"))
    
    print(f"Garment preprocessing complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["flat", "worn"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--schp_mask", help="Path to SCHP mask (required for worn type)")
    parser.add_argument("--output_dir", required=True)
    
    args = parser.parse_args()
    preprocess_garment(args.input, args.type, args.output_dir, args.schp_mask)
