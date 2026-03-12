import os
import cv2
import numpy as np
from PIL import Image

def restore_background(original_jpg, tryon_result_png, rembg_person_png, output_path):
    """
    Composites the try-on result back into the original background.
    """
    # 1. Load images
    original = Image.open(original_jpg).convert("RGB")
    tryon = Image.open(tryon_result_png).convert("RGB")
    rembg_person = Image.open(rembg_person_png).convert("RGBA")
    
    W_orig, H_orig = original.size
    
    # 2. Extract alpha mask from rembg output
    # This mask tells us exactly where the person is in the original image
    alpha = np.array(rembg_person)[:, :, 3]
    mask = (alpha > 0).astype(np.float32)
    
    # 3. Resize tryon result to original size
    tryon_resized = tryon.resize((W_orig, H_orig), Image.LANCZOS)
    tryon_np = np.array(tryon_resized).astype(np.float32)
    original_np = np.array(original).astype(np.float32)
    
    # 4. Composite
    # result = tryon * Mask + original * (1 - Mask)
    mask_3d = mask[:, :, None]
    composite_np = tryon_np * mask_3d + original_np * (1.0 - mask_3d)
    
    # 5. Save
    composite_img = Image.fromarray(np.clip(composite_np, 0, 255).astype(np.uint8))
    composite_img.save(output_path)
    print(f"Final composition saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Original person.jpg")
    parser.add_argument("--tryon", required=True, help="tryon_result.png from StyleVTON")
    parser.add_argument("--rembg_mask", required=True, help="person.png from rembg")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    restore_background(args.original, args.tryon, args.rembg_mask, args.output)
