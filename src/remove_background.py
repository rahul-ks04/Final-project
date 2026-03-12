import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove

def remove_background(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    input_image = Image.open(input_path)
    
    # Remove background
    output_image = remove(input_image)
    
    # person.png (RGBA) - contain transparent person
    output_image.save(os.path.join(output_dir, "person.png"))
    
    # Generate background.png
    rgba_np = np.array(output_image)
    alpha = rgba_np[:, :, 3]
    mask = alpha > 0
    
    original_np = np.array(input_image.convert("RGB"))
    background_np = original_np.copy()
    background_np[mask] = [0, 0, 0] # Black out the person
    
    background_image = Image.fromarray(background_np)
    background_image.save(os.path.join(output_dir, "background.png"))
    
    print(f"Background removal complete. Saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    remove_background(args.input, args.output_dir)
