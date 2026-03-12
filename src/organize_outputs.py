#!/usr/bin/env python3
"""
Reorganize outputs into clean pipeline structure with RGB visualizations
"""

import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

def gen_parsing_rgb(label_path, rgb_out_path):
    """Generate RGB from parsing labels."""
    if not os.path.isfile(label_path):
        return False
    
    parse_map = Image.open(label_path)
    parse_np = np.array(parse_map)
    
    palette = np.array([
        [0,0,0],[128,0,0],[255,0,0],[0,85,0],[170,0,51],
        [255,85,0],[0,0,85],[0,119,221],[85,85,0],[0,85,85],
        [85,51,0],[52,86,128],[0,128,0],[0,0,255],[51,170,221],
        [0,255,255],[85,255,170],[170,255,85],[255,255,0],[255,170,0]
    ], dtype=np.uint8)
    
    parse_rgb = palette[np.minimum(parse_np, len(palette)-1)]
    rgb_img = Image.fromarray(parse_rgb)
    rgb_img.save(rgb_out_path)
    return True

def gen_mask_rgb(mask_path, rgb_out_path):
    """Generate grayscale RGB from binary/grayscale mask."""
    if not os.path.isfile(mask_path):
        return False
    
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
    mask_rgb_img = Image.fromarray(mask_rgb)
    mask_rgb_img.save(rgb_out_path)
    return True

def main():
    outputs = os.path.abspath("outputs")
    
    print("\n" + "="*70)
    print("  ORGANIZING OUTPUTS")
    print("="*70 + "\n")
    
    # Create clean structure
    structure = {
        "rembg": None,
        "fashn": None,
        "mediapipe": None,
        "garment": None,
        "pam": None,
    }
    
    for stage, _ in structure.items():
        stage_dir = os.path.join(outputs, stage)
        os.makedirs(stage_dir, exist_ok=True)
        structure[stage] = stage_dir
    
    # ===== REMBG =====
    print("1. Background Removal (rembg)...")
    rembg_src = os.path.join(outputs, "rembg")
    if os.path.isdir(rembg_src) and len(os.listdir(rembg_src)) > 0:
        print(f"   ✓ Found existing rembg outputs")
    else:
        print(f"   ℹ Run remove_background.py to generate")
    
    # ===== FASHN =====
    print("2. FASHN Parser...")
    fashn_old_dir = os.path.join(outputs, "fashn_mediapipe_v1")
    fashn_new_dir = structure["fashn"]
    
    if os.path.isdir(fashn_old_dir):
        # Copy and reorganize
        person_parse = os.path.join(fashn_old_dir, "person.png")
        if os.path.isfile(person_parse):
            shutil.copy(person_parse, os.path.join(fashn_new_dir, "person.png"))
            print(f"   ✓ Copied person.png")
            
            if gen_parsing_rgb(person_parse, os.path.join(fashn_new_dir, "person_parsing_rgb.png")):
                print(f"   ✓ Generated person_parsing_rgb.png")
    
    # ===== MEDIAPIPE =====
    print("3. MediaPipe Pose...")
    mediapipe_old_dir = os.path.join(outputs, "pose_mediapipe_v1")
    mediapipe_new_dir = structure["mediapipe"]
    
    if os.path.isdir(mediapipe_old_dir):
        keypoints = os.path.join(mediapipe_old_dir, "person_keypoints.json")
        if os.path.isfile(keypoints):
            shutil.copy(keypoints, os.path.join(mediapipe_new_dir, "person_keypoints.json"))
            print(f"   ✓ Copied person_keypoints.json")
        
        pose_maps = os.path.join(mediapipe_old_dir, "pose_maps.npy")
        if os.path.isfile(pose_maps):
            shutil.copy(pose_maps, os.path.join(mediapipe_new_dir, "pose_maps.npy"))
            print(f"   ✓ Copied pose_maps.npy")
    
    # ===== GARMENT =====
    print("4. Garment Preprocessing...")
    garment_old_dir = os.path.join(outputs, "garment_processed")
    if not os.path.isdir(garment_old_dir):
        garment_old_dir = os.path.join(outputs, "garment_mediapipe_v1")
    
    garment_new_dir = structure["garment"]
    
    if os.path.isdir(garment_old_dir):
        cloth_rgb = os.path.join(garment_old_dir, "cloth.png")
        cloth_mask = os.path.join(garment_old_dir, "cloth_mask.png")
        
        if os.path.isfile(cloth_rgb):
            shutil.copy(cloth_rgb, os.path.join(garment_new_dir, "cloth.png"))
            print(f"   ✓ Copied cloth.png")
        
        if os.path.isfile(cloth_mask):
            shutil.copy(cloth_mask, os.path.join(garment_new_dir, "cloth_mask.png"))
            print(f"   ✓ Copied cloth_mask.png")
            
            if gen_mask_rgb(cloth_mask, os.path.join(garment_new_dir, "cloth_mask_rgb.png")):
                print(f"   ✓ Generated cloth_mask_rgb.png")
    
    # ===== PAM =====
    print("5. PAM - Target Parsing...")
    pam_old_dir = os.path.join(outputs, "pam_mediapipe_v1")
    pam_new_dir = structure["pam"]
    
    if os.path.isdir(pam_old_dir):
        pred_parse = os.path.join(pam_old_dir, "predicted_parsing.png")
        pred_20ch = os.path.join(pam_old_dir, "predicted_parsing_20ch.npy")
        
        if os.path.isfile(pred_parse):
            shutil.copy(pred_parse, os.path.join(pam_new_dir, "predicted_parsing.png"))
            print(f"   ✓ Copied predicted_parsing.png")
        
        if os.path.isfile(pred_20ch):
            shutil.copy(pred_20ch, os.path.join(pam_new_dir, "predicted_parsing_20ch.npy"))
            print(f"   ✓ Copied predicted_parsing_20ch.npy")
            
            # Generate RGB visualization
            try:
                pam_data = np.load(pred_20ch)
                pam_data = np.squeeze(pam_data)
                if pam_data.ndim == 3:
                    pam_labels = np.argmax(pam_data, axis=0).astype(np.uint8)
                else:
                    pam_labels = pam_data.astype(np.uint8)
                
                palette = np.array([
                    [0,0,0],[128,0,0],[255,0,0],[0,85,0],[170,0,51],
                    [255,85,0],[0,0,85],[0,119,221],[85,85,0],[0,85,85],
                    [85,51,0],[52,86,128],[0,128,0],[0,0,255],[51,170,221],
                    [0,255,255],[85,255,170],[170,255,85],[255,255,0],[255,170,0]
                ], dtype=np.uint8)
                
                pam_rgb = palette[np.minimum(pam_labels, len(palette)-1)]
                pam_rgb_img = Image.fromarray(pam_rgb)
                pam_rgb_img.save(os.path.join(pam_new_dir, "predicted_parsing_rgb.png"))
                print(f"   ✓ Generated predicted_parsing_rgb.png")
            except Exception as e:
                print(f"   ⚠ Error generating RGB: {e}")
        else:
            # If no 20ch, try to use label map
            if os.path.isfile(pred_parse):
                if gen_parsing_rgb(pred_parse, os.path.join(pam_new_dir, "predicted_parsing_rgb.png")):
                    print(f"   ✓ Generated predicted_parsing_rgb.png from label map")
    
    # ===== Summary =====
    print("\n" + "="*70)
    print("  ORGANIZED OUTPUT STRUCTURE")
    print("="*70 + "\n")
    
    for stage, stage_dir in structure.items():
        if os.path.isdir(stage_dir):
            files = os.listdir(stage_dir)
            if files:
                print(f"✓ {stage}/")
                for f in sorted(files):
                    fpath = os.path.join(stage_dir, f)
                    if os.path.isfile(fpath):
                        size = os.path.getsize(fpath)
                        print(f"    {f:40} ({size:,} bytes)")
            else:
                print(f"ℹ {stage}/ (empty)")
        else:
            print(f"✗ {stage}/ (does not exist)")
    
    print("\n" + "="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
