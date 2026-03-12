#!/usr/bin/env python3
"""
FVNT Complete Preprocessing Pipeline
Organized output structure with clear stages and RGB visualizations
"""

import os
import sys
import subprocess
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

def run_cmd(cmd, description=""):
    """Execute command with progress output."""
    if description:
        print(f"\n{'='*70}")
        print(f"  {description}")
        print(f"{'='*70}\n")
    print(f"$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] Stage failed: {description}")
        return False
    print(f"\n[OK] Stage complete: {description}")
    return True

def gen_parsing_rgb(label_path, rgb_path):
    """Generate RGB visualization from parsing label map."""
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
    rgb_img.save(rgb_path)
    print(f"  ✓ Generated RGB: {rgb_path}")

def gen_mask_rgb(mask_path, rgb_path):
    """Generate RGB visualization from binary mask (grayscale→RGB)."""
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    mask_rgb = np.stack([mask_np, mask_np, mask_np], axis=-1)
    mask_rgb_img = Image.fromarray(mask_rgb)
    mask_rgb_img.save(rgb_path)
    print(f"  ✓ Generated mask RGB: {rgb_path}")

def main():
    parser = argparse.ArgumentParser(description="FVNT Organized Pipeline")
    parser.add_argument("--project_root", default=".", help="Project root")
    parser.add_argument("--person", default="inputs/person.jpg", help="Input person image")
    parser.add_argument("--garment", default="inputs/garment.jpg", help="Input garment image")
    
    args = parser.parse_args()
    
    project_root = os.path.abspath(args.project_root)
    src_dir = os.path.join(project_root, "src")
    outputs_dir = os.path.join(project_root, "outputs")
    
    person_img = os.path.join(project_root, args.person)
    garment_img = os.path.join(project_root, args.garment)
    
    # Validate inputs
    if not os.path.isfile(person_img):
        print(f"[ERROR] Person image not found: {person_img}")
        return 1
    if not os.path.isfile(garment_img):
        print(f"[ERROR] Garment image not found: {garment_img}")
        return 1
    
    print(f"\n{'='*70}")
    print(f"  FVNT PREPROCESSING PIPELINE")
    print(f"  Person: {person_img}")
    print(f"  Garment: {garment_img}")
    print(f"  Output: {outputs_dir}")
    print(f"{'='*70}\n")
    
    # ===== STAGE 1: Background Removal (rembg) =====
    rembg_dir = os.path.join(outputs_dir, "rembg")
    os.makedirs(rembg_dir, exist_ok=True)
    
    if not run_cmd(
        [sys.executable, os.path.join(src_dir, "remove_background.py"),
         "--input", person_img, "--output_dir", rembg_dir],
        "STAGE 1: Background Removal (rembg)"
    ):
        return 1
    
    person_no_bg = os.path.join(rembg_dir, "person.png")
    
    # ===== STAGE 2: FASHN Parser =====
    fashn_dir = os.path.join(outputs_dir, "fashn")
    os.makedirs(fashn_dir, exist_ok=True)
    
    # Create temp input
    fashn_input = os.path.join(outputs_dir, "_temp_fashn_input")
    os.makedirs(fashn_input, exist_ok=True)
    shutil.copy(person_no_bg, os.path.join(fashn_input, "person.png"))
    
    if not run_cmd(
        [sys.executable, os.path.join(src_dir, "run_fashn_parser.py"),
         "--input_dir", fashn_input, "--output_dir", fashn_dir],
        "STAGE 2: FASHN - Person Semantic Parsing"
    ):
        return 1
    
    # Generate parsing RGB visualization
    print("\n  Generating FASHN RGB visualization...")
    gen_parsing_rgb(
        os.path.join(fashn_dir, "person.png"),
        os.path.join(fashn_dir, "person_parsing_rgb.png")
    )
    shutil.rmtree(fashn_input, ignore_errors=True)
    
    # ===== STAGE 3: MediaPipe Pose =====
    mediapipe_dir = os.path.join(outputs_dir, "mediapipe")
    os.makedirs(mediapipe_dir, exist_ok=True)
    
    if not run_cmd(
        [sys.executable, os.path.join(src_dir, "run_pose_mediapipe.py"),
         "--input", person_no_bg, "--output_dir", mediapipe_dir],
        "STAGE 3: MediaPipe - Pose Keypoint Extraction"
    ):
        return 1
    
    # ===== STAGE 4: Garment Preprocessing =====
    garment_dir = os.path.join(outputs_dir, "garment")
    os.makedirs(garment_dir, exist_ok=True)
    
    if not run_cmd(
        [sys.executable, os.path.join(src_dir, "quick_preprocess_garment.py"),
         "--input", garment_img, "--output_dir", garment_dir],
        "STAGE 4: Garment Preprocessing (rembg)"
    ):
        return 1
    
    # Generate garment mask RGB visualization
    print("\n  Generating garment mask RGB visualization...")
    gen_mask_rgb(
        os.path.join(garment_dir, "cloth_mask.png"),
        os.path.join(garment_dir, "cloth_mask_rgb.png")
    )
    
    # ===== STAGE 5: PAM (Target Parsing Generation) =====
    pam_dir = os.path.join(outputs_dir, "pam")
    os.makedirs(pam_dir, exist_ok=True)
    
    # Find PAM checkpoint
    pam_checkpoint = None
    checkpoint_candidates = [
        os.path.join(project_root, "FVNT", "model", "stage_1", "G_stage1_best.pth"),
        os.path.join(project_root, "FVNT", "model", "stage1_model"),
        os.path.join(project_root, "checkpoints", "pam.pth"),
    ]
    
    for ckpt_path in checkpoint_candidates:
        if os.path.isfile(ckpt_path):
            pam_checkpoint = ckpt_path
            break
    
    if not pam_checkpoint:
        print(f"\n{'='*70}")
        print(f"  ⚠ PAM CHECKPOINT NOT FOUND")
        print(f"{'='*70}")
        print(f"\n  Expected locations:")
        for ckpt_path in checkpoint_candidates:
            print(f"    - {ckpt_path}")
        print(f"\n  ℹ Pipeline continues with visualizations only (no PAM inference).")
        print(f"  ℹ Download checkpoint from: https://drive.google.com/...")
        
        # Skip PAM, but show structure
        print(f"\n  To run PAM later:")
        print(f"  $ python {os.path.join(src_dir, 'run_pam.py')} \\")
        print(f"      --cloth {os.path.join(garment_dir, 'cloth.png')} \\")
        print(f"      --cloth_mask {os.path.join(garment_dir, 'cloth_mask.png')} \\")
        print(f"      --parse {os.path.join(fashn_dir, 'person.png')} \\")
        print(f"      --pose {os.path.join(mediapipe_dir, 'person_keypoints.json')} \\")
        print(f"      --image {person_no_bg} \\")
        print(f"      --checkpoint <PATH_TO_CHECKPOINT> \\")
        print(f"      --output_dir {pam_dir}")
        
        # Use FASHN output as reference (not ideal, but for visualization)
        print(f"\n  Using FASHN output as reference for now...")
        shutil.copy(
            os.path.join(fashn_dir, "person.png"),
            os.path.join(pam_dir, "predicted_parsing.png")
        )
        shutil.copy(
            os.path.join(fashn_dir, "person_parsing_rgb.png"),
            os.path.join(pam_dir, "predicted_parsing_rgb.png")
        )
    else:
        print(f"\n  ✓ Found PAM checkpoint: {pam_checkpoint}")
        if not run_cmd(
            [sys.executable, os.path.join(src_dir, "run_pam.py"),
             "--cloth", os.path.join(garment_dir, "cloth.png"),
             "--cloth_mask", os.path.join(garment_dir, "cloth_mask.png"),
             "--parse", os.path.join(fashn_dir, "person.png"),
             "--pose", os.path.join(mediapipe_dir, "person_keypoints.json"),
             "--image", person_no_bg,
             "--checkpoint", pam_checkpoint,
             "--output_dir", pam_dir],
            "STAGE 5: PAM - Target Parsing Generation"
        ):
            return 1
        
        # Generate PAM RGB visualization
        print("\n  Generating PAM output RGB visualization...")
        pam_npy = np.load(os.path.join(pam_dir, "predicted_parsing_20ch.npy"))
        pam_npy = np.squeeze(pam_npy)
        
        if pam_npy.ndim == 3:
            pam_labels = np.argmax(pam_npy, axis=0).astype(np.uint8)
        else:
            pam_labels = pam_npy.astype(np.uint8)
        
        palette = np.array([
            [0,0,0],[128,0,0],[255,0,0],[0,85,0],[170,0,51],
            [255,85,0],[0,0,85],[0,119,221],[85,85,0],[0,85,85],
            [85,51,0],[52,86,128],[0,128,0],[0,0,255],[51,170,221],
            [0,255,255],[85,255,170],[170,255,85],[255,255,0],[255,170,0]
        ], dtype=np.uint8)
        
        pam_rgb = palette[np.minimum(pam_labels, len(palette)-1)]
        pam_rgb_img = Image.fromarray(pam_rgb)
        pam_rgb_img.save(os.path.join(pam_dir, "predicted_parsing_rgb.png"))
        print(f"  ✓ Generated PAM RGB: {os.path.join(pam_dir, 'predicted_parsing_rgb.png')}")
    
    # ===== Pipeline Complete =====
    print(f"\n{'='*70}")
    print(f"  ✓ PREPROCESSING PIPELINE COMPLETE!")
    print(f"{'='*70}")
    
    print(f"\nOutput Directory Structure:")
    print(f"""
    outputs/
    │
    ├── rembg/
    │   ├── person.png              (person without background)
    │   └── background.png          (extracted background)
    │
    ├── fashn/
    │   ├── person.png              (semantic parsing labels 0-19)
    │   └── person_parsing_rgb.png  (colored visualization)
    │
    ├── mediapipe/
    │   ├── person_keypoints.json   (18-point OpenPose-format pose)
    │   └── pose_maps.npy           (confidence heatmaps)
    │
    ├── garment/
    │   ├── cloth.png               (garment RGB, resized 768×1024)
    │   ├── cloth_mask.png          (binary mask, 768×1024)
    │   └── cloth_mask_rgb.png      (mask visualization)
    │
    └── pam/
        ├── predicted_parsing.png           (target parsing label map 256×192)
        ├── predicted_parsing_20ch.npy      (20-channel semantic tensor)
        └── predicted_parsing_rgb.png       (colored visualization)
    """)
    
    print(f"\nNext Steps:")
    print(f"  1. Flow Renderer (Stage 2): Warp garment using optical flow")
    print(f"     → outputs/flow/")
    print(f"  2. FEM (Stage 3): Final synthesis with GAN")
    print(f"     → outputs/fem/")
    print(f"  3. Background Restoration: Compose final image with original background")
    print(f"     → outputs/final/")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
