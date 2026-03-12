#!/usr/bin/env python3
"""
Convenience wrapper to run Flow Renderer (Stage 2) with latest pipeline outputs.
Automatically uses latest PAM + garment outputs and saves results to a versioned directory.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def get_latest_version(base_dir, prefix):
    """Get the latest versioned directory matching prefix."""
    if not os.path.exists(base_dir):
        return None
    versions = [d for d in os.listdir(base_dir) if d.startswith(prefix)]
    if not versions:
        return None
    return os.path.join(base_dir, sorted(versions)[-1])

def main():
    # Project directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs_dir = os.path.join(project_root, "outputs")
    src_dir = os.path.join(project_root, "src")
    
    # Find latest outputs
    pam_dir = get_latest_version(outputs_dir, "pam_")
    garment_dir = get_latest_version(outputs_dir, "fashn_")
    
    if not pam_dir:
        print("[ERROR] No PAM output found. Run PAM first.")
        return 1
    
    if not garment_dir:
        print("[ERROR] No garment output found. Run garment preprocessing first.")
        return 1
    
    pam_npy = os.path.join(pam_dir, "predicted_parsing_20ch.npy")
    garment_rgb = os.path.join(garment_dir, "garment.png")
    garment_mask = os.path.join(garment_dir, "garment.png")  # Using same as mask for now
    
    # Validate files
    if not os.path.isfile(pam_npy):
        print(f"[ERROR] PAM output not found: {pam_npy}")
        return 1
    
    if not os.path.isfile(garment_rgb):
        print(f"[ERROR] Garment RGB not found: {garment_rgb}")
        return 1
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    flow_output_dir = os.path.join(outputs_dir, f"flow_mediapipe_{timestamp}")
    os.makedirs(flow_output_dir, exist_ok=True)
    
    print(f"[*] Using PAM output: {pam_dir}")
    print(f"[*] Using garment: {garment_dir}")
    print(f"[*] Output directory: {flow_output_dir}")
    
    # Run flow renderer
    cmd = [
        sys.executable,
        os.path.join(src_dir, "fvnt_flow_renderer.py"),
        "--pam_output", pam_npy,
        "--garment_rgb", garment_rgb,
        "--garment_mask", garment_mask,
        "--output_dir", flow_output_dir,
    ]
    
    print(f"\n[*] Running command:\n    {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"\n[SUCCESS] Flow Renderer completed!")
        print(f"Results saved to: {flow_output_dir}")
        return 0
    else:
        print(f"\n[ERROR] Flow Renderer failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
