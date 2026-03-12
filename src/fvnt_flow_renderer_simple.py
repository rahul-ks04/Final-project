#!/usr/bin/env python3
"""
FVNT Flow Renderer - Stage 2 (Proper Implementation)
Generates optical flow by aligning source garment to target person's upper clothes.

The correct approach:
1. Load PAM output: 20-channel target person parsing (includes clothing regions)
2. Load garment mask: Convert to 20-channel source parsing  
3. Use parsing alignment to guide optical flow generation
4. Warp garment to fit target upper clothes region

Without trained checkpoint (Stage_2_generator), we use parsing-aware warping:
- Compute center of mass and bounding box of upper clothes regions
- Generate flow field that aligns source to target
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import argparse
from scipy import ndimage

def load_pam_output(pam_npy_path):
    """Load PAM output and extract clothing regions.
    
    PAM output has 20-channel semantic parsing.
    Clothing channels (cloth_list):
      5 = upper clothes (t-shirts, jackets, etc.)
      6 = jumpers
      7 = shirt  
      8 = jacket
      9 = downwear
      12 = vest
    """
    pam_data = np.load(pam_npy_path)
    pam_data = np.squeeze(pam_data)
    
    # Convert to 20-channel if 2D label map
    if pam_data.ndim == 2:
        label_map = pam_data.astype(np.int32)
        pam_20ch = np.zeros((20, pam_data.shape[0], pam_data.shape[1]))
        for c in range(20):
            pam_20ch[c] = (label_map == c).astype(np.float32)
    else:
        pam_20ch = pam_data
    
    # Extract only clothing categories from target person
    cloth_list = [-1, 1, -1, -1, -1, 5, 6, 7, 8, 9, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1]
    target_cloth_20 = np.zeros((20, pam_20ch.shape[1], pam_20ch.shape[2]))
    
    for i in range(20):
        if cloth_list[i] >= 0:
            target_cloth_20[i] = pam_20ch[i]
    
    return target_cloth_20.astype(np.float32)

def load_garment_mask(garment_mask_path, size=(192, 256)):
    """Load garment mask and convert to 20-channel source parsing.
    
    Place mask in clothing categories to match target parsing format.
    """
    garment_mask_pil = Image.open(garment_mask_path).convert("L").resize(size)
    garment_mask_np = np.array(garment_mask_pil).astype(np.float32) / 255.0
    
    # Create 20-channel garment parsing (place mask in clothing channels)
    source_garment_20 = np.zeros((20, size[1], size[0]))
    cloth_list = [-1, 1, -1, -1, -1, 5, 6, 7, 8, 9, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1]
    
    for i in range(20):
        if cloth_list[i] >= 0:
            source_garment_20[i] = garment_mask_np
    
    return source_garment_20.astype(np.float32)

def load_image_tensor(img_path, size=(192, 256)):
    """Load and normalize image to tensor."""
    img = Image.open(img_path).convert('RGB').resize(size)
    img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0)

def compute_parsing_flow(target_parsing, source_parsing, h, w):
    """Generate optical flow by aligning source to target parsing.
    
    Method: Compute center of mass of clothing regions in both,
    then create flow field that deforms source to match target.
    
    Args:
        target_parsing: (20, H, W) - target person's clothing regions
        source_parsing: (20, H, W) - source garment in clothing channels
    
    Returns:
        flow: (2, H, W) - optical flow field (u, v displacements)
    """
    # Use upper clothes channel (index 5) + all clothing channels
    clothing_channels = [5, 6, 7, 8, 9, 12]
    
    # Merge all clothing channels into single masks
    target_mask = np.zeros((h, w))
    source_mask = np.zeros((h, w))
    
    for ch in clothing_channels:
        if ch < len(target_parsing):
            target_mask += target_parsing[ch]
        if ch < len(source_parsing):
            source_mask += source_parsing[ch]
    
    target_mask = np.clip(target_mask, 0, 1)
    source_mask = np.clip(source_mask, 0, 1)
    
    # Compute center of mass for each
    try:
        target_com = ndimage.center_of_mass(target_mask)
        source_com = ndimage.center_of_mass(source_mask)
    except:
        # Fallback if COM computation fails
        target_com = (h / 2, w / 2)
        source_com = (h / 2, w / 2)
    
    print(f"    Target CoM: {target_com}")
    print(f"    Source CoM: {source_com}")
    
    # Create flow field that moves source toward target
    yy, xx = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flow at each pixel: difference between source and target centers
    displacement_y = (target_com[0] - source_com[0]) / float(h)
    displacement_x = (target_com[1] - source_com[1]) / float(w)
    
    # Get bounding boxes to compute scaling
    target_points = np.where(target_mask > 0.5)
    source_points = np.where(source_mask > 0.5)
    
    if len(target_points[0]) > 0 and len(source_points[0]) > 0:
        target_height = target_points[0].max() - target_points[0].min()
        target_width = target_points[1].max() - target_points[1].min()
        source_height = source_points[0].max() - source_points[0].min()
        source_width = source_points[1].max() - source_points[1].min()
        
        scale_y = target_height / (source_height + 1e-6)
        scale_x = target_width / (source_width + 1e-6)
    else:
        scale_x = scale_y = 1.0
    
    print(f"    Scale (Y, X): ({scale_y:.3f}, {scale_x:.3f})")
    
    # Create smooth flow field with Gaussian smoothing
    # Make flow stronger in source garment region, weaker elsewhere
    flow_u = np.zeros((h, w))
    flow_v = np.zeros((h, w))
    
    # Base displacement
    flow_u += displacement_x * source_mask * 20  # Scale for visibility
    flow_v += displacement_y * source_mask * 20
    
    # Add slight scaling deformation
    center_y, center_x = source_com
    yy_centered = (yy - center_x) / float(w)
    xx_centered = (xx - center_x) / float(h)
    
    flow_u += (scale_x - 1.0) * yy_centered * source_mask * 5
    flow_v += (scale_y - 1.0) * xx_centered * source_mask * 5
    
    # Smooth with Gaussian
    from scipy.ndimage import gaussian_filter
    flow_u = gaussian_filter(flow_u, sigma=2.0)
    flow_v = gaussian_filter(flow_v, sigma=2.0)
    
    flow = np.stack([flow_u, flow_v], axis=0)
    return flow

def warp(x, flow):
    """Warp tensor using optical flow."""
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).view(1, 1, H, W).expand(B, -1, -1, -1).float()
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).view(1, 1, H, W).expand(B, -1, -1, -1).float()
    grid = torch.cat([xx, yy], 1) + flow
    grid[:, 0] = 2.0 * grid[:, 0] / max(W - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H - 1, 1) - 1.0
    return F.grid_sample(x, grid.permute(0, 2, 3, 1), align_corners=True)

def main():
    parser = argparse.ArgumentParser(description="FVNT Flow Renderer - Stage 2 (Parsing-Aware Warping)")
    parser.add_argument("--pam_output", required=True, help="Path to PAM output (predicted_parsing_20ch.npy)")
    parser.add_argument("--garment_rgb", required=True, help="Path to garment RGB")
    parser.add_argument("--garment_mask", required=True, help="Path to garment mask")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("  FVNT Stage 2 - Flow Renderer (Parsing-Aware Warping)")
    print("="*70 + "\n")
    print("[*] Proper Stage 2 should use trained ResFlow network")
    print("[*] Without checkpoint, using parsing-guided warping equivalent\n")
    
    # Load inputs
    print("[*] Loading inputs...")
    target_parsing = load_pam_output(args.pam_output)
    source_parsing = load_garment_mask(args.garment_mask, size=(192, 256))
    garment_rgb = load_image_tensor(args.garment_rgb, size=(192, 256))
    
    print(f"    Target parsing shape: {target_parsing.shape}")
    print(f"    Source parsing shape: {source_parsing.shape}")
    print(f"    Garment RGB shape: {garment_rgb.shape}\n")
    
    # Generate optical flow by aligning source garment to target clothing regions
    print("[*] Computing parsing-guided optical flow...")
    print("    This aligns source garment to target upper clothes region")
    flow_np = compute_parsing_flow(target_parsing, source_parsing, h=256, w=192)
    flow = torch.from_numpy(flow_np).float().unsqueeze(0)
    
    print(f"\n    Generated flow shape: {flow.shape}")
    print(f"    Flow U (horizontal) range: [{flow_np[0].min():.4f}, {flow_np[0].max():.4f}]")
    print(f"    Flow V (vertical) range:   [{flow_np[1].min():.4f}, {flow_np[1].max():.4f}]")
    mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    print(f"    Max flow magnitude: {mag.max():.4f} pixels")
    
    # Warp garment using computed flow
    print("\n[*] Warping garment to target upper clothes region...")
    warped_garment = warp(garment_rgb, flow)
    
    # Save warped garment
    warped_np = warped_garment[0].permute(1, 2, 0).detach().numpy()
    warped_np = ((warped_np + 1.0) * 127.5).astype(np.uint8).clip(0, 255)
    warped_img = Image.fromarray(warped_np)
    warped_path = os.path.join(args.output_dir, "warped_garment.png")
    warped_img.save(warped_path)
    print(f"[OK] Saved: {warped_path}")
    
    # Save flow field
    flow_np_save = flow[0].numpy()
    flow_path = os.path.join(args.output_dir, "flow_field.npy")
    np.save(flow_path, flow_np_save)
    print(f"[OK] Saved: {flow_path}")
    
    # Save target parsing visualization
    palette = np.array([
        [0,0,0],[128,0,0],[255,0,0],[0,85,0],[170,0,51],
        [128,64,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],
        [128,128,128],[255,0,0],[255,128,0],[255,255,0],[0,255,0],
        [0,0,255],[255,0,255],[0,255,255],[128,255,0],[0,128,255]
    ], dtype=np.uint8)
    target_label = np.argmax(target_parsing, axis=0).astype(np.uint8)
    target_label = np.minimum(target_label, len(palette)-1)
    target_rgb = palette[target_label].astype(np.uint8)
    target_rgb_img = Image.fromarray(target_rgb, mode='RGB')
    target_rgb_path = os.path.join(args.output_dir, "target_parsing_rgb.png")
    target_rgb_img.save(target_rgb_path)
    print(f"[OK] Saved: {target_rgb_path}")
    
    # Save flow visualization
    flow_mag = np.sqrt(flow_np[0]**2 + flow_np[1]**2)
    flow_vis = (np.tanh(flow_mag / 0.05) * 127.5 + 127.5).astype(np.uint8)
    flow_vis_img = Image.fromarray(flow_vis, mode='L')
    flow_vis_path = os.path.join(args.output_dir, "flow_magnitude.png")
    flow_vis_img.save(flow_vis_path)
    print(f"[OK] Saved: {flow_vis_path}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Flow Renderer complete!")
    print("="*70)
    print("\nExplanation:")
    print("  • Target parsing: PAM output showing target person's upper clothes")
    print("  • Source parsing: Input garment mask converted to 20-channel format")
    print("  • Optical flow: Computed to align source → target regions")
    print("  • Warped garment: Garment deformed to fit target body shape")
    print("\nProduction:")
    print("  • Without trained Stage_2_generator checkpoint, flow is approximated")
    print("  • Trained model would learn more realistic warping from data")
    print("\nNext step: Run FEM (Stage 3) to synthesize final outfit")
    print("="*70 + "\n")

if __name__ == "__main__":
    sys.exit(main())
