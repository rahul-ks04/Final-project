#!/usr/bin/env python3
"""
FVNT Stage 2 - Flow Renderer (Proper Implementation)
Uses the actual Stage_2_generator network to predict optical flow.

Correct pipeline:
1. Load PAM output: 20-channel target person parsing
2. Extract upper clothes regions from PAM → target_mask
3. Load source garment mask
4. Create 20-channel tensors placing masks in clothing channels [4,5,6,7]
5. Feed both through trained Stage_2_generator
6. Warp high-resolution garment using predicted flow
7. Save warped garment for FEM (Stage 3)
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import argparse

# Add FVNT to path for model imports
FVNT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FVNT"))
if FVNT_DIR not in sys.path:
    sys.path.insert(0, FVNT_DIR)

# Inject DCN module (zero-build, pure Python fallback)
def inject_dcn():
    """Create lightweight DeformConvPack without compilation."""
    DCN_FOLDER = os.path.join(FVNT_DIR, "Deformable")
    if os.path.isdir(DCN_FOLDER):
        import shutil
        shutil.rmtree(DCN_FOLDER)
    os.makedirs(DCN_FOLDER, exist_ok=True)

    with open(os.path.join(DCN_FOLDER, "__init__.py"), "w") as f:
        f.write("from .modules import DeformConvPack")

    modules_py = """
import torch
from torch import nn
from torchvision.ops import deform_conv2d
import math

class DeformConvPack(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True, lr_mult=0.1):
        super(DeformConvPack, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv_offset = nn.Conv2d(in_channels,
                                     deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1],
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size: n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, x):
        offset = self.conv_offset(x)
        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=self.stride, padding=self.padding, dilation=self.dilation)
"""
    with open(os.path.join(DCN_FOLDER, "modules.py"), "w") as f:
        f.write(modules_py)

inject_dcn()

# Now import the model
from mine.network_stage_2_mine_x2_resflow import Stage_2_generator

# Constants
H_MODEL, W_MODEL = 256, 192
H_HD, W_HD = 1024, 768

def load_pam_output(pam_npy_path):
    """Load PAM output (20-channel parsing)."""
    pam_data = np.load(pam_npy_path)
    pam_data = np.squeeze(pam_data)
    
    if pam_data.ndim == 2:
        # If 2D (label map), convert to 20-channel one-hot
        label_map = pam_data.astype(np.int32)
        pam_20ch = np.zeros((20, pam_data.shape[0], pam_data.shape[1]))
        for c in range(20):
            pam_20ch[c] = (label_map == c).astype(np.float32)
    else:
        pam_20ch = pam_data
    
    return pam_20ch.astype(np.float32)

def extract_target_mask(pam_20ch, h_model=H_MODEL, w_model=W_MODEL):
    """Extract t-shirt target mask from PAM output.
    
    Use ONLY channel 5 (upper_clothes) for clean upper-body-only fitting.
    This ensures the garment warps only to the torso/shoulder region.
    
    Returns: Binary mask of upper clothing region only.
    """
    # Resize PAM to model size if needed
    if pam_20ch.shape[1] != h_model or pam_20ch.shape[2] != w_model:
        pam_pil = Image.fromarray(np.argmax(pam_20ch, axis=0).astype(np.uint8)).resize((w_model, h_model), Image.NEAREST)
        pam_resized_label = np.array(pam_pil)
        pam_20ch_resized = np.zeros((20, h_model, w_model))
        for c in range(20):
            pam_20ch_resized[c] = (pam_resized_label == c).astype(np.float32)
    else:
        pam_20ch_resized = pam_20ch
    
    # Use ONLY upper_clothes channel (ch 5) for clean t-shirt fitting
    target_mask = pam_20ch_resized[5].astype(np.float32)
    target_mask = np.clip(target_mask, 0, 1).astype(np.float32)
    return target_mask

def load_garment_mask(garment_mask_path, h_model=H_MODEL, w_model=W_MODEL):
    """Load source garment mask."""
    garment_mask_pil = Image.open(garment_mask_path).convert("L").resize((w_model, h_model), Image.NEAREST)
    garment_mask = np.array(garment_mask_pil).astype(np.float32) / 255.0
    return garment_mask

def create_20ch_input(mask, h_model=H_MODEL, w_model=W_MODEL):
    """Convert binary mask to 20-channel tensor for Stage 2 input.
    
    Place the mask in clothing channels [4, 5, 6, 7] as done in training.
    """
    input_20ch = np.zeros((20, h_model, w_model), dtype=np.float32)
    # Clothing channels from training
    for i in [4, 5, 6, 7]:  # Match training format
        input_20ch[i] = mask
    return input_20ch

def warp_high_res(img_t, low_res_flow, device):
    """Warp high-resolution image using low-resolution predicted flow.
    
    Strategy:
    1. Upsample flow to high-res
    2. Rescale flow values to match new pixel grid
    3. Apply grid_sample warp
    """
    B, C, H_hr, W_hr = img_t.shape
    _, _, H_lr, W_lr = low_res_flow.shape
    
    # Upsample flow to high-res
    flow_hr = F.interpolate(low_res_flow, size=(H_hr, W_hr), mode='bilinear', align_corners=True)
    
    # Rescale flow values to match high-res grid
    flow_hr[:, 0] = flow_hr[:, 0] * (W_hr / W_lr)
    flow_hr[:, 1] = flow_hr[:, 1] * (H_hr / H_lr)
    
    # Create coordinate grid
    gx = torch.arange(W_hr, device=device).view(1, -1).repeat(H_hr, 1).view(1, 1, H_hr, W_hr).expand(B, -1, -1, -1)
    gy = torch.arange(H_hr, device=device).view(-1, 1).repeat(1, W_hr).view(1, 1, H_hr, W_hr).expand(B, -1, -1, -1)
    grid = torch.cat([gx, gy], 1).float() + flow_hr
    
    # Normalize to [-1, 1] for grid_sample
    grid[:, 0] = 2.0 * grid[:, 0] / max(W_hr - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H_hr - 1, 1) - 1.0
    
    return F.grid_sample(img_t, grid.permute(0, 2, 3, 1), align_corners=True)

def main():
    parser = argparse.ArgumentParser(description="FVNT Stage 2 - Proper Flow Renderer with Trained Model")
    parser.add_argument("--pam_output", required=True, help="Path to PAM output (predicted_parsing_20ch.npy)")
    parser.add_argument("--garment_rgb", required=True, help="Path to garment RGB (high-res)")
    parser.add_argument("--garment_mask", required=True, help="Path to garment mask")
    parser.add_argument("--checkpoint", help="Path to Stage 2 checkpoint (auto-detects if not provided)")
    parser.add_argument("--output_dir", default="outputs/flow_stage2", help="Output directory")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("\n" + "="*70)
    print("  FVNT Stage 2 - Flow Renderer (Trained Model)")
    print("="*70)
    print(f"\n[*] Device: {device}")
    print(f"[*] Input PAM: {args.pam_output}")
    print(f"[*] Garment RGB: {args.garment_rgb}")
    print(f"[*] Garment mask: {args.garment_mask}\n")
    
    # Initialize model
    print("[*] Loading Stage_2_generator...")
    model = Stage_2_generator(input_dim_1=20).to(device)
    model.eval()
    
    # Patch warp method to use device instead of hardcoded .cuda()
    original_warp = model.warp
    def patched_warp(x, flo):
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1).to(device)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1).to(device)
        grid = torch.cat((xx, yy), 1).float()
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(x, vgrid)
        mask = torch.ones(x.size()).to(device)
        mask = F.grid_sample(mask, vgrid, mode='bilinear')
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1
        return output
    model.warp = patched_warp
    
    # Find checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        candidates = [
            os.path.join(FVNT_DIR, "model", "stage2_model"),
            os.path.join(FVNT_DIR, "model", "stage2_model.pth"),
            "FVNT/model/stage2_model",
            "FVNT/model/stage2_model.pth",
        ]
        for cand in candidates:
            if os.path.isfile(cand):
                checkpoint_path = cand
                break
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        print(f"[*] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'G' in ckpt:
            model.load_state_dict(ckpt['G'])
        else:
            model.load_state_dict(ckpt)
        print("[OK] Checkpoint loaded successfully\n")
    else:
        print("[!] WARNING: No checkpoint found. Using untrained model.")
        print("[!] Results will not be realistic. Provide checkpoint with --checkpoint\n")
    
    # Load inputs
    print("[*] Loading inputs...")
    pam_20ch = load_pam_output(args.pam_output)
    target_mask = extract_target_mask(pam_20ch, H_MODEL, W_MODEL)
    garment_mask = load_garment_mask(args.garment_mask, H_MODEL, W_MODEL)
    
    print(f"    PAM shape: {pam_20ch.shape}")
    print(f"    Target mask: {target_mask.shape}, coverage: {(target_mask > 0.5).sum() / (H_MODEL*W_MODEL)*100:.1f}%")
    print(f"    Garment mask: {garment_mask.shape}, coverage: {(garment_mask > 0.5).sum() / (H_MODEL*W_MODEL)*100:.1f}%")
    
    # Create 20-channel inputs for Stage 2
    input_target = create_20ch_input(target_mask, H_MODEL, W_MODEL)
    input_source = create_20ch_input(garment_mask, H_MODEL, W_MODEL)
    
    # Convert to tensors
    input_target_t = torch.from_numpy(input_target).float().unsqueeze(0).to(device)
    input_source_t = torch.from_numpy(input_source).float().unsqueeze(0).to(device)
    
    print(f"\n[*] Running Stage 2 generator...")
    print(f"    Input target shape: {input_target_t.shape}")
    print(f"    Input source shape: {input_source_t.shape}")
    
    # Predict flow
    with torch.no_grad():
        flow_list, res_flow_list = model(input_target_t, input_source_t)
    
    # Use finest flow (last in pyramid)
    flow_final = flow_list[-1]  # Shape: (1, 2, H_MODEL, W_MODEL)
    
    print(f"    Output flow shape: {flow_final.shape}")
    flow_mag = torch.sqrt(flow_final[0, 0]**2 + flow_final[0, 1]**2)
    print(f"    Flow magnitude: min={flow_mag.min():.4f}, max={flow_mag.max():.4f}, mean={flow_mag.mean():.4f} pixels")
    
    # Load high-resolution garment for warping
    print(f"\n[*] Loading high-resolution garment...")
    garment_hd_pil = Image.open(args.garment_rgb).convert('RGB').resize((W_HD, H_HD), Image.LANCZOS)
    garment_hd_np = np.array(garment_hd_pil).astype(np.float32) / 127.5 - 1.0
    garment_hd_t = torch.from_numpy(garment_hd_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    print(f"    High-res garment shape: {garment_hd_t.shape}")
    
    # Warp garment
    print(f"\n[*] Warping high-resolution garment...")
    warped_hd = warp_high_res(garment_hd_t, flow_final, device)
    
    # Save outputs
    print(f"\n[*] Saving outputs...")
    
    # Save warped garment
    warped_np = warped_hd[0].permute(1, 2, 0).cpu().numpy()
    warped_np = ((warped_np + 1.0) * 127.5).astype(np.uint8).clip(0, 255)
    warped_img = Image.fromarray(warped_np)
    warped_path = os.path.join(args.output_dir, "warped_garment.png")
    warped_img.save(warped_path)
    print(f"    -> {warped_path}")
    
    # Save flow field
    flow_np = flow_final[0].cpu().numpy()
    flow_path = os.path.join(args.output_dir, "flow_field.npy")
    np.save(flow_path, flow_np)
    print(f"    -> {flow_path} (shape: {flow_np.shape})")
    
    # Save flow magnitude visualization
    flow_mag_vis = (np.tanh((flow_mag.cpu().numpy() / 5.0)) * 127.5 + 127.5).astype(np.uint8)
    flow_mag_img = Image.fromarray(flow_mag_vis, mode='L')
    flow_mag_path = os.path.join(args.output_dir, "flow_magnitude.png")
    flow_mag_img.save(flow_mag_path)
    print(f"    -> {flow_mag_path}")
    
    # Save target mask visualization
    target_mask_vis = (target_mask * 255).astype(np.uint8)
    target_mask_img = Image.fromarray(target_mask_vis, mode='L')
    target_mask_path = os.path.join(args.output_dir, "target_mask.png")
    target_mask_img.save(target_mask_path)
    print(f"    -> {target_mask_path}")
    
    # Save garment mask visualization
    garment_mask_vis = (garment_mask * 255).astype(np.uint8)
    garment_mask_img = Image.fromarray(garment_mask_vis, mode='L')
    garment_mask_path = os.path.join(args.output_dir, "source_mask.png")
    garment_mask_img.save(garment_mask_path)
    print(f"    -> {garment_mask_path}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Stage 2 Flow Renderer complete!")
    print("="*70)
    print("\nKey outputs:")
    print(f"  • warped_garment.png - Garment warped to fit target person's body")
    print(f"  • flow_field.npy - (2, {H_MODEL}, {W_MODEL}) optical flow tensor")
    print(f"  • flow_magnitude.png - Visualization of flow strength")
    print(f"\nNext step: Run Stage 3 (FEM) to compose final outfit")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
