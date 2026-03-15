"""
FVNT Flow Renderer (Stage 2)
Generates optical flow field from target parsing (PAM output) and source garment,
then warps the garment for FEM input.

Pipeline:
1. Load PAM output (predicted_parsing_20ch.npy) as target person clothing parsing
2. Load garment mask and convert to 20-channel source clothing parsing
3. Run Stage 2 flow estimation network to generate optical flow
4. Warp source garment using the flow field
5. Save warped garment and flow visualization
"""

import os
import sys
import torch
import shutil
import numpy as np
import argparse
import cv2
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math

# Add FVNT to path
FVNT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "FVNT"))
if FVNT_DIR not in sys.path:
    sys.path.insert(0, FVNT_DIR)

# 1. Zero-Build DCN Injection (Pure Python fallback)
def inject_dcn():
    DCN_FOLDER = os.path.join(FVNT_DIR, "Deformable")
    if os.path.isdir(DCN_FOLDER):
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

# Constants
H_MODEL, W_MODEL = 256, 192
H_HD, W_HD = 1024, 768

def warp(x, flow):
    """Warp tensor x using optical flow field."""
    B, C, H, W = x.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float().to(x.device)
    vgrid = grid + flow
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    return F.grid_sample(x, vgrid, align_corners=True)

def flow_to_image(flow):
    """Convert optical flow to color visualization image (Middlebury color wheel encoding)."""
    flow_image = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    rad = np.sqrt(u**2 + v**2)
    rad_max = np.max(rad)
    
    if rad_max > 0:
        u = u / rad_max
        v = v / rad_max
    
    # Middlebury color wheel
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    
    col = 0
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(0, CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(0, MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    
    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            dx = u[i, j]
            dy = v[i, j]
            rad = np.sqrt(dx**2 + dy**2)
            a = np.arctan2(-dy, -dx) / np.pi
            fk = (a + 1.0) / 2.0 * (ncols - 1)
            k0 = np.floor(fk).astype(int)
            k1 = k0 + 1
            if k1 == ncols:
                k1 = 0
            f = fk - k0
            
            for c in range(3):
                tmp0 = colorwheel[k0, c]
                tmp1 = colorwheel[k1, c]
                col_tmp = (1 - f) * tmp0 + f * tmp1
                
                if rad <= 1:
                    col_tmp = 255 - rad * (255 - col_tmp)
                else:
                    col_tmp = col_tmp * 0.75
                
                flow_image[i, j, c] = np.uint8(col_tmp)
    
    return flow_image

def load_pam_output(pam_npy_path, device):
    """Load PAM output (20-channel parsing) and convert to target clothing parsing."""
    pam_data = np.load(pam_npy_path)  # Shape: (20, H, W) or (1, 20, H, W)
    pam_data = np.squeeze(pam_data)
    
    if pam_data.ndim == 2:
        # If 2D (label map), convert to 20-channel one-hot
        label_map = pam_data.astype(np.int32)
        pam_20ch = np.zeros((20, pam_data.shape[0], pam_data.shape[1]))
        for c in range(20):
            pam_20ch[c] = (label_map == c).astype(np.float32)
    else:
        pam_20ch = pam_data
    
    # Only keep upper-body clothing categories for tops try-on.
    # Channel indices correspond to the 20-class CIHP/LIP schema.
    upper_body_channels = [5, 6, 7]
    target_cloth_20 = np.zeros((20, pam_20ch.shape[1], pam_20ch.shape[2]))

    for i in upper_body_channels:
        target_cloth_20[i] = pam_20ch[i]

    # Clean noisy cloth support from PAM before stage-2 flow estimation.
    cloth_union = np.sum(target_cloth_20[upper_body_channels], axis=0)
    cloth_mask = (cloth_union > 0.5).astype(np.uint8)
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
    cloth_mask = cv2.morphologyEx(cloth_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    # Keep only the largest connected component (main torso garment).
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cloth_mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_label = int(np.argmax(areas)) + 1
        cloth_mask = (labels == keep_label).astype(np.uint8)

    for i in upper_body_channels:
        target_cloth_20[i] = target_cloth_20[i] * cloth_mask.astype(np.float32)
    
    return torch.from_numpy(target_cloth_20).float().unsqueeze(0).to(device)

def load_garment_mask(garment_mask_path, device, h_model=H_MODEL, w_model=W_MODEL):
    """Load garment mask and convert to 20-channel source clothing parsing."""
    garment_mask_pil = Image.open(garment_mask_path).convert("L").resize((w_model, h_model))
    garment_mask_np = np.array(garment_mask_pil).astype(np.float32) / 255.0
    
    # Create 20-channel garment parsing for upper-body clothing categories only.
    source_garment_20 = np.zeros((20, h_model, w_model))
    upper_body_channels = [5, 6, 7]
    
    for i in upper_body_channels:
        source_garment_20[i] = garment_mask_np
    
    return torch.from_numpy(source_garment_20).float().unsqueeze(0).to(device)


def load_binary_mask_tensor(garment_mask_path, device, h_model=H_MODEL, w_model=W_MODEL):
    garment_mask_pil = Image.open(garment_mask_path).convert("L").resize((w_model, h_model), Image.NEAREST)
    garment_mask_np = (np.array(garment_mask_pil).astype(np.float32) / 255.0)
    garment_mask_np = (garment_mask_np > 0.5).astype(np.float32)
    return torch.from_numpy(garment_mask_np).float().unsqueeze(0).unsqueeze(0).to(device)

def load_garment_rgb(garment_rgb_path, device, h_model=H_MODEL, w_model=W_MODEL):
    """Load garment RGB image."""
    garment_pil = Image.open(garment_rgb_path).convert("RGB").resize((w_model, h_model))
    garment_np = np.array(garment_pil).astype(np.float32) / 127.5 - 1.0
    garment_tensor = torch.from_numpy(garment_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return garment_tensor

def main():
    parser = argparse.ArgumentParser(description="FVNT Flow Renderer Script (Stage 2)")
    parser.add_argument("--pam_output", required=True, help="Path to PAM output: predicted_parsing_20ch.npy")
    parser.add_argument("--garment_rgb", required=True, help="Path to garment RGB image")
    parser.add_argument("--garment_mask", required=True, help="Path to garment mask")
    parser.add_argument("--checkpoint", help="Path to Stage 2 model checkpoint (optional for inference)")
    parser.add_argument("--output_dir", default="output_flow_renderer", help="Directory to save results")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU and use CPU")
    
    args = parser.parse_args()
    
    device = torch.device('cpu' if args.no_gpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[*] Using device: {device}")
    
    # Inject DCN before importing model
    inject_dcn()
    from mine.network_stage_2_mine_x2_resflow import Stage_2_generator
    
    # Load or initialize Stage 2 Flow Renderer
    print("[*] Initializing Stage 2 Flow Renderer...")
    flow_renderer = Stage_2_generator(input_dim_1=20).to(device)
    flow_renderer.eval()
    
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"[*] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'G' in ckpt:
            flow_renderer.load_state_dict(ckpt['G'])
        else:
            flow_renderer.load_state_dict(ckpt)
        print("[OK] Checkpoint loaded.")
    else:
        print("[!] No checkpoint provided or file not found. Using untrained model.")
    
    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load inputs
    print("[*] Loading inputs...")
    
    # 1. Load PAM output as target parsing
    target_parsing_20 = load_pam_output(args.pam_output, device)
    print(f"    Target (PAM) parsing shape: {target_parsing_20.shape}")
    
    # 2. Load garment mask as source parsing
    source_parsing_20 = load_garment_mask(args.garment_mask, device, H_MODEL, W_MODEL)
    print(f"    Source garment parsing shape: {source_parsing_20.shape}")
    
    # 3. Load garment RGB for warping
    garment_rgb = load_garment_rgb(args.garment_rgb, device, H_MODEL, W_MODEL)
    print(f"    Garment RGB shape: {garment_rgb.shape}")

    # 3b. Load binary garment mask for exact warped support
    garment_mask_binary = load_binary_mask_tensor(args.garment_mask, device, H_MODEL, W_MODEL)
    
    # 4. Run Stage 2 to predict optical flow
    print("[*] Predicting optical flow...")
    with torch.no_grad():
        flow_list, res_flow_list = flow_renderer(target_parsing_20, source_parsing_20)
    
    # Use the finest flow field (index -1)
    flow_final = flow_list[-1]
    print(f"    Flow field shape: {flow_final.shape}")
    
    # 5. Warp garment using predicted flow
    print("[*] Warping garment with predicted flow...")
    warped_garment = warp(garment_rgb, flow_final)
    warped_mask = warp(garment_mask_binary, flow_final)
    warped_mask = (warped_mask > 0.5).float()
    warped_garment = warped_garment * warped_mask
    
    # 6. Save outputs
    print("[*] Saving outputs...")
    
    # Save warped garment RGB
    warped_np = warped_garment[0].permute(1, 2, 0).cpu().numpy()
    warped_np = ((warped_np + 1.0) * 127.5).astype(np.uint8).clip(0, 255)
    warped_img = Image.fromarray(warped_np, mode='RGB')
    warped_path = os.path.join(args.output_dir, "warped_garment.png")
    warped_img.save(warped_path)
    print(f"    Saved: {warped_path}")

    warped_mask_np = (warped_mask[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
    warped_mask_path = os.path.join(args.output_dir, "warped_mask.png")
    Image.fromarray(warped_mask_np, mode='L').save(warped_mask_path)
    print(f"    Saved: {warped_mask_path}")
    
    # Save final flow field as numpy for Stage 3 FEM
    flow_final_np = flow_final[0].cpu().numpy()
    flow_final_npy_path = os.path.join(args.output_dir, "flow_field.npy")
    np.save(flow_final_npy_path, flow_final_np)
    print(f"    Saved: {flow_final_npy_path}")
    
    # Save input parsing visualizations
    target_np = target_parsing_20[0].cpu().numpy()
    palette = np.array([
        [0,0,0],[128,0,0],[255,0,0],[0,85,0],[170,0,51],
        [255,85,0],[0,0,85],[0,119,221],[85,85,0],[0,85,85],
        [85,51,0],[52,86,128],[0,128,0],[0,0,255],[51,170,221],
        [0,255,255],[85,255,170],[170,255,85],[255,255,0],[255,170,0]
    ], dtype=np.uint8)

    # Reconstruct a human-readable cloth-only visualization from the ORIGINAL PAM argmax,
    # preserving background for non-clothing pixels.
    pam_raw = np.squeeze(np.load(args.pam_output))
    if pam_raw.ndim == 3:
        pam_label = np.argmax(pam_raw, axis=0).astype(np.uint8)
    else:
        pam_label = pam_raw.astype(np.uint8)
    cloth_ids = np.array([5, 6, 7], dtype=np.uint8)
    cloth_mask = np.isin(pam_label, cloth_ids)
    target_label = np.where(cloth_mask, pam_label, 0).astype(np.uint8)

    target_rgb = palette[np.minimum(target_label, len(palette)-1)]
    target_rgb_img = Image.fromarray(target_rgb, mode='RGB')
    target_rgb_path = os.path.join(args.output_dir, "target_parsing_rgb.png")
    target_rgb_img.save(target_rgb_path)
    print(f"    Saved: {target_rgb_path}")

    target_mask_path = os.path.join(args.output_dir, "target_cloth_mask.png")
    Image.fromarray((cloth_mask.astype(np.uint8) * 255), mode='L').save(target_mask_path)
    print(f"    Saved: {target_mask_path}")
    
    source_np = source_parsing_20[0].cpu().numpy()
    source_label = np.argmax(source_np, axis=0).astype(np.uint8)
    source_rgb = palette[np.minimum(source_label, len(palette)-1)]
    source_rgb_img = Image.fromarray(source_rgb, mode='RGB')
    source_rgb_path = os.path.join(args.output_dir, "source_parsing_rgb.png")
    source_rgb_img.save(source_rgb_path)
    print(f"    Saved: {source_rgb_path}")
    
    print(f"\n[SUCCESS] Flow Renderer complete! Results saved to: {args.output_dir}")
    print(f"\nKey outputs for FEM Stage 3:")
    print(f"  - Warped garment: {warped_path}")
    print(f"  - Optical flow: {flow_final_npy_path}")
    print(f"  - Target parsing: {target_rgb_path}")

if __name__ == "__main__":
    main()
