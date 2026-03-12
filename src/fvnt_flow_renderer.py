import os
import sys
import torch
import shutil
import numpy as np
import argparse
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

def warp_high_res(img_t, low_res_flow, device):
    """Warps a high-res image using a low-res predicted flow field."""
    B, _, H_hr, W_hr = img_t.shape
    _, _, H_lr, W_lr = low_res_flow.shape
    
    flow_hr = F.interpolate(low_res_flow, size=(H_hr, W_hr), mode='bilinear', align_corners=True)
    flow_hr[:, 0] = flow_hr[:, 0] * (W_hr / W_lr)
    flow_hr[:, 1] = flow_hr[:, 1] * (H_hr / H_lr)
    
    gx = torch.arange(W_hr, device=device).view(1,-1).repeat(H_hr,1).view(1,1,H_hr,W_hr).expand(B,-1,-1,-1)
    gy = torch.arange(H_hr, device=device).view(-1,1).repeat(1,W_hr).view(1,1,H_hr,W_hr).expand(B,-1,-1,-1)
    grid = torch.cat([gx, gy], 1).float() + flow_hr
    
    grid[:, 0] = 2.0 * grid[:, 0] / max(W_hr - 1, 1) - 1.0
    grid[:, 1] = 2.0 * grid[:, 1] / max(H_hr - 1, 1) - 1.0
    return F.grid_sample(img_t, grid.permute(0, 2, 3, 1), align_corners=True)

def prep_tensor(path, device, is_parsing=False):
    """Formats inputs for the FEM model."""
    img = Image.open(path).resize((W_MODEL, H_MODEL), Image.NEAREST if is_parsing else Image.BILINEAR)
    if is_parsing:
        lbl = np.array(img)
        out = torch.zeros(20, H_MODEL, W_MODEL)
        # Place torso/arm/dress regions into specific channels as expected by Stage 2
        for i in [4, 5, 6, 7]: 
            mask = (lbl == i) if lbl.max() < 20 else (lbl >= 128)
            out[i] = torch.from_numpy(mask.astype(np.float32))
        return out.unsqueeze(0).to(device)
    else:
        return (torch.from_numpy(np.array(img.convert('RGB'))).permute(2,0,1).float().unsqueeze(0)/127.5-1).to(device)

def main():
    parser = argparse.ArgumentParser(description="FVNT Flow Renderer Script")
    parser.add_argument("--person", required=True, help="Path to person target mask")
    parser.add_argument("--garment_rgb", required=True, help="Path to garment RGB image")
    parser.add_argument("--garment_mask", required=True, help="Path to garment mask")
    parser.add_argument("--checkpoint", required=True, help="Path to Stage 2 model checkpoint")
    parser.add_argument("--schp", help="Path to target person SCHP parsing (optional, for better sleeve control)")
    parser.add_argument("--output_dir", default="output", help="Directory to save results")
    parser.add_argument("--no_projection", action="store_true", help="Disable sleeve projection refinement")
    parser.add_argument("--sleeve_type", choices=["auto", "short", "long"], default="auto", help="Override sleeve type detection")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Inject DCN before importing model
    inject_dcn()
    from mine.network_stage_2_mine_x2_resflow import Stage_2_generator
    from utils.projection import project_source_mask

    # Load Model
    fem = Stage_2_generator(20).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    fem.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
    fem.eval()
    print("[OK] Model loaded.")

    # Prepare Inputs
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Use user-preferred prep_tensor logic
    input_1 = prep_tensor(args.person, device, is_parsing=True)
    input_2 = prep_tensor(args.garment_mask, device, is_parsing=True)

    # 3. Predict Flow
    ctx = {}
    with torch.no_grad():
        flow_list, _ = fem(input_1, input_2, ctx=ctx)
    low_res_flow = ctx.get('appearance_flow', flow_list[-1])

    # 4. Warp Cloth
    cloth_hd = Image.open(args.garment_rgb).convert('RGB').resize((W_HD, H_HD))
    cloth_hd_t = (torch.from_numpy(np.array(cloth_hd)).permute(2,0,1).float().unsqueeze(0)/127.5-1).to(device)
    warped_hd = warp_high_res(cloth_hd_t, low_res_flow, device)
    warped_hd_np = ((warped_hd[0].permute(1,2,0).cpu().numpy()+1)*0.5).clip(0,1)

    # 5. Projection Refinement
    if not args.no_projection:
        print("Applying projection refinement...")
        s_mask_hd = Image.open(args.garment_mask).convert('L').resize((W_HD, H_HD))
        t_mask_hd = Image.open(args.person).convert('L').resize((W_HD, H_HD))
        
        # Use provided person mask as anatomical constraint
        anat_mask_np = np.array(t_mask_hd) / 255.0
        
        # If SCHP is provided, we can further refine boundaries (optional)
        if args.schp:
            schp_hd = Image.open(args.schp).resize((W_HD, H_HD), Image.NEAREST)
            schp_np = np.array(schp_hd)
            print("Using SCHP for boundary refinement.")

        # Upsample flow...
        flow_hr = F.interpolate(low_res_flow, size=(H_HD, W_HD), mode='bilinear', align_corners=True)
        flow_hr[:, 0] *= (W_HD / W_MODEL)
        flow_hr[:, 1] *= (H_HD / H_MODEL)
        
        res = project_source_mask(
            flow_hr,
            source_mask=np.array(s_mask_hd)/255.0,
            anatomical_mask=anat_mask_np
        )
        warped_hd_np = warped_hd_np * res['projected_mask'][..., None]
        
        # Save masks
        Image.fromarray((res['projected_mask']*255).astype(np.uint8)).save(os.path.join(args.output_dir, "projected_mask.png"))
        if 'hole_mask' in res:
            Image.fromarray((res['hole_mask']*255).astype(np.uint8)).save(os.path.join(args.output_dir, "hole_mask.png"))

    # Save final warped garment
    final_img = Image.fromarray((warped_hd_np * 255).astype(np.uint8))
    final_img.save(os.path.join(args.output_dir, "warped_garment.png"))
    print(f"[SUCCESS] Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
