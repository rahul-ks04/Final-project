import os
import cv2
import numpy as np
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# LIP / SCHP label groups
# ---------------------------------------------------------------------------
# Preservation zone — these pixels are NEVER touched, always taken from original
PRESERVE_LABELS  = [1, 2, 4, 13]   # Hat, Hair, Sunglasses, Face
# Old garment region — erase before pasting new cloth
GARMENT_LABELS   = [5, 6, 7, 10, 11]  # Upper-clothes, Dress, Coat, Jumpsuit, Scarf
# Arm/hand labels — erased only where new garment covers them
ARM_LABELS       = [14, 15]         # Left-arm, Right-arm
# Lower body — never touched
LOWER_LABELS     = [9, 12, 16, 17, 18, 19]  # Pants, Skirt, Left-leg, Right-leg, Left-shoe, Right-shoe


# ---------------------------------------------------------------------------
# Generator Architecture (Copied from Flow-Style-VTON for self-containment)
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, norm_layer=nn.BatchNorm2d):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU(True)
        if norm_layer == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, in_features, 3, 1, 1, bias=False),
                norm_layer(in_features)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out


class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=2, padding=1, bias=use_bias)
        res_downconv = [ResidualBlock(inner_nc, norm_layer), ResidualBlock(inner_nc, norm_layer)]
        res_upconv = [ResidualBlock(outer_nc, norm_layer), ResidualBlock(outer_nc, norm_layer)]

        downrelu = nn.ReLU(True)
        uprelu = nn.ReLU(True)
        if norm_layer != None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)

        if outermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            up = [upsample, upconv]
            model = down + [submodule] + up
        elif innermost:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            down = [downconv, downrelu] + res_downconv
            if norm_layer == None:
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                up = [upsample, upconv, upnorm, uprelu] + res_upconv
            model = down + up
        else:
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upconv = nn.Conv2d(inner_nc*2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
            if norm_layer == None:
                down = [downconv, downrelu] + res_downconv
                up = [upsample, upconv, uprelu] + res_upconv
            else:
                down = [downconv, downnorm, downrelu] + res_downconv
                up = [upsample, upconv, upnorm, uprelu] + res_upconv

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class ResUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGenerator, self).__init__()
        unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = ResUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = ResUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = ResUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        self.model = unet_block

    def forward(self, input):
        return self.model(input)


def load_rgba(path, target_size=None):
    """Load image as float32 RGBA [0,1]. target_size = (W, H)."""
    img = Image.open(path).convert("RGBA")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_rgb(path, target_size=None):
    """Load image as float32 RGB [0,1]. target_size = (W, H)."""
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_mask(path, target_size=None, threshold=0.5):
    """Load image as binary float32 mask [0 or 1]. target_size = (W, H)."""
    img = Image.open(path).convert("L")
    if target_size:
        img = img.resize(target_size, Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return (arr > threshold).astype(np.float32)


def load_parse(path, target_size=None):
    """Load SCHP parsing map as uint8. target_size = (W, H)."""
    img = Image.open(path)
    if target_size:
        img = img.resize(target_size, Image.NEAREST)
    return np.array(img).astype(np.uint8)


def soft_mask(binary_mask, blur_radius=7):
    """
    Feather a binary mask with a Gaussian blur for smooth blending edges.
    blur_radius must be odd.
    """
    r = blur_radius | 1  # ensure odd
    return cv2.GaussianBlur(binary_mask.astype(np.float32), (r, r), 0)


def build_label_mask(parse, labels):
    """Return binary mask where any of `labels` is present in `parse`."""
    mask = np.zeros(parse.shape, dtype=np.uint8)
    for lbl in labels:
        mask |= (parse == lbl).astype(np.uint8)
    return mask.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Layered Virtual Try-On Compositor (no GAN inpainting)"
    )
    # --- inputs shared with old interface so master_pipeline.py needs no changes ---
    parser.add_argument("--agnostic",      required=True,  help="Path to agnostic person image (kept for compatibility, not used for synthesis)")
    parser.add_argument("--original",      required=True,  help="Path to original person image WITHOUT background (rembg RGBA .png)")
    parser.add_argument("--agnostic_mask", required=True,  help="Path to binary agnostic mask (kept for compatibility)")
    parser.add_argument("--warped_cloth",  required=True,  help="Path to warped garment RGB image")
    parser.add_argument("--warped_mask",   required=True,  help="Path to warped garment binary mask")
    parser.add_argument("--parse",         default=None,   help="Path to SCHP parsing map of the person (strongly recommended)")
    parser.add_argument("--densepose",     default=None,   help="Path to DensePose image (recommended for hand protection)")
    parser.add_argument("--output_path",   required=True,  help="Path to save the final try-on result (.png)")
    # --- compositing controls ---
    parser.add_argument("--feather",       type=int, default=21,
                        help="Gaussian blur radius (px) for garment mask feathering. Larger = softer edge.")
    parser.add_argument("--garment_dilation", type=int, default=5,
                        help="Morphological dilation (px) applied to old-garment erase mask before erasing.")
    parser.add_argument("--preserve_feather", type=int, default=9,
                        help="Feather radius for the hard-preserve zone (face/hair) copy-back.")
    # kept for master_pipeline CLI compatibility
    parser.add_argument("--checkpoint",    default=None,   help="Path to generator checkpoint (for inpainting mode)")
    parser.add_argument("--inpaint_skin", action="store_true", help="Use GAN inpainting to fill skin holes (e.g. for full-to-half sleeve)")
    parser.add_argument("--initial_sleeve", choices=['none', 'half', 'full'], default='full', help="Initial sleeve type of the person")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 0. Resolve working size from the original person image
    # ------------------------------------------------------------------
    orig_pil = Image.open(args.original).convert("RGBA")
    W, H = orig_pil.size
    sz = (W, H)
    print(f"[INFO] Working resolution: {W}x{H}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load all inputs at working resolution
    # ------------------------------------------------------------------
    person_rgba = load_rgba(args.original, sz)          # (H,W,4) — rembg RGBA
    person_rgb  = person_rgba[:, :, :3]                  # (H,W,3)
    person_alpha = person_rgba[:, :, 3:4]                # (H,W,1) — person silhouette

    warped_cloth = load_rgb(args.warped_cloth, sz)       # (H,W,3)
    warped_mask  = load_mask(args.warped_mask, sz)       # (H,W)   binary

    # SCHP 
    if args.parse and os.path.exists(args.parse):
        parse = load_parse(args.parse, sz)               # (H,W)   uint8 label ids
        has_parse = True
        print("[INFO] SCHP parsing loaded.")
    else:
        parse = None
        has_parse = False
        print("[WARN] No SCHP parse provided.")

    # DensePose 
    if args.densepose and os.path.exists(args.densepose):
        dp_img = cv2.imread(args.densepose)
        dp = cv2.resize(dp_img[:, :, 0], sz, interpolation=cv2.INTER_NEAREST)
        has_dp = True
        print("[INFO] DensePose loaded for hand protection.")
    else:
        dp = None
        has_dp = False

    # ------------------------------------------------------------------
    # 2. Build semantic masks from SCHP & DensePose
    # ------------------------------------------------------------------
    if has_parse:
        old_garment_mask = build_label_mask(parse, GARMENT_LABELS)   # torso clothing
        arm_mask         = build_label_mask(parse, ARM_LABELS)        # both arms
        preserve_mask    = build_label_mask(parse, PRESERVE_LABELS)   # face/hair — NEVER touch
        lower_mask       = build_label_mask(parse, LOWER_LABELS)      # pants/shoes — NEVER touch
    else:
        # Fallback: use agnostic mask as erase region, no arm/preserve logic
        old_garment_mask = load_mask(args.agnostic_mask, sz)
        arm_mask         = np.zeros((H, W), dtype=np.float32)
        preserve_mask    = np.zeros((H, W), dtype=np.float32)
        lower_mask       = np.zeros((H, W), dtype=np.float32)

    # Hand Protection (Labels 3, 4 from DensePose)
    hand_mask = np.zeros((H, W), dtype=np.float32)
    if has_dp:
        hand_mask = ((dp == 3) | (dp == 4)).astype(np.float32)
        # Dilate hands slightly to be safe
        hand_mask = cv2.dilate(hand_mask, np.ones((5, 5), np.uint8), iterations=1)

    # ------------------------------------------------------------------
    # 3. Build the ERASE mask
    #    = old garment area  +  arm pixels that the warped garment covers
    #    Never erase: face/hair, lower body
    # ------------------------------------------------------------------
    # Dilate old garment mask slightly to catch stray fringe pixels
    if args.garment_dilation > 0:
        k = args.garment_dilation | 1
        dil_kernel = np.ones((k, k), np.uint8)
        old_garment_dilated = cv2.dilate(old_garment_mask.astype(np.uint8), dil_kernel, iterations=1).astype(np.float32)
    else:
        old_garment_dilated = old_garment_mask

    # Arms are erased only where the new warped garment physically lands on them
    arm_under_garment = arm_mask * warped_mask

    # --- SMART INPAINTING TRIGGER ---
    # We only inpaint if the user requested it AND the person is initially wearing full sleeves.
    # (If they are wearing half sleeves/tank, the arms/hands are already there in the 'original' image)
    do_inpaint = args.inpaint_skin and (args.initial_sleeve == 'full')
    
    # If inpainting is enabled, we MUST erase the entire original garment area 
    # (even if not covered by the new one) to allow skin to be generated.
    if do_inpaint:
        print("[INFO] Inpainting mode: enabling aggressive arm & garment erasure.")
        # Erase old garment AND arms (but PROTECT hands)
        aggressive_arm_erase = arm_mask * (1.0 - hand_mask)
        erase_mask = np.clip(old_garment_dilated + aggressive_arm_erase + arm_under_garment, 0.0, 1.0)
    else:
        # Conservative (Phase 4 legacy): only erase garment where we have cloth support nearby
        print("[INFO] Standard mode: using conservative garment erasure.")
        support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        paste_support = cv2.dilate((warped_mask > 0.08).astype(np.uint8), support_kernel, iterations=1).astype(np.float32)
        erase_mask = np.clip(old_garment_dilated * paste_support + arm_under_garment, 0.0, 1.0)

    # Hard-protect: never erase face/hair/lower body regardless of above
    # Added hands to protection list here
    protection = np.clip(preserve_mask + lower_mask + hand_mask, 0.0, 1.0)
    erase_mask = erase_mask * (1.0 - protection)

    # ------------------------------------------------------------------
    # 4. Build GARMENT paste mask with soft feathered edge
    # ------------------------------------------------------------------
    # Restrict cloth to person silhouette so it doesn't bleed into background
    paste_mask_hard = warped_mask * (person_alpha[:, :, 0] > 0.1).astype(np.float32)
    # Also don't paste over face/hair/lower
    paste_mask_hard = paste_mask_hard * (1.0 - protection)

    paste_mask_soft = soft_mask(paste_mask_hard, blur_radius=args.feather)
    paste_mask_soft = np.clip(paste_mask_soft, 0.0, 1.0)[:, :, None]  # (H,W,1)

    # ------------------------------------------------------------------
    # 5. Layer composition  (all operations in float32 [0,1])
    #
    #   canvas = person_rgb with conservative erase (cloth-supported)
    #          → paste warped_cloth using soft mask
    #          → restore uncovered erase-holes from original (avoid gray blocks)
    #          → copy-back hard-preserve zone from original
    # ------------------------------------------------------------------
    erase_mask_3d = erase_mask[:, :, None]   # (H,W,1)
    paste_alpha = paste_mask_soft[:, :, 0]

    # Step A: erase old garment / covered arms → neutral mid-grey.
    # If inpainting is off (Standard Mode), only erase where there is cloth support nearby (conservative).
    # If inpainting is ON (Full-to-Half), we use the full erase mask to create the skin holes.
    if not do_inpaint:
        support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        paste_support = cv2.dilate((paste_alpha > 0.08).astype(np.uint8), support_kernel, iterations=1).astype(np.float32)
        erase_mask_3d = erase_mask_3d * paste_support[:, :, None]
    else:
        # In inpainting mode, we assume full support within the provided mask
        paste_support = np.ones_like(paste_alpha)

    neutral = np.full_like(person_rgb, 0.5)
    canvas  = person_rgb * (1.0 - erase_mask_3d) + neutral * erase_mask_3d

    # Step B: paste warped garment with feathered soft mask
    canvas = canvas * (1.0 - paste_mask_soft) + warped_cloth * paste_mask_soft

    # Step B.1: Conditional Skin Inpainting (GAN-based)
    # If a region was erased but is NOT covered by cloth (e.g. forearm where old sleeve was),
    # we use the GAN to fill it with skin instead of just restoring the old sleeve.
    uncovered_holes = (erase_mask_3d[:, :, 0] > 0.15) & (paste_alpha < 0.12)
    
    if np.any(uncovered_holes):
        if do_inpaint and args.checkpoint and os.path.exists(args.checkpoint):
            print("[INFO] Inpainting uncovered skin holes using GAN...")
            # 1. Prepare GAN inputs
            
            torch.backends.cudnn.enabled = False
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Agnostic image (person_rgb with erased regions as grey)
            agnostic_t = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0).to(device) * 2.0 - 1.0
            # Warped cloth
            cloth_t = torch.from_numpy(warped_cloth).permute(2,0,1).unsqueeze(0).to(device) * 2.0 - 1.0
            # Warped mask (normalized)
            mask_t = torch.from_numpy(warped_mask).unsqueeze(0).unsqueeze(0).to(device)
            
            gen_input = torch.cat([agnostic_t, cloth_t, mask_t], 1)
            
            # 2. Load and Run Generator
            gen = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d).to(device)
            ckpt = torch.load(args.checkpoint, map_location=device)
            gen.load_state_dict(ckpt['G'] if 'G' in ckpt else ckpt)
            gen.eval()
            
            with torch.no_grad():
                gen_output = gen(gen_input)
                p_rendered, _ = torch.split(gen_output, [3, 1], 1)
                p_rendered = torch.tanh(p_rendered)
            
            # 3. Skin Tone Matching Fallback
            # Sample skin color from face (Label 13) to ensure GAN output matches person
            face_mask = build_label_mask(parse, [13]) if has_parse else np.zeros_like(warped_mask)
            if np.any(face_mask > 0.5):
                face_pixels = person_rgb[face_mask > 0.5]
                median_skin = np.median(face_pixels, axis=0) # [R, G, B]
                print(f"[INFO] Sampled skin tone from face: {median_skin}")
            else:
                median_skin = np.array([0.8, 0.7, 0.6]) # Fallback caucasian-ish
            
            # 4. Blending logic: Only take skin from GAN for the holes
            inpaint_rgb = (p_rendered[0].permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
            
            # Optional: adjust GAN brightness to match face tone better
            inpaint_median = np.median(inpaint_rgb[uncovered_holes], axis=0) if np.any(uncovered_holes) else median_skin
            tone_adjustment = median_skin / (inpaint_median + 1e-6)
            inpaint_rgb = np.clip(inpaint_rgb * tone_adjustment, 0.0, 1.0)

            hole_blend = soft_mask(uncovered_holes.astype(np.float32), blur_radius=9)[:, :, None]
            hole_blend = np.clip(hole_blend, 0.0, 1.0)
            
            # Preserve original image as much as possible, only using GAN for the "missing" skin
            canvas = canvas * (1.0 - hole_blend) + inpaint_rgb * hole_blend
            
            # Debug: Save what the GAN produced
            debug_path = os.path.join(output_dir if output_dir else ".", "dbg_gan_inpaint.png")
            Image.fromarray((inpaint_rgb * 255).astype(np.uint8)).save(debug_path)
        else:
            # Fallback (non-GAN): Restore original person 
            # (Warning: this will restore original sleeves if they were there!)
            print("[WARN] GAN Inpainting skipped. Restoring original image in holes.")
            hole_blend = soft_mask(uncovered_holes.astype(np.float32), blur_radius=9)[:, :, None]
            hole_blend = np.clip(hole_blend, 0.0, 1.0)
            canvas = canvas * (1.0 - hole_blend) + person_rgb * hole_blend

    # Step C: copy-back hard-preserve zone (face, hair, glasses, hat)
    #         feathered slightly so boundary doesn't look hard-cut
    preserve_soft = soft_mask(preserve_mask, blur_radius=args.preserve_feather)[:, :, None]
    canvas = canvas * (1.0 - preserve_soft) + person_rgb * preserve_soft

    # ------------------------------------------------------------------
    # 6. Re-apply person alpha so background stays transparent
    #    (restore_background.py in phase 5 will composite onto original bg)
    # ------------------------------------------------------------------
    canvas_uint8 = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    alpha_uint8  = np.clip(person_alpha[:, :, 0] * 255.0, 0, 255).astype(np.uint8)

    result_rgba = np.dstack([canvas_uint8, alpha_uint8])  # (H,W,4)
    result_img  = Image.fromarray(result_rgba, mode="RGBA")

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    result_img.save(args.output_path)
    print(f"[SUCCESS] Layered try-on result saved to: {args.output_path}")

    # ------------------------------------------------------------------
    # 8. Debug / diagnostic saves (same folder, prefixed with dbg_)
    # ------------------------------------------------------------------
    dbg_dir = output_dir if output_dir else "."
    Image.fromarray((erase_mask * 255).astype(np.uint8)).save(
        os.path.join(dbg_dir, "dbg_erase_mask.png"))
    Image.fromarray((paste_mask_soft[:, :, 0] * 255).astype(np.uint8)).save(
        os.path.join(dbg_dir, "dbg_paste_mask.png"))
    if has_parse:
        Image.fromarray((preserve_mask * 255).astype(np.uint8)).save(
            os.path.join(dbg_dir, "dbg_preserve_mask.png"))
        Image.fromarray((arm_under_garment * 255).astype(np.uint8)).save(
            os.path.join(dbg_dir, "dbg_arm_erased.png"))
    Image.fromarray((paste_support * 255).astype(np.uint8)).save(
        os.path.join(dbg_dir, "dbg_paste_support.png"))
    print(f"[DEBUG] Diagnostic masks saved to: {dbg_dir}")


if __name__ == "__main__":
    main()
