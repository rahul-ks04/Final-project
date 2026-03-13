import argparse
import os

import cv2
import numpy as np
from PIL import Image


# LIP/CIHP label groups (same schema as Fashn remap in this project)
PRESERVE_LABELS = [1, 2, 4, 13]          # hat, hair, sunglasses, face
GARMENT_LABELS = [5, 6, 7, 10, 11]       # upper/dress/coat/torso-skin/scarf
ARM_LABELS = [14, 15]                    # left/right arm
LOWER_LABELS = [9, 12, 16, 17, 18, 19]   # lower body


def load_rgba(path, target_size=None):
    img = Image.open(path).convert("RGBA")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_rgb(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_parse(path, target_size=None):
    img = Image.open(path)
    if target_size:
        img = img.resize(target_size, Image.NEAREST)
    return np.array(img).astype(np.uint8)


def load_mask(path, target_size=None, threshold=0.5):
    img = Image.open(path).convert("L")
    if target_size:
        img = img.resize(target_size, Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return (arr > threshold).astype(np.float32)


def soft_mask(binary_mask, blur_radius=9):
    r = blur_radius | 1
    return cv2.GaussianBlur(binary_mask.astype(np.float32), (r, r), 0)


def build_label_mask(parse, labels):
    m = np.zeros(parse.shape, dtype=np.uint8)
    for lbl in labels:
        m |= (parse == lbl).astype(np.uint8)
    return m.astype(np.float32)


def derive_warped_mask_from_rgb(warped_cloth_rgb, threshold=0.04):
    # FVNT warped cloth background is near black; threshold non-black support.
    return (np.max(warped_cloth_rgb, axis=2) > threshold).astype(np.float32)


def compute_uncovered_holes(erase_mask, paste_alpha):
    # Holes are regions erased from person but not covered by new garment.
    return (erase_mask > 0.15) & (paste_alpha < 0.12)


def match_skin_tone(inpaint_rgb, person_rgb, parse, uncovered_holes, erase_mask):
    # Prefer visible arm skin reference, then face. Affine per-channel match.
    ref_pixels = None
    arm_visible = build_label_mask(parse, ARM_LABELS) * (1.0 - erase_mask)
    if np.sum(arm_visible > 0.5) > 50:
        ref_pixels = person_rgb[arm_visible > 0.5]

    if ref_pixels is None or len(ref_pixels) < 50:
        face_mask = build_label_mask(parse, [13])
        if np.any(face_mask > 0.5):
            ref_pixels = person_rgb[face_mask > 0.5]

    if ref_pixels is None or not np.any(uncovered_holes):
        return np.clip(inpaint_rgb, 0.0, 1.0)

    ref_mean = np.mean(ref_pixels, axis=0)
    ref_std = np.std(ref_pixels, axis=0) + 1e-6

    hole_pixels = inpaint_rgb[uncovered_holes]
    hole_mean = np.mean(hole_pixels, axis=0)
    hole_std = np.std(hole_pixels, axis=0) + 1e-6

    corrected = (inpaint_rgb - hole_mean) * (ref_std / hole_std) + ref_mean
    return np.clip(corrected, 0.0, 1.0)


def run_composition(
    original_rgba_path,
    parse_path,
    warped_cloth_path,
    output_path,
    inpaint_skin=True,
    erase_arms_under_cloth=False,
    hand_mask_path=None,
    warped_mask_path=None,
    save_debug_masks=True,
):
    orig_rgba = load_rgba(original_rgba_path)
    h, w = orig_rgba.shape[:2]
    sz = (w, h)

    person_rgb = orig_rgba[:, :, :3]
    person_alpha = orig_rgba[:, :, 3:4]

    parse = load_parse(parse_path, target_size=sz)
    warped_cloth = load_rgb(warped_cloth_path, target_size=sz)
    if warped_mask_path and os.path.isfile(warped_mask_path):
        warped_mask = load_mask(warped_mask_path, target_size=sz)
    else:
        warped_mask = derive_warped_mask_from_rgb(warped_cloth)
    hand_mask = np.zeros((h, w), dtype=np.float32)
    if hand_mask_path and os.path.isfile(hand_mask_path):
        hand_mask = load_mask(hand_mask_path, target_size=sz)

    preserve_mask = build_label_mask(parse, PRESERVE_LABELS)
    lower_mask = build_label_mask(parse, LOWER_LABELS)
    arm_mask = build_label_mask(parse, ARM_LABELS)
    old_garment_mask = build_label_mask(parse, GARMENT_LABELS)

    protection = np.clip(preserve_mask + lower_mask + hand_mask, 0.0, 1.0)

    # Erase old garment; optional arm erase where new cloth overlays.
    garment_dil = cv2.dilate(old_garment_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    arm_under_cloth = arm_mask * warped_mask if erase_arms_under_cloth else np.zeros_like(warped_mask)

    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    paste_support = cv2.dilate((warped_mask > 0.08).astype(np.uint8), support_kernel, iterations=1).astype(np.float32)

    erase_mask = np.clip(garment_dil * paste_support + arm_under_cloth, 0.0, 1.0)
    erase_mask = erase_mask * (1.0 - protection)

    # Compose base canvas
    neutral = np.full_like(person_rgb, 0.5)
    erase_3d = erase_mask[:, :, None]
    canvas = person_rgb * (1.0 - erase_3d) + neutral * erase_3d

    # Paste garment
    paste_hard = warped_mask * (person_alpha[:, :, 0] > 0.1).astype(np.float32)
    paste_hard = paste_hard * (1.0 - protection)
    paste_soft = soft_mask(paste_hard, blur_radius=21)
    paste_soft = np.clip(paste_soft, 0.0, 1.0)[:, :, None]
    canvas = canvas * (1.0 - paste_soft) + warped_cloth * paste_soft

    # Compute holes and optionally inpaint with LaMa.
    # Raw holes include any erased-but-uncovered region.
    uncovered_holes_raw = compute_uncovered_holes(erase_mask, paste_soft[:, :, 0])
    # For half->half (or generally visible arms), do NOT inpaint arm/hand regions.
    # Those should be restored from the original person image.
    inpaint_exclusion = ((arm_mask > 0.5) | (hand_mask > 0.5))
    uncovered_holes = uncovered_holes_raw & (~inpaint_exclusion)
    did_inpaint = False
    if np.any(uncovered_holes):
        if inpaint_skin:
            try:
                from simple_lama_inpainting import SimpleLama

                lama = SimpleLama()
                in_pil = Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255).astype(np.uint8))
                hole_mask = Image.fromarray((uncovered_holes.astype(np.uint8) * 255), mode="L")
                out_pil = lama(in_pil, hole_mask)
                inpaint_rgb = np.array(out_pil).astype(np.float32) / 255.0
                inpaint_rgb = match_skin_tone(inpaint_rgb, person_rgb, parse, uncovered_holes, erase_mask)

                hole_blend = soft_mask(uncovered_holes.astype(np.float32), blur_radius=9)[:, :, None]
                hole_blend = np.clip(hole_blend, 0.0, 1.0)
                canvas = canvas * (1.0 - hole_blend) + inpaint_rgb * hole_blend
                did_inpaint = True
            except ImportError:
                print("[WARN] simple-lama-inpainting not installed; restoring original pixels in holes.")
                hole_blend = soft_mask(uncovered_holes.astype(np.float32), blur_radius=9)[:, :, None]
                hole_blend = np.clip(hole_blend, 0.0, 1.0)
                canvas = canvas * (1.0 - hole_blend) + person_rgb * hole_blend
        else:
            hole_blend = soft_mask(uncovered_holes.astype(np.float32), blur_radius=9)[:, :, None]
            hole_blend = np.clip(hole_blend, 0.0, 1.0)
            canvas = canvas * (1.0 - hole_blend) + person_rgb * hole_blend

    # Safety pass: restore original arm pixels where garment coverage is absent.
    # This directly prevents sleeve/background artifacts from bleeding over bare arms.
    cover_hard = (paste_soft[:, :, 0] > 0.35).astype(np.uint8)
    cover_hard = cv2.dilate(cover_hard, np.ones((5, 5), np.uint8), iterations=1).astype(np.float32)
    arm_restore_region = arm_mask * (1.0 - cover_hard)
    arm_restore_region = arm_restore_region * (1.0 - hand_mask)
    arm_restore_soft = soft_mask(arm_restore_region, blur_radius=7)[:, :, None]
    arm_restore_soft = np.clip(arm_restore_soft, 0.0, 1.0)
    canvas = canvas * (1.0 - arm_restore_soft) + person_rgb * arm_restore_soft

    # Restore preservation region last
    preserve_soft = soft_mask(preserve_mask, blur_radius=9)[:, :, None]
    canvas = canvas * (1.0 - preserve_soft) + person_rgb * preserve_soft

    out_rgb = np.clip(canvas * 255.0, 0, 255).astype(np.uint8)
    out_a = np.clip(person_alpha[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    out_rgba = np.dstack([out_rgb, out_a])

    out_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(out_dir, exist_ok=True)

    if save_debug_masks:
        Image.fromarray((warped_mask * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "dbg_warped_mask.png"))
        Image.fromarray((paste_soft[:, :, 0] * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "dbg_paste_mask_soft.png"))
        Image.fromarray((erase_mask * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "dbg_erase_mask.png"))
        Image.fromarray((uncovered_holes_raw.astype(np.uint8) * 255), mode="L").save(os.path.join(out_dir, "dbg_hole_mask_raw.png"))
        Image.fromarray((uncovered_holes.astype(np.uint8) * 255), mode="L").save(os.path.join(out_dir, "dbg_hole_mask_inpaint.png"))
        Image.fromarray((hand_mask * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "dbg_hand_mask.png"))
        Image.fromarray((arm_restore_region * 255.0).astype(np.uint8), mode="L").save(os.path.join(out_dir, "dbg_arm_restore_region.png"))

    print(f"[INFO] Hole pixels raw/inpaint: {int(np.sum(uncovered_holes_raw))}/{int(np.sum(uncovered_holes))} | Inpaint applied: {did_inpaint}")

    Image.fromarray(out_rgba, mode="RGBA").save(output_path)
    print(f"Saved composed try-on: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compose warped garment onto person (with optional LaMa hole inpaint)")
    parser.add_argument("--original", required=True, help="RGBA person image from remove_background.py")
    parser.add_argument("--parse", required=True, help="Fashn parse map path")
    parser.add_argument("--warped_cloth", required=True, help="FVNT warped garment RGB path")
    parser.add_argument("--warped_mask", default=None, help="Optional exact warped garment mask path")
    parser.add_argument("--output_path", required=True, help="Output RGBA path")
    parser.add_argument("--hand_mask", default=None, help="Optional hand/finger mask path to protect fingers")
    parser.add_argument("--no_inpaint_skin", action="store_true", help="Disable LaMa inpainting and restore original pixels in holes")
    parser.add_argument("--erase_arms_under_cloth", action="store_true", help="Aggressively erase arm pixels under warped cloth before compositing")
    parser.add_argument("--no_debug_masks", action="store_true", help="Disable saving debug masks")
    args = parser.parse_args()

    run_composition(
        original_rgba_path=args.original,
        parse_path=args.parse,
        warped_cloth_path=args.warped_cloth,
        output_path=args.output_path,
        inpaint_skin=(not args.no_inpaint_skin),
        erase_arms_under_cloth=args.erase_arms_under_cloth,
        hand_mask_path=args.hand_mask,
        warped_mask_path=args.warped_mask,
        save_debug_masks=(not args.no_debug_masks),
    )
