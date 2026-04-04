import os
import cv2
import numpy as np
import argparse
import torch
from PIL import Image


def _keep_largest_component(mask_u8):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_label = int(np.argmax(areas)) + 1
    out = np.zeros_like(mask_u8)
    out[labels == keep_label] = 255
    return out


def flat_mask_from_background_color(img_bgr):
    """Segment flat garment by modeling background color from image borders."""
    h, w = img_bgr.shape[:2]
    border = max(8, int(0.03 * min(h, w)))

    # Collect border pixels to estimate background color (catalog images: uniform backdrop).
    top = img_bgr[:border, :, :].reshape(-1, 3)
    bottom = img_bgr[h - border:, :, :].reshape(-1, 3)
    left = img_bgr[:, :border, :].reshape(-1, 3)
    right = img_bgr[:, w - border:, :].reshape(-1, 3)
    border_px = np.vstack([top, bottom, left, right]).astype(np.uint8)

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    border_lab = cv2.cvtColor(border_px.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    bg_med = np.median(border_lab, axis=0)

    # Distance in Lab space relative to border-estimated background color.
    diff = img_lab.astype(np.float32) - bg_med[None, None, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    # Build background-color candidate map from border stats.
    border_dist = np.sqrt(np.sum((border_lab - bg_med[None, :]) ** 2, axis=1))
    t_bg = float(np.percentile(border_dist, 97) + 6.0)
    t_bg = max(8.0, min(t_bg, 24.0))
    bg_candidate = (dist <= t_bg).astype(np.uint8)

    # Keep only background regions connected to image borders. This prevents
    # interior garment regions (e.g., white logos/stripes) from being dropped
    # even when their color is close to the backdrop.
    num_labels, labels = cv2.connectedComponents(bg_candidate, connectivity=8)
    border_labels = set()
    border_labels.update(np.unique(labels[0, :]).tolist())
    border_labels.update(np.unique(labels[h - 1, :]).tolist())
    border_labels.update(np.unique(labels[:, 0]).tolist())
    border_labels.update(np.unique(labels[:, w - 1]).tolist())

    bg_mask = np.isin(labels, list(border_labels)).astype(np.uint8)
    mask = ((1 - bg_mask) * 255).astype(np.uint8)

    # Clean edges and keep garment body.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = _keep_largest_component(mask)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return mask

def preprocess_garment(garment_path, garment_type, output_dir, parse_mask_path=None, flat_mask_method="color"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load garment
    img = cv2.imread(garment_path)
    if img is None:
        print(f"Error: Could not read garment image at {garment_path}")
        return

    # Resize to standard size (e.g., 768x1024 for VITON-HD)
    img = cv2.resize(img, (768, 1024))
    
    # 1. Generate Garment Mask
    if garment_type == "flat":
        # rembg path disabled per current pipeline decision.
        # if flat_mask_method == "rembg":
        #     print("Using rembg for flat garment masking...")
        #     from rembg import remove
        #
        #     pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #     rembg_img = remove(pil_img)
        #     mask = np.array(rembg_img)[:, :, 3]
        #     _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        # else:
        print("Using border-color segmentation for flat garment masking...")
        mask = flat_mask_from_background_color(img)
    else:
        # For worn garments, we use the parsing mask from the parser stage.
        if parse_mask_path and os.path.exists(parse_mask_path):
            # Keep raw label indices by loading via PIL (not OpenCV grayscale).
            parse_img = Image.open(parse_mask_path)
            parse_mask = np.array(parse_img)
            parse_mask = cv2.resize(parse_mask, (768, 1024), interpolation=cv2.INTER_NEAREST)
            # LIP/SCHP labels: 5 = Upper-clothes, 6 = Dress, 7 = Coat
            mask = np.where(np.isin(parse_mask, [5, 6, 7]), 255, 0).astype(np.uint8)
        else:
            print("Warning: Worn garment type selected but no parse mask provided. Using threshold fallback.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    # 2. Save Results
    cv2.imwrite(os.path.join(output_dir, "cloth.png"), img)
    cv2.imwrite(os.path.join(output_dir, "cloth_mask.png"), mask)
    
    # 3. Generate source_parsing.pt (Tensor expected by some flow renderers)
    # This is a dummy/simplified version - actual requirements vary by model
    # We'll save the mask as a long tensor
    mask_tensor = torch.from_numpy(mask).long()
    torch.save(mask_tensor, os.path.join(output_dir, "source_parsing.pt"))
    
    print(f"Garment preprocessing complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["flat", "worn"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--parse_mask", help="Path to parser output mask (required for worn type)")
    parser.add_argument("--schp_mask", help="Deprecated alias for --parse_mask")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--flat_mask_method", choices=["color", "rembg"], default="color")
    
    args = parser.parse_args()
    mask_path = args.parse_mask if args.parse_mask else args.schp_mask
    preprocess_garment(args.input, args.type, args.output_dir, mask_path, flat_mask_method=args.flat_mask_method)
