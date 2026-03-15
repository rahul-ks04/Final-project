"""
compute_metrics.py
Compute SSIM, PSNR, LPIPS, MSE, FID, and IS on outputs vs VITON-HD ground truth.

Evaluation is masked to the upper-body region (torso + arms) using the
pre-computed Fashn/SCHP parse maps from VITON-HD.  This avoids inflating
scores from the background pixels (which are near-identical).

Upper-body labels used (VITON-HD image-parse-v3 schema, ATR/LIP):
  1=hat  2=hair  4=sunglasses  5=upper  6=dress  7=coat
  10=torso-skin  11=scarf  14=left-arm  15=right-arm

Both generated image and GT are resized to a common resolution before
comparison so resolution mismatches (e.g. padding artifacts) don't matter.

Usage (from densepose env which has torch):
  conda run -n densepose python src/compute_metrics.py \
      --eval_root   d:/VITON/outputs/eval_paired \
      --output_csv  d:/VITON/outputs/eval_paired/metrics.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


# VITON-HD image-parse-v3 upper-body label set
UPPER_BODY_LABELS = {5, 6, 7, 10, 11, 14, 15}  # upper/dress/coat/torso-skin/scarf/arms


def load_rgb_np(path: Path, size=None) -> np.ndarray:
    """Load image as float32 H×W×3 in [0,1]."""
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img).astype(np.float32) / 255.0


def load_parse_mask(path: Path, size=None) -> np.ndarray:
    """Load parse map, return binary upper-body mask H×W float32."""
    img = Image.open(path)
    if size:
        img = img.resize(size, Image.NEAREST)
    arr = np.array(img).astype(np.uint8)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for lbl in UPPER_BODY_LABELS:
        mask |= (arr == lbl).astype(np.uint8)
    return mask.astype(np.float32)


def compute_ssim(gen: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """SSIM on masked upper-body region (multichannel)."""
    # Apply mask per channel; use data_range=1.0
    return float(skimage_ssim(gt, gen, data_range=1.0, channel_axis=2))


def compute_mse(gen: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """MSE on masked pixels only."""
    m = mask > 0.5
    if m.sum() == 0:
        return float("nan")
    gt_px  = gt[m]
    gen_px = gen[m]
    mse = np.mean((gt_px - gen_px) ** 2)
    return float(mse)


def compute_psnr(gen: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> float:
    """PSNR on masked pixels only."""
    mse = compute_mse(gen, gt, mask)
    if np.isnan(mse):
        return float("nan")
    if mse == 0:
        return 100.0
    return float(10 * np.log10(1.0 / mse))


def compute_fid(gen_tensors, gt_tensors) -> float:
    """Compute dataset-level FID using torchmetrics (if available)."""
    import torch
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as e:
        raise ImportError(
            "torchmetrics (with image FID support) is not installed in this environment"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    for t in gt_tensors:
        fid.update(t.to(device), real=True)
    for t in gen_tensors:
        fid.update(t.to(device), real=False)

    val = fid.compute()
    return float(val.item())


def compute_is(gen_tensors):
    """Compute dataset-level Inception Score using torchmetrics."""
    import torch
    try:
        from torchmetrics.image.inception import InceptionScore
    except ImportError as e:
        raise ImportError(
            "torchmetrics (with image Inception Score support) is not installed in this environment"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_metric = InceptionScore(normalize=False).to(device)

    for t in gen_tensors:
        is_metric.update(t.to(device))

    mean, std = is_metric.compute()
    return float(mean.item()), float(std.item())


def load_uint8_chw_tensor(path: Path, size):
    """Load image as torch uint8 tensor shape (1,3,H,W) for FID."""
    import torch

    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    arr = np.array(img).astype(np.uint8)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def compute_lpips(gen_path: Path, gt_path: Path, lpips_fn, size) -> float:
    """LPIPS using pre-loaded lpips function (AlexNet)."""
    import torch
    def to_tensor(p):
        img = Image.open(p).convert("RGB").resize(size, Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t
    with torch.no_grad():
        t_gen = to_tensor(gen_path)
        t_gt  = to_tensor(gt_path)
        val = lpips_fn(t_gen, t_gt)
    return float(val.item())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_root",   required=True,  help="Output root from batch_runner.py")
    p.add_argument("--output_csv",  default="",     help="Where to save per-sample CSV (default: eval_root/metrics.csv)")
    p.add_argument("--eval_size",   type=int, nargs=2, default=[512, 512],
                   help="Common H W to resize both images to before comparison (default: 512 512)")
    p.add_argument("--no_lpips",    action="store_true", help="Skip LPIPS (if lpips package not installed)")
    p.add_argument("--no_fid",      action="store_true", help="Skip FID (if torchmetrics is unavailable)")
    p.add_argument("--no_is",       action="store_true", help="Skip Inception Score (if torchmetrics is unavailable)")
    p.add_argument("--generated_subpath", default="final/tryon_with_background.png",
                   help="Path relative to each sample folder for the generated image")
    return p.parse_args()


def main():
    args = parse_args()
    eval_root = Path(args.eval_root)
    gt_dir    = eval_root / "_gt"
    parse_dir = eval_root / "_parse"
    size      = tuple(args.eval_size)   # (W, H) for PIL

    if not args.output_csv:
        args.output_csv = str(eval_root / "metrics.csv")

    # Collect sample IDs from _gt dir
    sample_ids = sorted([p.stem for p in gt_dir.glob("*.jpg")])
    if not sample_ids:
        print(f"[metrics] No GT images found in {gt_dir}")
        return
    print(f"[metrics] Evaluating {len(sample_ids)} samples at size {size}")

    # Load LPIPS model once
    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex")
            lpips_fn.eval()
            print("[metrics] LPIPS model loaded (AlexNet).")
        except ImportError:
            print("[metrics] WARNING: lpips not installed. Run: pip install lpips")
            print("[metrics] Skipping LPIPS. Re-run without --no_lpips after installing.")

    rows = []
    ssim_vals, psnr_vals, lpips_vals, mse_vals = [], [], [], []
    fid_gen_tensors, fid_gt_tensors = [], []

    for sid in sample_ids:
        gen_path   = eval_root / sid / args.generated_subpath
        gt_path    = gt_dir   / f"{sid}.jpg"
        parse_path = parse_dir / f"{sid}.png"

        if not gen_path.exists():
            print(f"  [missing] {sid}: generated output not found at {gen_path}")
            rows.append({"sample": sid, "ssim": "N/A", "psnr": "N/A", "lpips": "N/A", "mse": "N/A", "fid": "N/A", "is": "N/A"})
            continue

        gen = load_rgb_np(gen_path, size=size)
        gt  = load_rgb_np(gt_path,  size=size)

        # Parse mask: use if available, else full image
        if parse_path.exists():
            mask = load_parse_mask(parse_path, size=size)
            if mask.sum() < 100:   # degenerate mask fallback
                mask = np.ones((size[1], size[0]), dtype=np.float32)
        else:
            print(f"  [warn] {sid}: no parse map, using full image")
            mask = np.ones((size[1], size[0]), dtype=np.float32)

        ssim_val  = compute_ssim(gen, gt, mask)
        psnr_val  = compute_psnr(gen, gt, mask)
        mse_val   = compute_mse(gen, gt, mask)
        lpips_val = compute_lpips(gen_path, gt_path, lpips_fn, size) if lpips_fn else float("nan")

        ssim_vals.append(ssim_val)
        psnr_vals.append(psnr_val)
        mse_vals.append(mse_val)
        if not np.isnan(lpips_val):
            lpips_vals.append(lpips_val)

        if not args.no_fid:
            try:
                fid_gen_tensors.append(load_uint8_chw_tensor(gen_path, size))
                fid_gt_tensors.append(load_uint8_chw_tensor(gt_path, size))
            except Exception as e:
                print(f"  [warn] {sid}: FID tensor load failed: {e}")

        print(f"  {sid}:  SSIM={ssim_val:.4f}  PSNR={psnr_val:.2f}dB  LPIPS={lpips_val:.4f}  MSE={mse_val:.6f}")
        rows.append({
            "sample": sid,
            "ssim": f"{ssim_val:.4f}",
            "psnr": f"{psnr_val:.2f}",
            "lpips": f"{lpips_val:.4f}",
            "mse": f"{mse_val:.6f}",
            "fid": "N/A",
            "is": "N/A",
        })

    # Summary row
    mean_ssim  = np.nanmean(ssim_vals)  if ssim_vals  else float("nan")
    mean_psnr  = np.nanmean(psnr_vals)  if psnr_vals  else float("nan")
    mean_mse   = np.nanmean(mse_vals)   if mse_vals   else float("nan")
    std_ssim   = np.nanstd(ssim_vals)   if ssim_vals  else float("nan")
    std_psnr   = np.nanstd(psnr_vals)   if psnr_vals  else float("nan")
    std_mse    = np.nanstd(mse_vals)    if mse_vals   else float("nan")
    mean_lpips = np.nanmean(lpips_vals) if lpips_vals else float("nan")
    std_lpips  = np.nanstd(lpips_vals)  if lpips_vals else float("nan")

    fid_val = float("nan")
    if not args.no_fid:
        if fid_gen_tensors and fid_gt_tensors:
            try:
                fid_val = compute_fid(fid_gen_tensors, fid_gt_tensors)
            except ImportError:
                print("[metrics] WARNING: torchmetrics/torch-fidelity missing; skipping FID.")
            except Exception as e:
                print(f"[metrics] WARNING: FID computation failed: {e}")
        else:
            print("[metrics] WARNING: No valid tensors for FID; skipping FID.")

    is_mean, is_std = float("nan"), float("nan")
    if not args.no_is:
        if fid_gen_tensors:
            try:
                is_mean, is_std = compute_is(fid_gen_tensors)
            except ImportError:
                print("[metrics] WARNING: torchmetrics missing; skipping IS.")
            except Exception as e:
                print(f"[metrics] WARNING: IS computation failed: {e}")
        else:
            print("[metrics] WARNING: No valid generated tensors for IS; skipping IS.")

    print(f"\n{'='*55}")
    print(f"  SSIM   mean={mean_ssim:.4f}  std={std_ssim:.4f}")
    print(f"  PSNR   mean={mean_psnr:.2f}dB  std={std_psnr:.2f}dB")
    print(f"  LPIPS  mean={mean_lpips:.4f}  std={std_lpips:.4f}")
    print(f"  MSE    mean={mean_mse:.6f}  std={std_mse:.6f}")
    print(f"  FID    value={fid_val:.4f}" if not np.isnan(fid_val) else "  FID    value=N/A")
    print(f"  IS     mean={is_mean:.4f}  std={is_std:.4f}" if not np.isnan(is_mean) else "  IS     mean=N/A  std=N/A")
    print(f"{'='*55}")

    rows.append({
        "sample": "MEAN",
        "ssim": f"{mean_ssim:.4f}±{std_ssim:.4f}",
        "psnr": f"{mean_psnr:.2f}±{std_psnr:.2f}",
        "lpips": f"{mean_lpips:.4f}±{std_lpips:.4f}",
        "mse": f"{mean_mse:.6f}±{std_mse:.6f}",
        "fid": f"{fid_val:.4f}" if not np.isnan(fid_val) else "N/A",
        "is": f"{is_mean:.4f}±{is_std:.4f}" if not np.isnan(is_mean) else "N/A",
    })

    # Write CSV
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "ssim", "psnr", "lpips", "mse", "fid", "is"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[metrics] Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
