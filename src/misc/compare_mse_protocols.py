"""
compare_mse_protocols.py

Compute MSE side-by-side under multiple evaluation protocols so values can be
compared fairly against papers that may use different conventions.

Protocols reported:
  1) masked_01: upper-body masked MSE on [0, 1]
  2) full_01:   full-image MSE on [0, 1]
  3) masked_255: upper-body masked MSE on [0, 255]
  4) full_255:   full-image MSE on [0, 255]

Usage:
  conda run -n densepose python src/compare_mse_protocols.py \
      --eval_root d:/VITON/outputs/eval_paired_30 \
      --output_csv d:/VITON/outputs/eval_paired_30/mse_protocols.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


UPPER_BODY_LABELS = {5, 6, 7, 10, 11, 14, 15}


def load_rgb_01(path: Path, size) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_parse_mask(path: Path, size) -> np.ndarray:
    parse_img = Image.open(path).resize(size, Image.NEAREST)
    arr = np.asarray(parse_img, dtype=np.uint8)
    mask = np.zeros(arr.shape[:2], dtype=np.uint8)
    for lbl in UPPER_BODY_LABELS:
        mask |= (arr == lbl).astype(np.uint8)
    return mask.astype(np.float32)


def mse_full(gen_01: np.ndarray, gt_01: np.ndarray) -> float:
    return float(np.mean((gen_01 - gt_01) ** 2))


def mse_masked(gen_01: np.ndarray, gt_01: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0.5
    if m.sum() == 0:
        return float("nan")
    return float(np.mean((gen_01[m] - gt_01[m]) ** 2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True, help="Batch eval output root")
    parser.add_argument("--output_csv", default="", help="Output CSV path")
    parser.add_argument("--eval_size", type=int, nargs=2, default=[512, 512], help="H W")
    parser.add_argument(
        "--generated_subpath",
        default="final/tryon_with_background.png",
        help="Generated image path relative to sample folder",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_root = Path(args.eval_root)
    gt_dir = eval_root / "_gt"
    parse_dir = eval_root / "_parse"
    size = (args.eval_size[1], args.eval_size[0])  # PIL uses (W, H)

    if not args.output_csv:
        args.output_csv = str(eval_root / "mse_protocols.csv")

    sample_ids = sorted([p.stem for p in gt_dir.glob("*.jpg")])
    if not sample_ids:
        print(f"[mse-protocols] No GT images found in {gt_dir}")
        return

    rows = []
    m_masked_01 = []
    m_full_01 = []

    print(f"[mse-protocols] Evaluating {len(sample_ids)} samples at size {size}")
    for sid in sample_ids:
        gen_path = eval_root / sid / args.generated_subpath
        gt_path = gt_dir / f"{sid}.jpg"
        parse_path = parse_dir / f"{sid}.png"

        if not gen_path.exists():
            print(f"  [missing] {sid}: generated image not found")
            rows.append(
                {
                    "sample": sid,
                    "mse_masked_01": "N/A",
                    "mse_full_01": "N/A",
                    "mse_masked_255": "N/A",
                    "mse_full_255": "N/A",
                }
            )
            continue

        gen = load_rgb_01(gen_path, size)
        gt = load_rgb_01(gt_path, size)

        if parse_path.exists():
            mask = load_parse_mask(parse_path, size)
            if mask.sum() < 100:
                mask = np.ones((size[1], size[0]), dtype=np.float32)
        else:
            mask = np.ones((size[1], size[0]), dtype=np.float32)

        v_masked_01 = mse_masked(gen, gt, mask)
        v_full_01 = mse_full(gen, gt)

        v_masked_255 = v_masked_01 * (255.0 ** 2) if not np.isnan(v_masked_01) else float("nan")
        v_full_255 = v_full_01 * (255.0 ** 2)

        m_masked_01.append(v_masked_01)
        m_full_01.append(v_full_01)

        rows.append(
            {
                "sample": sid,
                "mse_masked_01": f"{v_masked_01:.6f}",
                "mse_full_01": f"{v_full_01:.6f}",
                "mse_masked_255": f"{v_masked_255:.2f}",
                "mse_full_255": f"{v_full_255:.2f}",
            }
        )

    mean_masked_01 = float(np.nanmean(m_masked_01)) if m_masked_01 else float("nan")
    mean_full_01 = float(np.nanmean(m_full_01)) if m_full_01 else float("nan")
    std_masked_01 = float(np.nanstd(m_masked_01)) if m_masked_01 else float("nan")
    std_full_01 = float(np.nanstd(m_full_01)) if m_full_01 else float("nan")

    mean_masked_255 = mean_masked_01 * (255.0 ** 2) if not np.isnan(mean_masked_01) else float("nan")
    mean_full_255 = mean_full_01 * (255.0 ** 2) if not np.isnan(mean_full_01) else float("nan")
    std_masked_255 = std_masked_01 * (255.0 ** 2) if not np.isnan(std_masked_01) else float("nan")
    std_full_255 = std_full_01 * (255.0 ** 2) if not np.isnan(std_full_01) else float("nan")

    rows.append(
        {
            "sample": "MEAN±STD",
            "mse_masked_01": f"{mean_masked_01:.6f}±{std_masked_01:.6f}",
            "mse_full_01": f"{mean_full_01:.6f}±{std_full_01:.6f}",
            "mse_masked_255": f"{mean_masked_255:.2f}±{std_masked_255:.2f}",
            "mse_full_255": f"{mean_full_255:.2f}±{std_full_255:.2f}",
        }
    )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample",
                "mse_masked_01",
                "mse_full_01",
                "mse_masked_255",
                "mse_full_255",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n[mse-protocols] Summary")
    print(f"  masked_01   : {mean_masked_01:.6f} ± {std_masked_01:.6f}")
    print(f"  full_01     : {mean_full_01:.6f} ± {std_full_01:.6f}")
    print(f"  masked_255  : {mean_masked_255:.2f} ± {std_masked_255:.2f}")
    print(f"  full_255    : {mean_full_255:.2f} ± {std_full_255:.2f}")
    print(f"[mse-protocols] Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
