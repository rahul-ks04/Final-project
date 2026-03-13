"""
batch_runner.py
Run Path A (master_pipeline.py) on N random PAIRED samples from VITON-HD test set.

Paired = person XXXXX_00.jpg tries on cloth XXXXX_00.jpg (same ID).
Ground truth = test/image/XXXXX_00.jpg (the person's original photo).

The selected pairs are saved to eval_root/eval_pairs.txt so that
batch_runner_pathb.py can run Path B on the EXACT same pairs.

Usage (first run — samples randomly and saves pairs):
  conda run -n densepose python src/batch_runner.py \
      --dataset_root "D:/Final Project Viton/data" \
      --output_root  d:/VITON/outputs/eval_paired \
      --n_samples    10 --seed 42

Usage (re-run with fixed pairs file — skips random sampling):
  conda run -n densepose python src/batch_runner.py \
      --dataset_root "D:/Final Project Viton/data" \
      --output_root  d:/VITON/outputs/eval_paired \
      --pairs_file   d:/VITON/outputs/eval_paired/eval_pairs.txt
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", default=r"D:/Final Project Viton/data")
    p.add_argument("--output_root",  default="d:/VITON/outputs/eval_paired")
    p.add_argument("--project_root", default="d:/VITON")
    p.add_argument("--n_samples",    type=int, default=10)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--pairs_file",   default="",
                   help="Path to an existing eval_pairs.txt to reuse fixed pairs instead of random sampling")
    p.add_argument("--conda_path",   default=r"C:/Users/hp/anaconda3/condabin/conda.bat")
    p.add_argument("--preprocess_env", default="densepose")
    p.add_argument("--parser_env",     default="densepose")
    p.add_argument("--pam_env",        default="fvnt_env")
    p.add_argument("--fvnt_env",       default="fvnt_env")
    p.add_argument("--compose_env",    default="densepose")
    p.add_argument("--pam_mode",       default="predict")
    p.add_argument("--skip_existing",  action="store_true", help="Skip samples that already have output")
    return p.parse_args()


def pick_paired_samples(dataset_root: Path, n: int, seed: int):
    """
    Paired evaluation: person XXXXX_00.jpg wears cloth XXXXX_00.jpg (same stem).
    Both must exist in test/image and test/cloth.
    """
    image_dir = dataset_root / "test" / "image"
    cloth_dir = dataset_root / "test" / "cloth"

    stems = sorted({
        p.stem for p in image_dir.glob("*.jpg")
        if (cloth_dir / f"{p.stem}.jpg").exists()
    })
    print(f"[batch_runner] {len(stems)} valid paired samples found in test set.")

    random.seed(seed)
    chosen = random.sample(stems, min(n, len(stems)))
    return chosen


def run_one(sample_id: str, dataset_root: Path, output_root: Path, args) -> bool:
    """Run master_pipeline for one sample. Returns True on success."""
    person_path = dataset_root / "test" / "image" / f"{sample_id}.jpg"
    cloth_path  = dataset_root / "test" / "cloth"  / f"{sample_id}.jpg"
    sample_out  = output_root / sample_id

    if args.skip_existing and (sample_out / "final" / "tryon_with_background.png").exists():
        print(f"  [skip] {sample_id} — output already exists")
        return True

    sample_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.conda_path, "run", "--no-capture-output", "-n", args.preprocess_env,
        "python", str(Path(args.project_root) / "src" / "master_pipeline.py"),
        "--person",         str(person_path),
        "--garment",        str(cloth_path),
        "--type",           "flat",
        "--project_root",   args.project_root,
        "--output_root",    str(sample_out),
        "--conda_path",     args.conda_path,
        "--preprocess_env", args.preprocess_env,
        "--parser_env",     args.parser_env,
        "--pam_env",        args.pam_env,
        "--fvnt_env",       args.fvnt_env,
        "--compose_env",    args.compose_env,
        "--pam_mode",       args.pam_mode,
    ]

    print(f"\n[batch_runner] Running sample: {sample_id}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [ERROR] {sample_id} failed (exit {result.returncode})")
        return False
    return True


def load_pairs_file(path: Path):
    """Load stems from an existing eval_pairs.txt (format: 'person.jpg cloth.jpg' per line)."""
    stems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                person_file = line.split()[0]
                stems.append(Path(person_file).stem)
    return stems


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_root  = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.pairs_file and Path(args.pairs_file).exists():
        samples = load_pairs_file(Path(args.pairs_file))
        print(f"[batch_runner] Loaded {len(samples)} pairs from {args.pairs_file}: {samples}")
        # Copy the pairs file to output_root if it's not already there
        pairs_file = output_root / "eval_pairs.txt"
        if not pairs_file.exists():
            shutil.copy2(args.pairs_file, pairs_file)
    else:
        samples = pick_paired_samples(dataset_root, args.n_samples, args.seed)
        print(f"[batch_runner] Selected {len(samples)} samples: {samples}")
        # Save the selected pair list so Path B can reuse them
        pairs_file = output_root / "eval_pairs.txt"
        with open(pairs_file, "w") as f:
            for s in samples:
                f.write(f"{s}.jpg {s}.jpg\n")   # person cloth (same for paired)
        print(f"[batch_runner] Pairs saved to {pairs_file} — pass this to batch_runner_pathb.py")

    # Save GT paths alongside for compute_metrics convenience
    gt_links_dir = output_root / "_gt"
    gt_links_dir.mkdir(exist_ok=True)
    for s in samples:
        src = dataset_root / "test" / "image" / f"{s}.jpg"
        dst = gt_links_dir / f"{s}.jpg"
        if not dst.exists():
            shutil.copy2(src, dst)

    # Also copy parse maps for upper-body masking
    parse_links_dir = output_root / "_parse"
    parse_links_dir.mkdir(exist_ok=True)
    for s in samples:
        src = dataset_root / "test" / "image-parse-v3" / f"{s}.png"
        dst = parse_links_dir / f"{s}.png"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # Run pipeline for each
    passed, failed = [], []
    for sample_id in samples:
        ok = run_one(sample_id, dataset_root, output_root, args)
        (passed if ok else failed).append(sample_id)

    print(f"\n[batch_runner] Done. {len(passed)}/{len(samples)} succeeded.")
    if failed:
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
