import argparse
import os

import numpy as np
from PIL import Image


def parse_to_20ch(parse_path):
    label_map = np.array(Image.open(parse_path)).astype(np.int32)
    h, w = label_map.shape[:2]
    out = np.zeros((20, h, w), dtype=np.float32)
    for c in range(20):
        out[c] = (label_map == c).astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description="Create PAM-like 20ch output directly from parse map (debug fallback)")
    parser.add_argument("--parse", required=True, help="Path to parse label map PNG")
    parser.add_argument("--output_dir", required=True, help="Output directory for PAM-like files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    parse = np.array(Image.open(args.parse)).astype(np.uint8)
    parse_20 = parse_to_20ch(args.parse)

    np.save(os.path.join(args.output_dir, "predicted_parsing_20ch.npy"), parse_20)
    Image.fromarray(parse).save(os.path.join(args.output_dir, "predicted_parsing.png"))

    print("Saved PAM fallback outputs from parse map:")
    print(os.path.join(args.output_dir, "predicted_parsing_20ch.npy"))
    print(os.path.join(args.output_dir, "predicted_parsing.png"))


if __name__ == "__main__":
    main()
