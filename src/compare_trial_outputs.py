import argparse
import os
from PIL import Image, ImageDraw


def load_rgb(path, size):
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    return img


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side comparison for Path A and Path B outputs")
    parser.add_argument("--path_a", required=True, help="Path A image")
    parser.add_argument("--path_b", required=True, help="Path B image")
    parser.add_argument("--out", required=True, help="Output comparison image path")
    args = parser.parse_args()

    target_size = (384, 512)
    a = load_rgb(args.path_a, target_size)
    b = load_rgb(args.path_b, target_size)

    pad = 24
    label_h = 40
    w = target_size[0] * 2 + pad * 3
    h = target_size[1] + pad * 2 + label_h

    canvas = Image.new("RGB", (w, h), (245, 246, 248))
    draw = ImageDraw.Draw(canvas)

    ax = pad
    bx = pad * 2 + target_size[0]
    y = pad + label_h

    canvas.paste(a, (ax, y))
    canvas.paste(b, (bx, y))

    draw.text((ax, pad), "Path A (Current Custom Composition)", fill=(20, 20, 20))
    draw.text((bx, pad), "Path B (FVNT Native Final Stage)", fill=(20, 20, 20))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"Saved comparison: {args.out}")


if __name__ == "__main__":
    main()
