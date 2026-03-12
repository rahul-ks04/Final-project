import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import argparse
from tqdm import tqdm
from pathlib import Path

# CIHP / LIP 20-class schema (expected by FVNT PAM)
CIHP_LABELS = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "glove": 3,
    "sunglasses": 4,
    "upper_clothes": 5, "top": 5, "upper_body": 5,
    "dress": 6, "one_piece": 6,
    "coat": 7, "outerwear": 7,
    "socks": 8,
    "pants": 9, "bottom": 9, "bottom_clothes": 9, "trousers": 9,
    "torso_skin": 10,
    "scarf": 11,
    "skirt": 12,
    "face": 13,
    "left_arm": 14,
    "right_arm": 15,
    "left_leg": 16,
    "right_leg": 17,
    "left_shoe": 18, "footwear": 18,
    "right_shoe": 19
}

def _normalise(name: str) -> str:
    # Fashn labels use dashes and mixed case
    return name.lower().strip().replace(" ", "_").replace("-", "_")

class FashnParser:
    def __init__(self, model_id="mattmdjaga/segformer_b2_clothes"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Build remapping array
        max_fashn_id = max(int(k) for k in self.model.config.id2label.keys())
        self.remap = np.zeros(max_fashn_id + 1, dtype=np.uint8)
        
        for fashn_id, raw_name in self.model.config.id2label.items():
            norm = _normalise(raw_name)
            viton_id = CIHP_LABELS.get(norm, 0)
            self.remap[int(fashn_id)] = viton_id

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        upsampled = F.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False
        )
        
        pred = upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        viton_parse = self.remap[pred]
        return viton_parse

def run_parser(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    parser = FashnParser()
    
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    img_paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in img_exts
    )
    
    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(img_paths)} images...")
    for img_path in tqdm(img_paths):
        parse_map = parser.predict(str(img_path))
        out_path = os.path.join(output_dir, img_path.stem + ".png")
        Image.fromarray(parse_map).save(out_path)
    
    print(f"Done. Parse maps saved to: {output_dir}")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_dir", required=True)
    arg_parser.add_argument("--output_dir", required=True)
    args = arg_parser.parse_args()
    
    run_parser(args.input_dir, args.output_dir)
