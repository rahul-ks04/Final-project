import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import sys

# Add project root to path to import mine
sys.path.append("d:/VITON/FVNT")
from mine.network_stage_1_mine_final_viton import define_G

class PAMRunner:
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"PAM using device: {self.device}")
        
        # PAM Model: input_nc=42, output_nc=20, ngf=64, netG='unet_128'
        self.model = define_G(42, 20, 64, 'unet_128', norm='batch', use_dropout=False).to(self.device)
        
        # Load Checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'G' in ckpt:
            self.model.load_state_dict(ckpt['G'])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        
        self.transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.height = 256
        self.width = 192
        self.radius = 4

    def generate_pose_maps(self, json_path, original_image_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            # Support both official OP-18 and Detectron2 JSON formats
            if 'people' in data and len(data['people']) > 0:
                pose_data = data['people'][0]['pose_keypoints']
            else:
                pose_data = data.get('pose_keypoints', [])
            keypoints = np.array(pose_data).reshape((-1, 3))
        
        orig_img = Image.open(original_image_path)
        orig_w, orig_h = orig_img.size
        # Scale factor (192x256)
        scale_x = self.width / orig_w
        scale_y = self.height / orig_h
        
        pose_maps = torch.zeros((18, self.height, self.width))
        for i in range(min(18, len(keypoints))):
            # Dataset1 uses RGB 'white' followed by transform()[0]
            one_map = Image.new('RGB', (self.width, self.height))
            draw = ImageDraw.Draw(one_map)
            px, py = keypoints[i][0] * scale_x, keypoints[i][1] * scale_y
            
            # dataset1.py line 110: if pointX > 1 or pointY > 1:
            if px > 1 or py > 1:
                draw.ellipse((px - self.radius, py - self.radius, px + self.radius, py + self.radius), 
                             fill='white', outline='white')
            
            # transform converts RGB (255,255,255) to 1.0 in all channels
            pose_maps[i] = transforms.ToTensor()(one_map)[0]
        return pose_maps

    def predict(self, cloth_img_path, cloth_mask_path, person_parse_path, keypoints_json_path, original_image_path):
        # 1. Cloth RGB (3 ch) - MUST have 0.0 background (dataset1 line 72)
        c_rgb_pil = Image.open(cloth_img_path).convert("RGB").resize((self.width, self.height))
        c_mask_pil = Image.open(cloth_mask_path).convert("L").resize((self.width, self.height))
        c_mask_torch = transforms.ToTensor()(c_mask_pil) # [1, H, W], range [0, 1]
        
        c_rgb = self.transform_norm(c_rgb_pil) # [3, H, W], range [-1, 1]
        c_rgb = c_rgb * c_mask_torch # Background becomes EXACTLY 0.0
        
        # 2. Source Cloth Parsing (20 ch) - TRAINING EXACT MATCH
        # The model was trained to receive the SOURCE cloth mask ONLY in categories
        # where the TARGET person has CLOTHING (from cloth_list).
        # This is the critical fix - do NOT place cloth in all categories.
        
        source_cloth_parsing = torch.zeros((20, self.height, self.width))
        
        p_parse_pil = Image.open(person_parse_path).resize((self.width, self.height), Image.NEAREST)
        p_parse_np = np.array(p_parse_pil)
        
        # Target parsing mask (which clothing categories exist?)
        target_parsing_cloth_20 = torch.zeros((20, self.height, self.width))
        cloth_list = [-1, 1, -1, -1, -1, 5, 6, 7, 8, 9, -1, -1, 12, -1, -1, -1, -1, -1, -1, -1]
        
        for i in range(20):
            if cloth_list[i] >= 0:
                # This category CAN have clothing
                mask = (p_parse_np == cloth_list[i]).astype(np.float32)
                target_parsing_cloth_20[i] += torch.from_numpy(mask)
        
        # ONLY place source cloth mask in categories where target has clothing
        for i in range(20):
            if target_parsing_cloth_20[i].sum() > 0:
                source_cloth_parsing[i] += c_mask_torch[0]

        # 3. Hair-Face-Shoes Binary Mask (1 ch) - (dataset1 line 91)
        # HFS: 2: Hair, 13: Face, 18: L-Shoe, 19: R-Shoe
        hfs = np.isin(p_parse_np, [2, 13, 18, 19]).astype(np.float32)
        hfs_torch = torch.from_numpy(hfs).unsqueeze(0)
        
        # 4. Target Pose (18 ch) - (dataset1 line 102)
        self.radius = 4 # Official training radius
        pose_maps = self.generate_pose_maps(keypoints_json_path, original_image_path)
        
        # 5. Assemble 42 channels in EXACT training order: (RGB, Parse, HFS, Pose)
        input_tensor = torch.cat((c_rgb, source_cloth_parsing, hfs_torch, pose_maps), dim=0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor) # [1, 20, H, W]
            
        return output[0].cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cloth", required=True)
    parser.add_argument("--cloth_mask", required=True)
    parser.add_argument("--parse", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    runner = PAMRunner(args.checkpoint)
    
    pred = runner.predict(args.cloth, args.cloth_mask, args.parse, args.pose, args.image)
    
    # Save the max-probability class map
    final_map = np.argmax(pred, axis=0).astype(np.uint8)
    out_img = Image.fromarray(final_map)
    out_img.save(os.path.join(args.output_dir, "predicted_parsing.png"))
    
    # Save the raw 20-channel data (optional for FEM)
    np.save(os.path.join(args.output_dir, "predicted_parsing_20ch.npy"), pred)
    
    print(f"PAM prediction saved to {args.output_dir}")

if __name__ == "__main__":
    main()
