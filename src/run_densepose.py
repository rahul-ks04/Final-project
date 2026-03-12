import os
import cv2
import torch
import numpy as np
import argparse
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsFineSegmentationVisualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor

def run_densepose(input_path, output_dir, project_root):
    # Setup paths
    config_file = os.path.join(project_root, "detectron2", "projects", "DensePose", "configs", "densepose_rcnn_R_50_FPN_s1x.yaml")
    model_weights = os.path.join(project_root, "detectron2", "checkpoints", "densepose_r50_fpn.pkl")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.basename(input_path)
    # Ensure output is png
    output_filename = os.path.splitext(filename)[0] + "_densepose.png"
    output_image_path = os.path.join(output_dir, output_filename)

    # Setup configuration
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()

    # Load image
    print(f"Loading image: {input_path}")
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Error: Could not read image at {input_path}")

    # Create predictor
    print("Initializing DensePose model...")
    predictor = DefaultPredictor(cfg)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        outputs = predictor(img)["instances"]

    # Initialize extractor (visualizer is no longer needed for this output type)
    extractor = DensePoseResultExtractor()
    
    # Step 5: Extract raw part indices
    results = extractor(outputs)
    
    # Create raw part index map (same size as input image)
    part_map = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # We need to iterate over instances and their corresponding extractor results
    for i in range(len(outputs)):
        inst_res = results[i]
        # inst_res is usually a list of DensePose results
        if not isinstance(inst_res, list) or len(inst_res) == 0:
            continue
            
        # Get bbox from instances (for coordinate mapping)
        # box is [x1, y1, x2, y2]
        bbox = outputs[i].pred_boxes.tensor[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)
        
        # Take the first sub-result (most common for single person)
        for dp_res in inst_res:
            if dp_res is None or not hasattr(dp_res, 'labels'):
                continue
            
            # labels is the segmentation map [0-24] for the box
            labels = dp_res.labels.cpu().numpy()
            
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            # Interpolate labels to the detected bounding box size
            labels_resized = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Clip boundaries to image size
            im_h, im_w = part_map.shape
            y1_c = max(0, y1)
            y2_c = min(im_h, y2)
            x1_c = max(0, x1)
            x2_c = min(im_w, x2)
            
            # Crop resized labels to match clipped bbox
            crop_y1 = y1_c - y1
            crop_y2 = crop_y1 + (y2_c - y1_c)
            crop_x1 = x1_c - x1
            crop_x2 = crop_x1 + (x2_c - x1_c)
            
            if crop_y2 > crop_y1 and crop_x2 > crop_x1:
                # Use maximum if multiple detections overlap
                part_map[y1_c:y2_c, x1_c:x2_c] = np.maximum(
                    part_map[y1_c:y2_c, x1_c:x2_c], 
                    labels_resized[crop_y1:crop_y2, crop_x1:crop_x2]
                )

    # Save output as grayscale indices (essential for generate_target_mask.py)
    cv2.imwrite(output_image_path, part_map)
    print(f"Raw part indices saved to: {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", required=True)
    args = parser.parse_args()
    run_densepose(args.input, args.output_dir, args.project_root)
