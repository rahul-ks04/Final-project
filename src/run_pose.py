import os
import torch
import numpy as np
import cv2
import json
import argparse
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

def run_pose(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Detectron2 Keypoint Model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    predictor = DefaultPredictor(cfg)
    
    # Load Image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image {input_path}")
        return
    
    outputs = predictor(img)
    
    # Extract Keypoints
    # Detectron2 R-50 FPN output is [17] keypoints (COCO standard)
    # PAM (OpenPose) expects 18. We will map them.
    # COCO: 0: nose, 1: L eye, 2: R eye, 3: L ear, 4: R ear, 5: L shoulder, 6: R shoulder, 
    #       7: L elbow, 8: R elbow, 9: L wrist, 10: R wrist, 11: L hip, 12: R hip, 
    #       13: L knee, 14: R knee, 15: L ankle, 16: R ankle
    # OpenPose 18: 0:Nose, 1:Neck, 2:RShoulder, 3:RElbow, 4:RWrist, 5:LShoulder, 6:LElbow, 7:LWrist,
    #              8:RHip, 9:RKnee, 10:RAnkle, 11:LHip, 12:LKnee, 13:LAnkle, 14:REye, 15:LEye, 16:REar, 17:LEar
    
    if len(outputs["instances"]) == 0:
        print("No person detected.")
        return
    
    # Take the person with the highest score
    inst = outputs["instances"][0]
    keypoints = inst.get("pred_keypoints")[0].cpu().numpy() # [17, 3] (x, y, conf)
    
    # Map to 18 keypoints
    out_keypoints = np.zeros((18, 3))
    
    # 0: Nose
    out_keypoints[0] = keypoints[0]
    # 1: Neck (Average of shoulders)
    out_keypoints[1] = (keypoints[5] + keypoints[6]) / 2
    # 2: RShoulder
    out_keypoints[2] = keypoints[6]
    # 3: RElbow
    out_keypoints[3] = keypoints[8]
    # 4: RWrist
    out_keypoints[4] = keypoints[10]
    # 5: LShoulder
    out_keypoints[5] = keypoints[5]
    # 6: LElbow
    out_keypoints[6] = keypoints[7]
    # 7: LWrist
    out_keypoints[7] = keypoints[9]
    # 8: RHip
    out_keypoints[8] = keypoints[12]
    # 9: RKnee
    out_keypoints[9] = keypoints[14]
    # 10: RAnkle
    out_keypoints[10] = keypoints[16]
    # 11: LHip
    out_keypoints[11] = keypoints[11]
    # 12: LKnee
    out_keypoints[12] = keypoints[13]
    # 13: LAnkle
    out_keypoints[13] = keypoints[15]
    # 14: REye
    out_keypoints[14] = keypoints[2]
    # 15: LEye
    out_keypoints[15] = keypoints[1]
    # 16: REar
    out_keypoints[16] = keypoints[4]
    # 17: LEar
    out_keypoints[17] = keypoints[3]

    # Save format expected by PAM (OpenPose JSON style)
    res = {
        "people": [{
            "pose_keypoints": out_keypoints.flatten().tolist()
        }]
    }
    
    filename = os.path.basename(input_path).replace(".jpg", "_keypoints.json").replace(".png", "_keypoints.json")
    out_path = os.path.join(output_dir, filename)
    
    with open(out_path, "w") as f:
        json.dump(res, f)
    
    print(f"Pose keypoints saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    run_pose(args.input, args.output_dir)
