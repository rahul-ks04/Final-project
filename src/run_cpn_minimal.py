"""
Minimal CPN pose extraction - single image inference
Avoids tfflat complexity, works directly with TF1 checkpoint
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2

# Change to CPN model directory for relative imports
os.chdir("d:/VITON/tf-cpn/models/COCO.res50.384x288.CPN")
sys.path.insert(0, "d:/VITON/tf-cpn/models/COCO.res50.384x288.CPN")
sys.path.insert(0, "d:/VITON/tf-cpn/lib")

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from config import cfg
from network import Network
from dataset import Preprocessing

def coco_to_openpose_18(keypoints_17):
    """Convert 17-point COCO format to 18-point OpenPose format"""
    # COCO keypoints order
    keypoints_18 = np.zeros((18, 3))
    
    # 0: Nose
    keypoints_18[0] = keypoints_17[0]
    # 1: Neck (average of shoulders)
    keypoints_18[1] = (keypoints_17[5] + keypoints_17[6]) / 2
    # 2: RShoulder
    keypoints_18[2] = keypoints_17[6]
    # 3: RElbow
    keypoints_18[3] = keypoints_17[8]
    # 4: RWrist
    keypoints_18[4] = keypoints_17[10]
    # 5: LShoulder
    keypoints_18[5] = keypoints_17[5]
    # 6: LElbow
    keypoints_18[6] = keypoints_17[7]
    # 7: LWrist
    keypoints_18[7] = keypoints_17[9]
    # 8: RHip
    keypoints_18[8] = keypoints_17[12]
    # 9: RKnee
    keypoints_18[9] = keypoints_17[14]
    # 10: RAnkle
    keypoints_18[10] = keypoints_17[16]
    # 11: LHip
    keypoints_18[11] = keypoints_17[11]
    # 12: LKnee
    keypoints_18[12] = keypoints_17[13]
    # 13: LAnkle
    keypoints_18[13] = keypoints_17[15]
    # 14: REye
    keypoints_18[14] = keypoints_17[2]
    # 15: LEye
    keypoints_18[15] = keypoints_17[1]
    # 16: REar
    keypoints_18[16] = keypoints_17[4]
    # 17: LEar
    keypoints_18[17] = keypoints_17[3]
    
    return keypoints_18

def run_cpn_pose(image_path, output_dir, gpu_id='0'):
    """
    Run CPN pose estimation on a single image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    h_orig, w_orig = img_orig.shape[:2]
    print(f"Image size: {w_orig}x{h_orig}")
    
    # Create simple detection box (full image)
    bbox = np.array([0, 0, w_orig, h_orig], dtype=np.float32)
    
    det = {
        'image_id': 0,
        'category_id': 1,
        'bbox': bbox.tolist(),
        'score': 1.0,
        'imgpath': image_path
    }
    
    # Configure
    cfg.set_args(gpu_id)
    
    # Create TensorFlow graph
    graph = tf.Graph()
    with graph.as_default():
        # Build network
        net = Network()
        image_ph = tf.placeholder(tf.float32, shape=[None, cfg.data_shape[0], cfg.data_shape[1], 3])
        outputs = net.output(image_ph, is_train=False)
        
        # Load checkpoint
        checkpoint_path = "log/model_dump/snapshot_350.ckpt"
        saver = tf.train.Saver()
        
        with tf.Session(graph=graph) as sess:
            print(f"Loading checkpoint: {checkpoint_path}")
            saver.restore(sess, checkpoint_path)
            print("✓ Checkpoint loaded")
            
            # Preprocess
            test_img, detail = Preprocessing(det, stage='test')
            feed = [test_img]
            
            # Add flipped version
            ori_img = test_img[0].transpose(1, 2, 0)
            flip_img = cv2.flip(ori_img, 1)
            feed.append(flip_img.transpose(2, 0, 1)[np.newaxis, ...])
            feed = np.vstack(feed)
            
            # Run inference
            print("Running inference...")
            heatmaps = sess.run(outputs, feed_dict={
                image_ph: feed.transpose(0, 2, 3, 1).astype(np.float32)
            })
            
            heatmaps = heatmaps.transpose(0, 3, 1, 2)
            print(f"Heatmap shape: {heatmaps.shape}")
            
            # Average flipped predictions
            fmp = heatmaps[1].transpose((1, 2, 0))
            fmp = cv2.flip(fmp, 1)
            fmp = list(fmp.transpose((2, 0, 1)))
            for (q, w) in cfg.symmetry:
                fmp[q], fmp[w] = fmp[w], fmp[q]
            fmp = np.array(fmp)
            heatmaps[0] += fmp
            heatmaps[0] /= 2
            
            # Extract keypoints from heatmaps
            heatmaps = heatmaps[0]  # Shape: (17, h_out, w_out)
            keypoints_17 = np.zeros((17, 3))
            
            for i in range(17):
                hm = heatmaps[i]
                y, x = np.unravel_index(hm.argmax(), hm.shape)
                
                # Scale back to original image
                x_orig = x * w_orig / cfg.output_shape[1]
                y_orig = y * h_orig / cfg.output_shape[0]
                conf = float(hm[y, x])
                
                keypoints_17[i] = [x_orig, y_orig, conf]
            
            # Convert to OpenPose 18-point
            keypoints_18 = coco_to_openpose_18(keypoints_17)
            
            # Save JSON
            output_json = {
                "people": [{
                    "pose_keypoints": keypoints_18.flatten().tolist()
                }]
            }
            
            filename = os.path.basename(image_path).replace(".jpg", "_keypoints.json").replace(".png", "_keypoints.json")
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, 'w') as f:
                json.dump(output_json, f)
            
            print(f"\n✓ CPN pose extracted!")
            print(f"✓ Saved to {output_path}")
            print(f"✓ Format: 18-point OpenPose")
            print(f"✓ Average confidence: {keypoints_18[:, 2].mean():.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPN Minimal Pose Extraction")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--gpu", default="0", help="GPU ID")
    args = parser.parse_args()
    
    run_cpn_pose(args.image, args.output_dir, args.gpu)
