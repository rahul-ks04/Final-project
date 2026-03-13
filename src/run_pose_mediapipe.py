import os
import json
import argparse

import cv2
import numpy as np
import mediapipe as mp

try:
    # Older/common API path.
    MP_POSE = mp.solutions.pose
    MP_HANDS = mp.solutions.hands
except AttributeError:
    # Some builds expose solutions under mediapipe.python.
    from mediapipe.python import solutions as mp_solutions
    MP_POSE = mp_solutions.pose
    MP_HANDS = mp_solutions.hands


def _kp_from_landmark(landmarks, idx, width, height):
    lm = landmarks[idx]
    x = float(lm.x * width)
    y = float(lm.y * height)
    c = float(max(0.0, min(1.0, lm.visibility)))
    return np.array([x, y, c], dtype=np.float32)


def mediapipe_to_openpose18(landmarks, width, height):
    # OpenPose 18 format:
    # 0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist,
    # 5 LShoulder, 6 LElbow, 7 LWrist, 8 RHip, 9 RKnee,
    # 10 RAnkle, 11 LHip, 12 LKnee, 13 LAnkle,
    # 14 REye, 15 LEye, 16 REar, 17 LEar
    k = np.zeros((18, 3), dtype=np.float32)

    # MediaPipe pose landmarks indices (subset)
    # 0 nose, 2 left eye, 5 right eye, 7 left ear, 8 right ear,
    # 11 left shoulder, 12 right shoulder,
    # 13 left elbow, 14 right elbow,
    # 15 left wrist, 16 right wrist,
    # 23 left hip, 24 right hip,
    # 25 left knee, 26 right knee,
    # 27 left ankle, 28 right ankle

    k[0] = _kp_from_landmark(landmarks, 0, width, height)   # Nose

    ls = _kp_from_landmark(landmarks, 11, width, height)
    rs = _kp_from_landmark(landmarks, 12, width, height)
    neck = (ls + rs) / 2.0
    k[1] = neck                                               # Neck

    k[2] = rs                                                 # RShoulder
    k[3] = _kp_from_landmark(landmarks, 14, width, height)   # RElbow
    k[4] = _kp_from_landmark(landmarks, 16, width, height)   # RWrist

    k[5] = ls                                                 # LShoulder
    k[6] = _kp_from_landmark(landmarks, 13, width, height)   # LElbow
    k[7] = _kp_from_landmark(landmarks, 15, width, height)   # LWrist

    k[8] = _kp_from_landmark(landmarks, 24, width, height)   # RHip
    k[9] = _kp_from_landmark(landmarks, 26, width, height)   # RKnee
    k[10] = _kp_from_landmark(landmarks, 28, width, height)  # RAnkle

    k[11] = _kp_from_landmark(landmarks, 23, width, height)  # LHip
    k[12] = _kp_from_landmark(landmarks, 25, width, height)  # LKnee
    k[13] = _kp_from_landmark(landmarks, 27, width, height)  # LAnkle

    k[14] = _kp_from_landmark(landmarks, 5, width, height)   # REye
    k[15] = _kp_from_landmark(landmarks, 2, width, height)   # LEye
    k[16] = _kp_from_landmark(landmarks, 8, width, height)   # REar
    k[17] = _kp_from_landmark(landmarks, 7, width, height)   # LEar

    return k


def run_pose(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
    ) as pose:
        result = pose.process(img_rgb)

    if not result.pose_landmarks:
        raise RuntimeError("No person landmarks detected by MediaPipe Pose.")

    keypoints_18 = mediapipe_to_openpose18(result.pose_landmarks.landmark, w, h)

    output = {
        "people": [
            {
                "pose_keypoints": keypoints_18.reshape(-1).tolist()
            }
        ]
    }

    # Build a hand/finger mask using MediaPipe Hands so composition can protect fingers.
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    hands_payload = []
    with MP_HANDS.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    ) as hands:
        hand_result = hands.process(img_rgb)

    if hand_result.multi_hand_landmarks:
        for hand_lms in hand_result.multi_hand_landmarks:
            pts = []
            one_hand = []
            for lm in hand_lms.landmark:
                x = int(np.clip(lm.x * w, 0, w - 1))
                y = int(np.clip(lm.y * h, 0, h - 1))
                pts.append([x, y])
                one_hand.append([float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0))])
            hands_payload.append(one_hand)

            pts_np = np.array(pts, dtype=np.int32)
            if len(pts_np) >= 3:
                hull = cv2.convexHull(pts_np)
                cv2.fillConvexPoly(hand_mask, hull, 255)

            # Strengthen fingertips and palm center coverage.
            for tip_idx in [4, 8, 12, 16, 20, 0, 9]:
                tx, ty = pts[tip_idx]
                cv2.circle(hand_mask, (tx, ty), 10, 255, -1)

    hand_mask = cv2.dilate(hand_mask, np.ones((7, 7), np.uint8), iterations=1)

    base = os.path.splitext(os.path.basename(input_path))[0]
    out_json = os.path.join(output_dir, f"{base}_keypoints.json")
    with open(out_json, "w") as f:
        json.dump(output, f)

    hand_json = os.path.join(output_dir, f"{base}_hands.json")
    with open(hand_json, "w") as f:
        json.dump({"hands": hands_payload}, f)

    hand_mask_path = os.path.join(output_dir, f"{base}_hands_mask.png")
    cv2.imwrite(hand_mask_path, hand_mask)

    print(f"Saved pose JSON: {out_json}")
    print(f"Saved hand mask: {hand_mask_path}")
    print(f"Average confidence: {float(np.mean(keypoints_18[:, 2])):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe -> OpenPose18 pose JSON")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output_dir", required=True, help="Output directory for keypoints JSON")
    args = parser.parse_args()

    run_pose(args.input, args.output_dir)
