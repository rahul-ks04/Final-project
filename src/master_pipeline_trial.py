import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image


def run_cmd(python_cmd, script_path, args, cwd=None):
    if isinstance(python_cmd, str):
        python_cmd = [python_cmd]

    cmd = python_cmd + [script_path] + args
    quoted_cmd = [f'"{a}"' if " " in str(a) else str(a) for a in cmd]
    print(f"\n>>> Running: {' '.join(quoted_cmd)}")

    is_windows = os.name == "nt"
    result = subprocess.run(cmd, cwd=cwd, shell=is_windows)
    if result.returncode != 0:
        print(f"!!! Error return code {result.returncode} from command.")
        return False
    return True


def first_existing(candidates, name):
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Missing {name}. Checked: {candidates}")


def find_input(project_root, pattern, label):
    matches = glob.glob(os.path.join(project_root, "inputs", pattern))
    if not matches:
        print(f"!!! Error: Could not find {label} matching '{pattern}' in inputs/ folder.")
        print(f"Please upload a file named '{pattern.replace('*', 'your_ext')}' to {os.path.join(project_root, 'inputs')}")
        sys.exit(1)
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        for m in matches:
            if m.lower().endswith(ext):
                return m
    return matches[0]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_rgb_jpg(src_path, dst_path, size_wh=None):
    img = Image.open(src_path).convert("RGB")
    if size_wh is not None:
        img = img.resize(size_wh, Image.BILINEAR)
    img.save(dst_path, quality=95)


def save_mask_jpg(src_path, dst_path, size_wh=None):
    m = Image.open(src_path).convert("L")
    if size_wh is not None:
        m = m.resize(size_wh, Image.NEAREST)
    arr = (np.array(m) >= 128).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(dst_path, quality=95)


def save_parse_png(src_path, dst_path, size_wh=None):
    p = Image.open(src_path)
    if p.mode != "L":
        p = p.convert("L")
    if size_wh is not None:
        p = p.resize(size_wh, Image.NEAREST)
    p.save(dst_path)


def pam20_to_label_png(pam_npy_path, dst_png_path):
    arr = np.load(pam_npy_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D PAM tensor (20,H,W), got shape {arr.shape}")
    label = np.argmax(arr, axis=0).astype(np.uint8)
    Image.fromarray(label, mode="L").save(dst_png_path)


def scale_pose_json_for_fvnt(src_pose_json, dst_pose_json, src_w, src_h, dst_w=192, dst_h=256):
    with open(src_pose_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "people" not in data or not data["people"]:
        raise ValueError("Pose JSON does not contain people[0].")

    people0 = data["people"][0]
    if "pose_keypoints" not in people0:
        raise ValueError("Pose JSON does not contain people[0].pose_keypoints.")

    k = np.array(people0["pose_keypoints"], dtype=np.float32).reshape(-1, 3)
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    k[:, 0] = k[:, 0] * sx
    k[:, 1] = k[:, 1] * sy

    out = {
        "people": [
            {
                "pose_keypoints": k.reshape(-1).tolist()
            }
        ]
    }

    with open(dst_pose_json, "w", encoding="utf-8") as f:
        json.dump(out, f)


def build_fvnt_trial_dataset(
    dataset_root,
    person_img,
    garment_rgb,
    garment_mask,
    person_parse,
    person_pose_json,
    pam_output_npy,
):
    # FVNT test dataset layout expected by FVNT/mine/dataset_viton_stage3_nomask_addpants.py
    # Root:
    #   test_pairs.txt
    #   test/image/person.jpg
    #   test/cloth/garment.jpg
    #   test/cloth-mask/garment.jpg
    #   test/image-parse/person.png
    #   test/pose/person_keypoints.json
    #   test/generate_parsing_cross/garmentperson.png
    test_dir = os.path.join(dataset_root, "test")
    image_dir = os.path.join(test_dir, "image")
    cloth_dir = os.path.join(test_dir, "cloth")
    cloth_mask_dir = os.path.join(test_dir, "cloth-mask")
    parse_dir = os.path.join(test_dir, "image-parse")
    pose_dir = os.path.join(test_dir, "pose")
    gen_parse_dir = os.path.join(test_dir, "generate_parsing_cross")

    for d in [image_dir, cloth_dir, cloth_mask_dir, parse_dir, pose_dir, gen_parse_dir]:
        ensure_dir(d)

    target_name = "person.jpg"
    source_name = "garment.jpg"

    target_img_out = os.path.join(image_dir, target_name)
    source_img_out = os.path.join(cloth_dir, source_name)
    source_mask_out = os.path.join(cloth_mask_dir, source_name)
    target_parse_out = os.path.join(parse_dir, "person.png")
    target_pose_out = os.path.join(pose_dir, "person_keypoints.json")

    # FVNT expects 192x256 tensors.
    out_size = (192, 256)

    save_rgb_jpg(person_img, target_img_out, size_wh=out_size)
    save_rgb_jpg(garment_rgb, source_img_out, size_wh=out_size)
    save_mask_jpg(garment_mask, source_mask_out, size_wh=out_size)
    save_parse_png(person_parse, target_parse_out, size_wh=out_size)

    # Pose from MediaPipe output is in original image coordinates: rescale to 192x256 for untouched FVNT loader.
    with Image.open(person_img) as im:
        src_w, src_h = im.size
    scale_pose_json_for_fvnt(person_pose_json, target_pose_out, src_w=src_w, src_h=src_h, dst_w=192, dst_h=256)

    # File naming logic in FVNT loader:
    # source_name = source.replace('/', '_').replace('.jpg', '') -> 'garment'
    # target_name = target.replace('/', '_').replace('.jpg', '.png') -> 'person.png'
    # file_name = source_name + target_name -> 'garmentperson.png'
    generated_parsing_name = "garmentperson.png"
    generated_parsing_out = os.path.join(gen_parse_dir, generated_parsing_name)

    pam20_to_label_png(pam_output_npy, generated_parsing_out)

    pairs_file = os.path.join(dataset_root, "test_pairs.txt")
    with open(pairs_file, "w", encoding="utf-8") as f:
        f.write(f"{target_name} {source_name}\n")

    return {
        "dataset_root": dataset_root,
        "pairs_file": pairs_file,
        "target_name": target_name,
        "source_name": source_name,
    }


def copy_fvnt_result_to_outputs(fvnt_result_dir, output_final_dir):
    ensure_dir(output_final_dir)
    # FVNT test.py names outputs by iteration index, so single-pair output is 0.png.
    src_candidates = [
        os.path.join(fvnt_result_dir, "0.png"),
        os.path.join(fvnt_result_dir, "00000.png"),
    ]
    src = first_existing(src_candidates, "FVNT stage-3 output image")
    dst = os.path.join(output_final_dir, "tryon_result_fvnt_native.png")
    shutil.copy2(src, dst)
    return dst


def main():
    parser = argparse.ArgumentParser(
        description="Alternate Trial Pipeline: Fashn + MediaPipe + PAM, then untouched FVNT stage-2+stage-3 via FVNT/test.py"
    )
    parser.add_argument("--person", help="Path to person image (auto-detected in inputs/ if omitted)")
    parser.add_argument("--garment", help="Path to garment image (auto-detected in inputs/ if omitted)")
    parser.add_argument("--type", choices=["flat", "worn"], default="flat", help="Garment type")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU for PAM/FVNT stages")
    parser.add_argument("--pam_mode", choices=["predict", "from_parse"], default="predict", help="Use PAM model prediction or parse->20ch fallback")

    # Environments
    parser.add_argument("--preprocess_env", default="densepose", help="Env for rembg, garment preprocessing, MediaPipe")
    parser.add_argument("--parser_env", default="densepose", help="Env for Fashn parser")
    parser.add_argument("--pam_env", default="fvnt_env", help="Env for PAM stage")
    parser.add_argument("--fvnt_env", default="fvnt_env", help="Env for untouched FVNT stage-2+stage-3 test.py")

    parser.add_argument("--conda_path", default=r"C:/Users/hp/anaconda3/condabin/conda.bat")
    parser.add_argument("--project_root", default="d:/VITON")
    parser.add_argument("--output_root", default="d:/VITON/outputs")

    # Checkpoints
    parser.add_argument("--pam_ckpt", default="", help="Optional explicit PAM checkpoint path")
    parser.add_argument("--fvnt_stage2_ckpt", default="d:/VITON/FVNT/model/stage2_model", help="FVNT stage-2 checkpoint")
    parser.add_argument("--fvnt_stage3_ckpt", default="d:/VITON/FVNT/model/stage3_model", help="FVNT stage-3 checkpoint")

    args = parser.parse_args()

    p_root = args.project_root
    o_root = args.output_root
    src_dir = os.path.join(p_root, "src")
    fvnt_root = os.path.join(p_root, "FVNT")

    ensure_dir(o_root)

    def conda_python(env_name):
        return [args.conda_path, "run", "--no-capture-output", "-n", env_name, "python"]

    person_img = args.person if args.person else find_input(p_root, "person.*", "person image")
    garment_img = args.garment if args.garment else find_input(p_root, "garment.*", "garment image")

    print("--- Using Inputs ---")
    print(f"Person : {person_img}")
    print(f"Garment: {garment_img}")
    print("--------------------")

    # Output paths
    rembg_dir = os.path.join(o_root, "rembg")
    garment_dir = os.path.join(o_root, "garment")
    parse_dir = os.path.join(o_root, "parse")
    pose_dir = os.path.join(o_root, "mediapipe")
    pam_dir = os.path.join(o_root, "pam")
    fvnt_trial_dir = os.path.join(o_root, "fvnt_trial_data")
    fvnt_result_dir = os.path.join(o_root, "fvnt_native_result")
    final_dir = os.path.join(o_root, "final")

    ensure_dir(final_dir)

    person_no_bg = os.path.join(rembg_dir, "person.png")
    person_parse = os.path.join(parse_dir, "person.png")
    person_pose = os.path.join(pose_dir, "person_keypoints.json")
    garment_rgb = os.path.join(garment_dir, "cloth.png")
    garment_mask = os.path.join(garment_dir, "cloth_mask.png")
    pam_npy = os.path.join(pam_dir, "predicted_parsing_20ch.npy")

    # Resolve PAM ckpt if not explicitly passed.
    pam_ckpt = args.pam_ckpt.strip() if args.pam_ckpt else ""
    if not pam_ckpt:
        pam_ckpt = first_existing(
            [
                os.path.join(p_root, "FVNT", "model", "stage_1", "G_stage1_best.pth"),
                os.path.join(p_root, "FVNT", "model", "stage1_model"),
                os.path.join(p_root, "checkpoints", "pam.pth"),
            ],
            "PAM checkpoint",
        )

    print("\n=== STAGE 1: PERSON BACKGROUND REMOVAL ===")
    if not run_cmd(
        conda_python(args.preprocess_env),
        os.path.join(src_dir, "remove_background.py"),
        ["--input", person_img, "--output_dir", rembg_dir],
    ):
        sys.exit(1)

    print("\n=== STAGE 2: GARMENT PREPROCESSING ===")
    pre_garment_args = ["--type", args.type, "--input", garment_img, "--output_dir", garment_dir]
    if not run_cmd(conda_python(args.preprocess_env), os.path.join(src_dir, "preprocess_garment.py"), pre_garment_args):
        sys.exit(1)

    print("\n=== STAGE 3: FASHN PARSING (CIHP-20) ===")
    parse_input_dir = os.path.join(o_root, "parse_input")
    ensure_dir(parse_input_dir)
    parse_person_path = os.path.join(parse_input_dir, "person.png")
    shutil.copy2(person_img, parse_person_path)

    if not run_cmd(
        conda_python(args.parser_env),
        os.path.join(src_dir, "run_fashn_parser.py"),
        ["--input_dir", parse_input_dir, "--output_dir", parse_dir],
    ):
        sys.exit(1)

    print("\n=== STAGE 4: MEDIAPIPE POSE (OPENPOSE-18 JSON) ===")
    if not run_cmd(
        conda_python(args.preprocess_env),
        os.path.join(src_dir, "run_pose_mediapipe.py"),
        ["--input", person_img, "--output_dir", pose_dir],
    ):
        sys.exit(1)

    print("\n=== STAGE 5: PAM (FVNT STAGE-1) ===")
    if args.pam_mode == "predict":
        if not run_cmd(
            conda_python(args.pam_env),
            os.path.join(src_dir, "run_pam.py"),
            [
                "--cloth", garment_rgb,
                "--cloth_mask", garment_mask,
                "--parse", person_parse,
                "--pose", person_pose,
                "--image", person_img,
                "--checkpoint", pam_ckpt,
                "--output_dir", pam_dir,
            ],
        ):
            sys.exit(1)
    else:
        print("[*] PAM bypass enabled: using parse map directly as PAM 20ch.")
        if not run_cmd(
            conda_python(args.parser_env),
            os.path.join(src_dir, "make_pam_from_parse.py"),
            [
                "--parse", person_parse,
                "--output_dir", pam_dir,
            ],
        ):
            sys.exit(1)

    print("\n=== STAGE 6: BUILD FVNT NATIVE TEST INPUTS ===")
    if os.path.isdir(fvnt_trial_dir):
        shutil.rmtree(fvnt_trial_dir)
    ensure_dir(fvnt_trial_dir)

    dataset_info = build_fvnt_trial_dataset(
        dataset_root=fvnt_trial_dir,
        person_img=person_img,
        garment_rgb=garment_rgb,
        garment_mask=garment_mask,
        person_parse=person_parse,
        person_pose_json=person_pose,
        pam_output_npy=pam_npy,
    )
    print(f"[*] Trial dataset ready: {dataset_info['dataset_root']}")

    print("\n=== STAGE 7: FVNT UNTOUCHED FEM+FRM (test.py via launcher) ===")
    if os.path.isdir(fvnt_result_dir):
        shutil.rmtree(fvnt_result_dir)
    ensure_dir(fvnt_result_dir)

    fvnt_args = [
        "--fvnt_root", fvnt_root,
        "--mode", "test",
        "--data_root", fvnt_trial_dir,
        "--file_path", os.path.basename(dataset_info["pairs_file"]),
        "--stage2_model", args.fvnt_stage2_ckpt,
        "--stage3_model", args.fvnt_stage3_ckpt,
        "--genetate_parsing", "generate_parsing_cross",
        "--result", fvnt_result_dir,
        "--height", "256",
        "--width", "192",
        "--image_size", "256",
    ]

    if not run_cmd(
        conda_python(args.fvnt_env),
        os.path.join(src_dir, "fvnt_test_launcher.py"),
        fvnt_args,
        cwd=fvnt_trial_dir,
    ):
        sys.exit(1)

    print("\n=== STAGE 8: COLLECT FVNT NATIVE OUTPUT ===")
    final_img = copy_fvnt_result_to_outputs(fvnt_result_dir=fvnt_result_dir, output_final_dir=final_dir)

    print("\n" + "=" * 64)
    print("ALTERNATE TRIAL PIPELINE COMPLETED (FVNT UNTOUCHED FINAL STAGE)")
    print(f"Final output: {final_img}")
    print(f"FVNT native result dir: {fvnt_result_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
