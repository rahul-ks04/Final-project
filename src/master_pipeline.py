import argparse
import glob
import os
import shutil
import subprocess
import sys


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


def main():
    parser = argparse.ArgumentParser(description="Master Pipeline (Fashn + MediaPipe + PAM + FVNT + LaMa Composition)")
    parser.add_argument("--person", help="Path to person image (auto-detected in inputs/ if omitted)")
    parser.add_argument("--garment", help="Path to garment image (auto-detected in inputs/ if omitted)")
    parser.add_argument("--type", choices=["flat", "worn"], default="flat", help="Garment type")
    parser.add_argument("--no_inpaint_skin", action="store_true", help="Disable LaMa inpainting and restore original pixels in holes")
    parser.add_argument("--inpaint_arms", action="store_true", help="Allow LaMa inpainting on arm regions (hands still protected)")
    parser.add_argument("--no_gpu", action="store_true", help="Force CPU for PAM/FVNT stages")
    parser.add_argument("--pam_mode", choices=["predict", "from_parse"], default="predict", help="Use PAM model prediction or direct parse->20ch fallback for debugging")
    parser.add_argument("--erase_arms_under_cloth", action="store_true", help="Aggressively erase arm pixels under cloth in composition")

    # Environments
    parser.add_argument("--preprocess_env", default="densepose", help="Env for rembg, garment preprocessing, MediaPipe")
    parser.add_argument("--parser_env", default="densepose", help="Env for Fashn parser")
    parser.add_argument("--pam_env", default="fvnt_env", help="Env for PAM stage")
    parser.add_argument("--fvnt_env", default="fvnt_env", help="Env for FVNT stage-2")
    parser.add_argument("--compose_env", default="densepose", help="Env for composition (LaMa)")

    parser.add_argument("--conda_path", default=r"C:/Users/hp/anaconda3/condabin/conda.bat")
    parser.add_argument("--project_root", default="d:/VITON")
    parser.add_argument("--output_root", default="d:/VITON/outputs")

    # Checkpoints
    parser.add_argument("--pam_ckpt", default="", help="Optional explicit PAM checkpoint path")
    parser.add_argument("--fvnt_ckpt", default="d:/VITON/FVNT/model/stage2_model", help="FVNT stage-2 checkpoint path")

    args = parser.parse_args()

    p_root = args.project_root
    o_root = args.output_root
    src_dir = os.path.join(p_root, "src")
    os.makedirs(o_root, exist_ok=True)

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
    flow_dir = os.path.join(o_root, "flow_renderer")
    final_dir = os.path.join(o_root, "final")
    os.makedirs(final_dir, exist_ok=True)

    person_no_bg = os.path.join(rembg_dir, "person.png")
    person_parse = os.path.join(parse_dir, "person.png")
    # Keypoints filename = {stem}_keypoints.json where stem = input image stem.
    # e.g. person.jpg -> person_keypoints.json, 09204_00.jpg -> 09204_00_keypoints.json
    _pose_stem = os.path.splitext(os.path.basename(person_img))[0]
    person_pose = os.path.join(pose_dir, f"{_pose_stem}_keypoints.json")
    person_hands_mask = os.path.join(pose_dir, f"{_pose_stem}_hands_mask.png")
    garment_rgb = os.path.join(garment_dir, "cloth.png")
    garment_mask = os.path.join(garment_dir, "cloth_mask.png")
    pam_npy = os.path.join(pam_dir, "predicted_parsing_20ch.npy")
    warped_garment = os.path.join(flow_dir, "warped_garment.png")
    warped_mask = os.path.join(flow_dir, "warped_mask.png")
    tryon_rgba = os.path.join(final_dir, "tryon_result.png")
    comp_path = os.path.join(final_dir, "tryon_with_background.png")

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
    os.makedirs(parse_input_dir, exist_ok=True)
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

    print("\n=== STAGE 6: FVNT FLOW RENDERER (STAGE-2) ===")
    flow_args = [
        "--pam_output", pam_npy,
        "--garment_rgb", garment_rgb,
        "--garment_mask", garment_mask,
        "--checkpoint", args.fvnt_ckpt,
        "--output_dir", flow_dir,
    ]
    if args.no_gpu:
        flow_args.append("--no_gpu")

    if not run_cmd(conda_python(args.fvnt_env), os.path.join(src_dir, "fvnt_flow_renderer.py"), flow_args):
        sys.exit(1)

    print("\n=== STAGE 7: COMPOSITION + LAMA INPAINT ===")
    compose_args = [
        "--original", person_no_bg,
        "--parse", person_parse,
        "--warped_cloth", warped_garment,
        "--warped_mask", warped_mask,
        "--hand_mask", person_hands_mask,
        "--output_path", tryon_rgba,
    ]
    if args.no_inpaint_skin:
        compose_args.append("--no_inpaint_skin")
    if args.inpaint_arms:
        compose_args.append("--inpaint_arms")
    if args.erase_arms_under_cloth:
        compose_args.append("--erase_arms_under_cloth")

    if not run_cmd(conda_python(args.compose_env), os.path.join(src_dir, "compose_tryon.py"), compose_args):
        sys.exit(1)

    print("\n=== STAGE 8: RESTORE ORIGINAL BACKGROUND ===")
    if not run_cmd(
        conda_python(args.compose_env),
        os.path.join(src_dir, "restore_background.py"),
        ["--original", person_img, "--tryon", tryon_rgba, "--rembg_mask", person_no_bg, "--output", comp_path],
    ):
        print("Warning: Background restoration failed, but tryon RGBA is available.")

    print("\n" + "=" * 60)
    print("NEW FVNT PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Try-on RGBA: {tryon_rgba}")
    print(f"Final output: {comp_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
