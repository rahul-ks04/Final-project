import os
import argparse
import subprocess
import shutil
import sys
from pathlib import Path

def run_cmd(python_cmd, script_path, args, cwd=None):
    """
    python_cmd: list or string. For conda envs, use ['conda', 'run', '-n', 'envname', 'python']
    """
    if isinstance(python_cmd, str):
        python_cmd = [python_cmd]
        
    cmd = python_cmd + [script_path] + args
    # Quote arguments with spaces
    quoted_cmd = ['"' + str(arg) + '"' if ' ' in str(arg) else str(arg) for arg in cmd]
    print(f"\n>>> Running: {' '.join(quoted_cmd)}")
    
    is_windows = os.name == 'nt'
    result = subprocess.run(cmd, cwd=cwd, shell=is_windows)
    
    if result.returncode != 0:
        print(f"!!! Error return code {result.returncode} from command.")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Master VITON Inference Pipeline")
    parser.add_argument("--person", help="Path to person image (auto-detected in inputs/ if omitted)")
    parser.add_argument("--garment", help="Path to garment image (auto-detected in inputs/ if omitted)")
    parser.add_argument("--type", choices=["flat", "worn"], default="flat", help="Garment type")
    parser.add_argument("--sleeve_type", "--sleeve_length", choices=["none", "half", "full"], default="full", dest="sleeve_type", help="Sleeve type of target garment (also accepts --sleeve_length)")
    parser.add_argument("--initial_sleeve", choices=["none", "half", "full"], default="half", help="Initial sleeve type of the person (what they are wearing in the photo)")
    parser.add_argument("--preserve_arms", action="store_true", help="Preserve original arms in agnostic generation")
    parser.add_argument("--inpaint_skin", action="store_true", default=True, help="Enable GAN-based skin inpainting for occluded arms (e.g. full-to-half sleeve)")
    
    # Environment Names
    parser.add_argument("--parser_env", default="densepose", help="Env for Human Parsing (Fashn Parser)")
    parser.add_argument("--dp_env", default="densepose")
    parser.add_argument("--fvnt_env", default="fvnt_env")
    parser.add_argument("--stylevton_env", default="stylevton")
    
    # Conda executable path (using forward slashes for Windows compatibility)
    parser.add_argument("--conda_path", default=r"C:/Users/hp/anaconda3/condabin/conda.bat")

    # Project and Checkpoints
    parser.add_argument("--project_root", default="d:/Final Project Viton/virtual-tryon")
    parser.add_argument("--fvnt_ckpt", default="d:/Final Project Viton/virtual-tryon/FVNT/model/stage2_model")
    parser.add_argument("--stylevton_ckpt", default="d:/Final Project Viton/virtual-tryon/Flow-Style-VTON/checkpoints/ckp/non_aug/PFAFN_gen_epoch_101.pth")
    parser.add_argument("--output_root", default="d:/Final Project Viton/virtual-tryon/outputs")

    args = parser.parse_args()
    
    p_root = args.project_root
    o_root = args.output_root
    os.makedirs(o_root, exist_ok=True)

    # Conda runner helper
    def conda_python(env_name):
        return [args.conda_path, "run", "--no-capture-output", "-n", env_name, "python"]

    # --- Auto-Detection Trace ---
    def find_input(pattern, label):
        import glob
        matches = glob.glob(os.path.join(p_root, "inputs", pattern))
        if not matches:
            print(f"!!! Error: Could not find {label} matching '{pattern}' in inputs/ folder.")
            print(f"Please upload a file named '{pattern.replace('*', 'your_ext')}' to {os.path.join(p_root, 'inputs')}")
            sys.exit(1)
        # Prefer jpeg/jpg/png
        for ext in ['.jpg', '.jpeg', '.png']:
            for m in matches:
                if m.lower().endswith(ext):
                    return m
        return matches[0]

    person_img = args.person if args.person else find_input("person.*", "person image")
    garment_img = args.garment if args.garment else find_input("garment.*", "garment image")
    
    print(f"--- Using Inputs ---")
    print(f"Person: {person_img}")
    print(f"Garment: {garment_img}")
    print(f"--------------------")

    # Paths to generated files
    person_no_bg = os.path.join(o_root, "rembg", "person.png")
    person_parse = os.path.join(o_root, "parse", "person.png")
    densepose_img = os.path.join(o_root, "densepose", "person_densepose.png")
    garment_rgb = os.path.join(o_root, "garment", "cloth.png")
    garment_mask = os.path.join(o_root, "garment", "cloth_mask.png")
    target_mask = os.path.join(o_root, "target_mask", "target_mask.png")
    
    flow_dir = os.path.join(o_root, "flow_renderer")
    warped_garment = os.path.join(flow_dir, "warped_garment.png")
    projected_mask = os.path.join(flow_dir, "projected_mask.png")
    
    agnostic_dir = os.path.join(o_root, "agnostic")
    agnostic_img = os.path.join(agnostic_dir, "img", "person.png")
    agnostic_mask = os.path.join(agnostic_dir, "mask", "person.png")
    
    final_output = os.path.join(o_root, "final", "tryon_result.png")
    os.makedirs(os.path.dirname(final_output), exist_ok=True)

    # 1. Preprocessing Pipeline
    print("\n=== PHASE 1: PREPROCESSING ===")
    pre_args = [
        "--person", person_img,
        "--garment", garment_img,
        "--type", args.type,
        "--sleeve_type", args.sleeve_type,
        "--output_root", o_root,
        "--project_root", p_root,
        "--schp_py", " ".join(conda_python(args.parser_env)),
        "--dp_py", " ".join(conda_python(args.dp_env)),
        "--conda_path", args.conda_path
    ]
    # Run preprocessing in the densepose environment (has rembg, torch, etc.)
    if not run_cmd(conda_python(args.dp_env), os.path.join(p_root, "src/preprocess_pipeline.py"), pre_args):
        sys.exit(1)

    # 2. FVNT Flow Renderer
    print("\n=== PHASE 2: FLOW ESTIMATION (FVNT) ===")
    flow_args = [
        "--person", target_mask,
        "--garment_rgb", garment_rgb,
        "--garment_mask", garment_mask,
        "--checkpoint", args.fvnt_ckpt,
        "--output_dir", flow_dir,
        "--schp", person_parse
    ]
    if not run_cmd(conda_python(args.fvnt_env), os.path.join(p_root, "src/fvnt_flow_renderer.py"), flow_args):
        sys.exit(1)

    # 3. Agnostic Person Generation (Layering Mode)
    print("\n=== PHASE 3: AGNOSTIC GENERATION (Layering) ===")
    agnostic_args = [
        "--image", person_no_bg,
        "--parse", person_parse,
        "--output_dir", agnostic_dir,
        "--warped_mask", projected_mask
    ]
    if not run_cmd(conda_python(args.dp_env), os.path.join(p_root, "src/generate_agnostic_person.py"), agnostic_args):
        sys.exit(1)

    # 4. Layered Try-On Composition (surgical compositing, no GAN)
    print("\n=== PHASE 4: LAYERED TRY-ON COMPOSITION ===")
    style_args = [
        "--agnostic", agnostic_img,
        "--original", person_no_bg,
        "--agnostic_mask", agnostic_mask,
        "--warped_cloth", warped_garment,
        "--warped_mask", projected_mask,
        "--parse", person_parse,
        "--densepose", densepose_img,
        "--output_path", final_output,
        "--initial_sleeve", args.initial_sleeve
    ]
    if args.inpaint_skin:
        style_args += ["--inpaint_skin", "--checkpoint", args.stylevton_ckpt]

    if not run_cmd(conda_python(args.stylevton_env), os.path.join(p_root, "src/run_stylevton.py"), style_args):
        sys.exit(1)

    # 5. Result Compositing
    print("\n=== PHASE 5: RESULT COMPOSITING ===")
    comp_path = os.path.join(o_root, "final", "tryon_with_background.png")
    composite_cmd = [
        "--original", person_img,
        "--tryon", final_output,
        "--rembg_mask", person_no_bg, # This is actually the person image with background removed, not just the mask
        "--output", comp_path
    ]
    # We can use stylevton env or densepose env for this, it just needs PIL and numpy
    if not run_cmd(conda_python(args.stylevton_env), os.path.join(p_root, "src/restore_background.py"), composite_cmd):
        print("Warning: Background restoration failed, but tryon result is preserved.")

    print("\n" + "="*50)
    print("MASTER PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Final Tryon: {final_output}")
    print(f"Composite Result: {comp_path}")
    print("="*50)

if __name__ == "__main__":
    main()
