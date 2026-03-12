import os
import argparse
import subprocess
import shutil
import sys

def run_cmd(cmd, cwd=None):
    # Use shell=True on Windows to help resolve commands
    is_windows = os.name == 'nt'
    # If cmd is a list, we might need to handle quoting if windows shell is used
    # But usually subprocess handles this.
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, shell=is_windows)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        return False
    return True

def main(person_image, garment_image, garment_type, sleeve_type, output_root, project_root, schp_python="python", dp_python="python", conda_path=None):
    # Split the python commands into lists (handles "conda run -n env python")
    import shlex
    schp_python_cmd = shlex.split(schp_python)
    dp_python_cmd = shlex.split(dp_python)
    gen_python_cmd = [sys.executable] # Use current python for base steps
    
    # Setup output paths
    rembg_dir = os.path.join(output_root, "rembg")
    parse_dir = os.path.join(output_root, "parse")
    densepose_dir = os.path.join(output_root, "densepose")
    garment_dir = os.path.join(output_root, "garment")
    target_mask_dir = os.path.join(output_root, "target_mask")
    
    os.makedirs(output_root, exist_ok=True)
    
    # 1. Remove background from person image
    print("\n--- Step 1: Background Removal (rembg) ---")
    if not run_cmd(gen_python_cmd + [os.path.join(project_root, "src/remove_background.py"), 
                    "--input", person_image, "--output_dir", rembg_dir]):
        return False

    person_no_bg = os.path.join(rembg_dir, "person.png")
    
    # 2. Run Fashn Parser for parsing
    parse_input_dir = os.path.join(output_root, "temp_parse_input")
    if os.path.exists(parse_input_dir):
        shutil.rmtree(parse_input_dir)
    os.makedirs(parse_input_dir, exist_ok=True)
    shutil.copy(person_no_bg, os.path.join(parse_input_dir, "person.png"))
    
    print("\n--- Step 2: Human Parsing (Fashn Parser) ---")
    if not run_cmd(schp_python_cmd + [os.path.join(project_root, "src/run_fashn_parser.py"), 
                    "--input_dir", parse_input_dir, "--output_dir", parse_dir]):
        return False
    
    parse_mask = os.path.join(parse_dir, "person.png")
    
    # 3. Run DensePose
    print("\n--- Step 3: DensePose Extraction ---")
    if not run_cmd(dp_python_cmd + [os.path.join(project_root, "src/run_densepose.py"), 
                    "--input", person_no_bg, "--output_dir", densepose_dir, "--project_root", project_root]):
        return False
    
    densepose_mask = os.path.join(densepose_dir, "person_densepose.png")
    
    # 4. Preprocess Garment
    print("\n--- Step 4: Garment Preprocessing ---")
    garment_cmd = gen_python_cmd + [os.path.join(project_root, "src/preprocess_garment.py"),
                   "--type", garment_type, "--input", garment_image, "--output_dir", garment_dir]
    if garment_type == "worn":
        # Run SCHP on the garment image (the person wearing the target garment)
        # to get their parsing mask, so we can extract just the garment region
        schp_garment_input_dir = os.path.join(output_root, "temp_schp_garment_input")
        schp_garment_dir = os.path.join(output_root, "schp_garment")
        if os.path.exists(schp_garment_input_dir):
            shutil.rmtree(schp_garment_input_dir)
        os.makedirs(schp_garment_input_dir, exist_ok=True)
        shutil.copy(garment_image, os.path.join(schp_garment_input_dir, "garment.png"))
        
        print("\n--- Step 4a: Parsing Garment Wearer (SCHP) ---")
        if not run_cmd(schp_python_cmd + [os.path.join(project_root, "src/run_schp.py"),
                        "--input_dir", schp_garment_input_dir, "--output_dir", schp_garment_dir, "--project_root", project_root]):
            return False
        
        garment_schp_mask = os.path.join(schp_garment_dir, "garment.png")
        garment_cmd += ["--schp_mask", garment_schp_mask]
        
        # Clean up temp dir
        if os.path.exists(schp_garment_input_dir):
            shutil.rmtree(schp_garment_input_dir)
        
    if not run_cmd(garment_cmd):
        return False
        
    # 5. Generate Target Mask
    print("\n--- Step 5: Target Mask Generation ---")
    if not run_cmd(gen_python_cmd + [os.path.join(project_root, "src/generate_target_mask.py"),
                    "--schp", parse_mask, "--densepose", densepose_mask, 
                    "--person_mask", person_no_bg,
                    "--output_dir", target_mask_dir, "--sleeve_type", sleeve_type]):
        return False

    # Clean up temp dir
    if os.path.exists(parse_input_dir):
        shutil.rmtree(parse_input_dir)

    print("\n" + "="*40)
    print("Preprocessing Pipeline Finished Successfully!")
    print("="*40)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-click Preprocessing Pipeline for Viton")
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--type", choices=["flat", "worn"], default="flat")
    parser.add_argument("--sleeve_type", choices=["none", "half", "full"], default="full")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--schp_py", default="python")
    parser.add_argument("--dp_py", default="python")
    parser.add_argument("--conda_path", default=None)
    
    args = parser.parse_args()
    if not main(args.person, args.garment, args.type, args.sleeve_type, args.output_root, args.project_root, args.schp_py, args.dp_py, args.conda_path):
        sys.exit(1)
