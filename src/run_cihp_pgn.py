import os
import subprocess
import argparse
import sys
import shutil

def run_cihp_pgn(input_dir, output_dir, project_root):
    """
    Wraps test_pgn.py from CIHP_PGN.
    Creates necessary list files and directory structure.
    """
    cihp_root = os.path.join(project_root, "CIHP_PGN")
    # Path to weights - inferred from user's placement
    checkpoint_name = "model.ckpt-593292"
    model_path = os.path.join(cihp_root, "checkpoint", checkpoint_name)
    
    # 1. Prepare datasets structure for CIHP_PGN
    cihp_datasets = os.path.join(cihp_root, "datasets", "CIHP")
    cihp_list_dir = os.path.join(cihp_datasets, "list")
    os.makedirs(cihp_list_dir, exist_ok=True)
    
    # We need to link the input images to cihp_datasets/images
    target_images_dir = os.path.join(cihp_datasets, "images")
    if os.path.exists(target_images_dir):
        shutil.rmtree(target_images_dir)
    shutil.copytree(input_dir, target_images_dir)
    
    # Generate val.txt and val_id.txt
    image_files = [f for f in os.listdir(target_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    with open(os.path.join(cihp_list_dir, "val.txt"), "w") as f, \
         open(os.path.join(cihp_list_dir, "val_id.txt"), "w") as f_id:
        for img in image_files:
            img_id = os.path.splitext(img)[0]
            f.write(f"/images/{img} /images/{img}\n") # Placeholder label
            f_id.write(f"{img_id}\n")

    # 2. Modify test_pgn.py paths temporarily or pass via environment if possible
    # Since test_pgn.py has hardcoded paths, it's easier to run it with relative paths
    
    cmd = [
        sys.executable, "test_pgn.py"
    ]
    
    # We might need to override the RESTORE_FROM in test_pgn.py
    # Let's create a temporary patched test_pgn.py
    patched_test_pgn = os.path.join(cihp_root, "test_pgn_patched.py")
    with open(os.path.join(cihp_root, "test_pgn.py"), "r") as f:
        content = f.read()
    
    # Update RESTORE_FROM and NUM_STEPS (though it tries to read val_id.txt)
    # The load() function in utils expects a directory containing a 'checkpoint' file
    content = content.replace("RESTORE_FROM = './checkpoint/CIHP_pgn'", "RESTORE_FROM = './checkpoint'")
    
    with open(patched_test_pgn, "w") as f:
        f.write(content)

    print(f"Running CIHP_PGN in {cihp_root}...")
    
    # Run
    try:
        # Note: test_pgn.py saves results to ./output/cihp_parsing_maps
        result = subprocess.run([sys.executable, "test_pgn_patched.py"], cwd=cihp_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"CIHP_PGN failed with error: {e}")
        sys.exit(1)
        
    # 3. Move results to output_dir
    cihp_output_maps = os.path.join(cihp_root, "output", "cihp_parsing_maps")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for img in image_files:
        img_id = os.path.splitext(img)[0]
        src_path = os.path.join(cihp_output_maps, f"{img_id}.png")
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(output_dir, f"{img_id}.png"))
        else:
            print(f"Warning: Could not find result for {img_id}")

    print(f"CIHP_PGN complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", required=True)
    args = parser.parse_args()
    run_cihp_pgn(args.input_dir, args.output_dir, args.project_root)
