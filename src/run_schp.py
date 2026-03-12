import os
import subprocess
import argparse
import sys

def run_schp(input_dir, output_dir, project_root):
    """
    Wraps simple_extractor.py from Self-Correction-Human-Parsing.
    Assumes current python environment has the necessary dependencies.
    """
    # Paths relative to project root
    schp_root = os.path.join(project_root, "Self-Correction-Human-Parsing")
    model_path = os.path.join(schp_root, "checkpoints", "schp.pth")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    cmd = [
        sys.executable, "simple_extractor.py",
        "--dataset", "lip",
        "--model-restore", model_path,
        "--input-dir", input_dir,
        "--output-dir", output_dir
    ]
    
    print(f"Running SCHP in {schp_root}...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run with check=True to raise error if it fails
    try:
        result = subprocess.run(cmd, cwd=schp_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"SCHP failed with error: {e}")
        sys.exit(1)
        
    print(f"SCHP complete. Results in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--project_root", required=True)
    args = parser.parse_args()
    run_schp(args.input_dir, args.output_dir, args.project_root)
