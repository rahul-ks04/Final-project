import argparse
import os
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description="Launcher for untouched FVNT test.py with Windows-safe runtime patches")
    parser.add_argument("--fvnt_root", required=True)
    parser.add_argument("--mode", default="test")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--stage2_model", required=True)
    parser.add_argument("--stage3_model", required=True)
    parser.add_argument("--genetate_parsing", default="generate_parsing_cross")
    parser.add_argument("--result", required=True)
    parser.add_argument("--height", default="256")
    parser.add_argument("--width", default="192")
    parser.add_argument("--image_size", default="256")
    args = parser.parse_args()

    # Runtime-only compatibility patch: force DataLoader num_workers=0 on Windows
    # so worker re-import does not execute module-level CUDA initialization in FVNT/test.py.
    import torch
    import torch.utils.data as tud

    original_dataloader = tud.DataLoader

    def dataloader_no_workers(*dl_args, **dl_kwargs):
        dl_kwargs["num_workers"] = 0
        return original_dataloader(*dl_args, **dl_kwargs)

    tud.DataLoader = dataloader_no_workers

    # If CUDA is unavailable in this environment, keep untouched FVNT logic runnable on CPU.
    if not torch.cuda.is_available():
        import torch.nn as nn

        def _module_cuda_noop(self, device=None):
            return self

        def _tensor_cuda_noop(self, device=None, non_blocking=False, memory_format=None):
            return self

        nn.Module.cuda = _module_cuda_noop
        torch.Tensor.cuda = _tensor_cuda_noop

    fvnt_test = os.path.join(args.fvnt_root, "test.py")
    if not os.path.isfile(fvnt_test):
        raise FileNotFoundError(f"FVNT test.py not found: {fvnt_test}")

    if args.fvnt_root not in sys.path:
        sys.path.insert(0, args.fvnt_root)

    # Build argv exactly as FVNT/test.py expects.
    sys.argv = [
        fvnt_test,
        "--mode", args.mode,
        "--data_root", args.data_root,
        "--file_path", args.file_path,
        "--stage2_model", args.stage2_model,
        "--stage3_model", args.stage3_model,
        "--genetate_parsing", args.genetate_parsing,
        "--result", args.result,
        "--height", str(args.height),
        "--width", str(args.width),
        "--image_size", str(args.image_size),
    ]

    runpy.run_path(fvnt_test, run_name="__main__")


if __name__ == "__main__":
    main()
