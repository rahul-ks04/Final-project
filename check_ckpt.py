import torch
import os

ckpt_path = 'd:/VITON/FVNT/model/stage1_model'
print(f'Checkpoint path: {ckpt_path}')
print(f'File exists: {os.path.isfile(ckpt_path)}')
print(f'File size: {os.path.getsize(ckpt_path) / 1024 / 1024:.1f} MB')

try:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f'Successfully loaded checkpoint')
    if isinstance(ckpt, dict):
        print(f'Checkpoint is dict with keys: {list(ckpt.keys())}')
        for key in ckpt.keys():
            if isinstance(ckpt[key], dict):
                sample_keys = list(ckpt[key].keys())[:3]
                print(f'  {key}: dict with {len(ckpt[key])} keys, sample: {sample_keys}')
            elif isinstance(ckpt[key], torch.Tensor):
                print(f'  {key}: tensor shape {ckpt[key].shape}')
            else:
                print(f'  {key}: {type(ckpt[key]).__name__}')
    else:
        print(f'Checkpoint is {type(ckpt).__name__}')
        if isinstance(ckpt, torch.Tensor):
            print(f'Shape: {ckpt.shape}')
except Exception as e:
    print(f'Error loading checkpoint: {e}')
    import traceback
    traceback.print_exc()
