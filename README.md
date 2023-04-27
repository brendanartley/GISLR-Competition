# Google-ISR-Competition

Files for the Google - Isolated Sign Language Recognition competition.

## GPU Notes

22,919 MB per node

GPU info: `watch nvidia-smi`

Watch log file: `tail -n 25 -f logfile`

## Saved Weights

Downloading weights file
`cp PATH_TO_WEIGHTS ./`


## Sweeps Notes

Wandb Sweeps Docs: https://docs.wandb.ai/guides/sweeps
Sweeps Lesson Files: https://github.com/wandb/edu/tree/main/mlops-001/lesson2

Run a sweep on a specific GPU
```
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/GISLR-keras/5v4ph7k7
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/GISLR-keras/5v4ph7k7
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/GISLR-keras/5v4ph7k7
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/GISLR-keras/5v4ph7k7
```

## Training Notes

Test Run
```
CUDA_VISIBLE_DEVICES=0 python train.py --lr_decay=True --num_cycles=5.5 --learning_rate=5e-3 --no_wandb --max_epochs=1 --verbose=1 --aug=True --aug_rotate_ratio=0.15 --aug_rotate_degrees=15 --aug_expand_ratio=0.15 --aug_expand_pct=0.05
CUDA_VISIBLE_DEVICES=1 python train.py --lr_decay=True --num_cycles=5.5 --learning_rate=5e-3 --no_wandb --max_epochs=1 --verbose=1 --aug=True --aug_rotate_ratio=0.15 --aug_rotate_degrees=15 --aug_expand_ratio=0.15 --aug_expand_pct=0.05
CUDA_VISIBLE_DEVICES=2 python train.py --lr_decay=True --num_cycles=5.5
CUDA_VISIBLE_DEVICES=3 python train.py

CUDA_VISIBLE_DEVICES=0 python train.py --no_wandb --max_epochs=1 --verbose=1
CUDA_VISIBLE_DEVICES=1 python train.py --file gislr-mw-16b --no_wandb --max_epochs=1 --verbose=1 --lr_decay=True --num_cycles=5.5 --augment=True --augment_ratio=0.90 --augment_scale=50
CUDA_VISIBLE_DEVICES=2 python train.py --file gislr-mw-16b --no_wandb --max_epochs=1 --verbose=1 --lr_decay=True --num_cycles=6.5
CUDA_VISIBLE_DEVICES=3 python train.py --file gislr-mw-16b --no_wandb --max_epochs=1 --verbose=1 --lr_decay=True --num_cycles=3.5
```