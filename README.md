# Google-ISR-Competition

Files for the Google - Isolated Sign Language Recognition competition.

## GPU Notes

22,919 MB per node

Log GPU usage in 1 sec intervals: `nvidia-smi dmon -s mu -d 5 -o TD`

Watch log file: `tail -n 25 -f logfile`


## Sweeps Notes

Wandb Sweeps Docs: https://docs.wandb.ai/guides/sweeps
Sweeps Lesson Files: https://github.com/wandb/edu/tree/main/mlops-001/lesson2

Run a sweep on a specific GPU
```
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/GISLR-keras/zgc28xe2
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/GISLR-keras/zgc28xe2
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/GISLR-keras/zgc28xe2
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/GISLR-keras/zgc28xe2
```

Triplet Tests
```
TF_GPU_ALLOCATOR=cuda_malloc_async CUDA_VISIBLE_DEVICES=0 python main.py --do_triplet --max_epochs 1 --no_wandb --verbose 1
TF_GPU_ALLOCATOR=cuda_malloc_async CUDA_VISIBLE_DEVICES=0 python main.py --do_triplet --triplet_epochs 2 --max_epochs 1 --no_wandb --verbose 1

CUDA_VISIBLE_DEVICES=0 python main.py --no_wandb --max_epochs 1 --do_triplet
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 1 --do_triplet
```