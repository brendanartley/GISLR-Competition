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

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/GISLR-keras/vqdxwsc6
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/GISLR-keras/vqdxwsc6
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/GISLR-keras/vqdxwsc6
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/GISLR-keras/vqdxwsc6
```

## Training Notes

Test Run
```
CUDA_VISIBLE_DEVICES=0 python train.py --file gislr-mw-16f
CUDA_VISIBLE_DEVICES=1 python train.py --file gislr-mw-16f
CUDA_VISIBLE_DEVICES=2 python train.py --file gislr-mw-16f
CUDA_VISIBLE_DEVICES=3 python train.py --file gislr-mw-16f

CUDA_VISIBLE_DEVICES=0 python train.py --file gislr-mw-16b --no_wandb --max_epochs=1 --verbose=1
CUDA_VISIBLE_DEVICES=1 python train.py --file gislr-mw-16b --no_wandb --max_epochs=1 --verbose=1
CUDA_VISIBLE_DEVICES=2 python train.py --file gislr-mw-16f --no_wandb --max_epochs=1 --verbose=1
CUDA_VISIBLE_DEVICES=3 python train.py --file gislr-mw-16f --no_wandb --max_epochs=1 --verbose=1

Assorted testing commands
```
CUDA_VISIBLE_DEVICES=0 python train.py --batch_all_signs_n=4 --triplet=True --triplet_dist=eu --triplet_epochs=25
CUDA_VISIBLE_DEVICES=1 python train.py --batch_all_signs_n=4 --triplet=True --triplet_dist=eu --triplet_epochs=25
CUDA_VISIBLE_DEVICES=2 python train.py --batch_all_signs_n=8 --triplet=True --triplet_dist=eu --triplet_epochs=25
CUDA_VISIBLE_DEVICES=3 python train.py --batch_all_signs_n=8 --triplet=True --triplet_dist=eu --triplet_epochs=25

CUDA_VISIBLE_DEVICES=1 python train.py --no_train --triplet true --triplet_epochs 25 --triplet_dist sq --triplet_alpha 10 --no_wandb --verbose 1 --batch_all_signs_n 4
CUDA_VISIBLE_DEVICES=2 python train.py --no_train --triplet true --triplet_epochs 25 --triplet_dist sq --triplet_alpha 100 --no_wandb --verbose 1 --batch_all_signs_n 4
CUDA_VISIBLE_DEVICES=3 python train.py --no_train --triplet true --triplet_epochs 25 --triplet_dist sq --triplet_alpha 1000 --no_wandb --verbose 1 --batch_all_signs_n 4
```

Using pre-trained embeddings
```
CUDA_VISIBLE_DEVICES=3 python main.py --triplet_fname 1681842019_embeddings
```