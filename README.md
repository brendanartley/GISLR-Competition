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

CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/GISLR-keras/vpuw6r5a
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/GISLR-keras/vpuw6r5a
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/GISLR-keras/vpuw6r5a
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/GISLR-keras/vpuw6r5a
```

## Triplet Training Notes

Test Run
```
CUDA_VISIBLE_DEVICES=0  python main.py --no_wandb --triplet true --triplet_epochs 1 --max_epochs 1 --verbose 1
CUDA_VISIBLE_DEVICES=1 python main.py --no_wandb --triplet true --triplet_epochs 1 --max_epochs 1 --verbose 1
```

Assorted testing commands
```
CUDA_VISIBLE_DEVICES=0 python main.py --batch_all_signs_n=4 --triplet=True --triplet_alpha=1000 --triplet_dist=eu --triplet_epochs=25 --triplet_learning_rate=0.0001
CUDA_VISIBLE_DEVICES=1 python main.py --batch_all_signs_n=4 --triplet=True --triplet_alpha=1000 --triplet_dist=sq --triplet_epochs=25 --triplet_learning_rate=0.0001
CUDA_VISIBLE_DEVICES=2 python main.py --batch_all_signs_n=8 --triplet=True --triplet_alpha=1000 --triplet_dist=eu --triplet_epochs=25 --triplet_learning_rate=0.0001
CUDA_VISIBLE_DEVICES=3 python main.py --batch_all_signs_n=8 --triplet=True --triplet_alpha=1000 --triplet_dist=sq --triplet_epochs=25 --triplet_learning_rate=0.0001

CUDA_VISIBLE_DEVICES=1 python main.py --no_train --triplet true --triplet_epochs 25 --triplet_dist sq --triplet_alpha 10 --no_wandb --verbose 1 --batch_all_signs_n 4
CUDA_VISIBLE_DEVICES=2 python main.py --no_train --triplet true --triplet_epochs 25 --triplet_dist sq --triplet_alpha 100 --no_wandb --verbose 1 --batch_all_signs_n 4
CUDA_VISIBLE_DEVICES=3 python main.py --no_train --triplet true --triplet_epochs 25 --triplet_dist sq --triplet_alpha 1000 --no_wandb --verbose 1 --batch_all_signs_n 4
```

Using pre-trained embeddings
```
CUDA_VISIBLE_DEVICES=3 python main.py --triplet_fname 1681842019_embeddings
```