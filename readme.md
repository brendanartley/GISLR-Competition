## GPU Notes

22,919 MB per node

Log GPU usage in 1 sec intervals: `nvidia-smi dmon -s mu -d 5 -o TD`
Watch log file: `tail -n 25 -f logfile`

Pass: #Qa123456AI


## Sweeps Notes

Wandb Sweeps Docs: https://docs.wandb.ai/guides/sweeps
Sweeps Lesson Files: https://github.com/wandb/edu/tree/main/mlops-001/lesson2

Run a sweep on a specific GPU
CUDA_VISIBLE_DEVICES=0 wandb agent sweep_ID
CUDA_VISIBLE_DEVICES=1 wandb agent sweep_ID

Ex.
CUDA_VISIBLE_DEVICES=0 wandb agent brendanartley/GISLR-keras/zgc28xe2
CUDA_VISIBLE_DEVICES=1 wandb agent brendanartley/GISLR-keras/zgc28xe2
CUDA_VISIBLE_DEVICES=2 wandb agent brendanartley/GISLR-keras/zgc28xe2
CUDA_VISIBLE_DEVICES=3 wandb agent brendanartley/GISLR-keras/zgc28xe2


CUDA_VISIBLE_DEVICES=0 python main.py --do_wandb_log --max_epochs 1 --do_triplet
CUDA_VISIBLE_DEVICES=1 python main.py --max_epochs 1 --do_triplet


git -c user.email=brendan.artley@gmail.com -c user.name='brendanartley'