import json
import wandb
import numpy as np

"""
Very simple script to test the W&B Logger.
"""

def main():

    with open("config.json", "r+") as f:
        wandb_key = json.load(f)['wandb_key']
    wandb.login(key=wandb_key)
    wandb.init(project="my-awesome-project")

    ys = np.random.choice([1,2,3], size=100) 
    class_labels = ['a', 'b', 'c']

    for i in range(10):
        y_preds = np.random.choice([1,2,3], size=100)

        train_acc = (ys == y_preds).sum() / 100

        wandb.log({'accuracy': train_acc, 'loss': np.random.random()}, commit=False)

    # Get the name of the current WandB run
    run_name = wandb.run.name
    print("WandB run name:", run_name)


if __name__ == '__main__':
    main()