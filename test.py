import wandb
import numpy as np
def main():
    wandb.init(project='my-awesome-project')

    for i in range(7):
        wandb.log({"fold-{} Val-Acc".format(i): np.random.random()})

    return

if __name__ == '__main__':
    main()