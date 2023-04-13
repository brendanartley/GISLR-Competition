import json
import socket
import wandb

class CFG:
    if socket.gethostname() == 'gpu1':
        LOG_DATA_DIR = "/data/bartley/gpu_test/"
        MY_DATA_DIR = LOG_DATA_DIR + "my-gislr-data/"
        COMP_DATA_DIR = LOG_DATA_DIR + "asl-signs/"
        OTS_DATA_DIR = LOG_DATA_DIR + "my-gislr-data/" 

        with open("config.json", "r+") as f:
            wandb_key = json.load(f)['wandb_key']

    else:
        LOG_DATA_DIR = '/kaggle/working/'
        MY_DATA_DIR = '/kaggle/input/my-gislr-data/'
        COMP_DATA_DIR = "/kaggle/input/asl-signs/"
        OTS_DATA_DIR = "/kaggle/input/gislr-feature-data-on-the-shoulders/" 

        from kaggle_secrets import UserSecretsClient # type: ignore
        wandb_key = UserSecretsClient().get_secret("wandb") # type: ignore
    
    wandb.login(key=wandb_key)