import socket, json
import numpy as np
import tensorflow as tf
import wandb

class CFG:
    if socket.gethostname() == 'gpu1':
        LOG_DATA_DIR = "/data/bartley/gpu_test/"
        MY_DATA_DIR = LOG_DATA_DIR + "my-gislr-data/"
        COMP_DATA_DIR = LOG_DATA_DIR + "asl-signs/"
        MW_DATA_DIR = LOG_DATA_DIR + "gislr-mw-16/"
        WEIGHTS_DIR = LOG_DATA_DIR + "saved_weights/"
        TRIPLET_DATA = LOG_DATA_DIR + "gislr-triplet-data/"

        with open("config.json", "r+") as f:
            wandb_key = json.load(f)['wandb_key']

    else:
        LOG_DATA_DIR = '/kaggle/working/'
        MY_DATA_DIR = '/kaggle/input/my-gislr-data/'
        COMP_DATA_DIR = "/kaggle/input/asl-signs/"
        MW_DATA_DIR = "/kaggle/input/gislr-mw-16/"
        WEIGHTS_DIR = "/kaggle/input/my-gislr-data/"

        from kaggle_secrets import UserSecretsClient # type: ignore
        wandb_key = UserSecretsClient().get_secret("wandb") # type: ignore
    
    wandb.login(key=wandb_key)
    
    MODEL_PATH = LOG_DATA_DIR + "mymodel.h5"
    INFER_PATH = LOG_DATA_DIR + "tf_model_infer"
    TFLITE_PATH = LOG_DATA_DIR + "model.tflite"
    
    # Used in Kaggle env
    SUBMIT = True

    # ---------- Data Config ----------
    N_ROWS = 543
    N_DIMS = 2
    INPUT_SIZE = 64

    # Dense layer units for landmarks
    LIPS_UNITS = 384
    HANDS_UNITS = 384
    POSE_UNITS = 384

    LIPS_IDXS0 = np.array([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ])

    EYES_IDXS0 = np.array([33,159,133,145,362,386,263,374,1,2,98,327])

    # Landmark indices in original data
    LEFT_HAND_IDXS0 = np.arange(468,489)
    RIGHT_HAND_IDXS0 = np.arange(522,543)
    LEFT_POSE_IDXS0 = np.concatenate((np.array([502, 504, 506, 508, 510]), EYES_IDXS0))
    RIGHT_POSE_IDXS0 = np.concatenate((np.array([503, 505, 507, 509, 511]) , EYES_IDXS0))
    LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((LIPS_IDXS0, LEFT_HAND_IDXS0, LEFT_POSE_IDXS0))
    LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((LIPS_IDXS0, RIGHT_HAND_IDXS0, RIGHT_POSE_IDXS0))
    HAND_IDXS0 = np.concatenate((LEFT_HAND_IDXS0, RIGHT_HAND_IDXS0), axis=0)
    N_COLS = LANDMARK_IDXS_LEFT_DOMINANT0.size + 1 # added frame time feature

    # Landmark indices in processed data
    LIPS_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LIPS_IDXS0)).squeeze()
    LEFT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_HAND_IDXS0)).squeeze()
    RIGHT_HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, RIGHT_HAND_IDXS0)).squeeze()
    HAND_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, HAND_IDXS0)).squeeze()
    POSE_IDXS = np.argwhere(np.isin(LANDMARK_IDXS_LEFT_DOMINANT0, LEFT_POSE_IDXS0)).squeeze()

    # Start of IDxs
    LIPS_START = 0
    LEFT_HAND_START = LIPS_IDXS.size
    RIGHT_HAND_START = LEFT_HAND_START + LEFT_HAND_IDXS.size
    POSE_START = RIGHT_HAND_START + RIGHT_HAND_IDXS.size

    # Initiailizers
    INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
    INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
    INIT_ZEROS = tf.keras.initializers.constant(0.0)
    # Activations
    GELU = tf.keras.activations.gelu

def set_specific_gpu(ID):
    gpus = tf.config.list_physical_devices(device_type='GPU')    
    tf.config.set_visible_devices(gpus[ID], 'GPU')