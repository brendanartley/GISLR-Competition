import socket, json
import numpy as np
import tensorflow as tf
import wandb

class RUN_CFG:
    def __init__(self, file):
        if socket.gethostname() == 'gpu1':
            self.LOG_DATA_DIR = "/data/bartley/gpu_test/"
            self.MY_DATA_DIR = self.LOG_DATA_DIR + "my-gislr-data/"
            self.COMP_DATA_DIR = self.LOG_DATA_DIR + "asl-signs/"
            self.MW_DATA_DIR = self.LOG_DATA_DIR + file + "/"
            self.WEIGHTS_DIR = self.LOG_DATA_DIR + "saved_weights/"

            with open("config.json", "r+") as f:
                wandb_key = json.load(f)['wandb_key']
        else:
            self.LOG_DATA_DIR = '/kaggle/working/'
            self.MY_DATA_DIR = '/kaggle/input/my-gislr-data/'
            self.COMP_DATA_DIR = "/kaggle/input/asl-signs/"
            self.MW_DATA_DIR = "/kaggle/input/" + file
            self.WEIGHTS_DIR = "/kaggle/input/my-gislr-data/"

            from kaggle_secrets import UserSecretsClient # type: ignore
            wandb_key = UserSecretsClient().get_secret("wandb") # type: ignore

        wandb.login(key=wandb_key)

        self.MODEL_PATH = self.LOG_DATA_DIR + "mymodel.h5"
        self.INFER_PATH = self.LOG_DATA_DIR + "tf_model_infer"
        self.TFLITE_PATH_16 = self.LOG_DATA_DIR + "model_fp16.tflite"
        self.TFLITE_PATH_32 = self.LOG_DATA_DIR + "model_fp32.tflite"
        
        # Used in Kaggle env
        self.SUBMIT = True

        # # ---------- Model Config ----------
        self.N_ROWS = 543
        self.N_DIMS = 2

        if file in ["gislr-mw-24"]:
            self.INPUT_SIZE=24
            # option: V26 (85 landmarks)
            self.LIPS_IDXS0 = np.array([
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                ])

            self.EYES_IDXS0 = np.array([33,159,133,145,362,386,263,374,1,2,98,327])

            # Landmark indices in original data
            self.LEFT_HAND_IDXS0 = np.arange(468,489)
            self.RIGHT_HAND_IDXS0 = np.arange(522,543)
            self.LEFT_POSE_IDXS0 = np.concatenate((np.array([489, 491, 494, 496, 497, 500, 501, 502, 504, 506, 508, 510]), self.EYES_IDXS0))
            self.RIGHT_POSE_IDXS0 = np.concatenate((np.array([489, 491, 494, 496, 497, 500, 501, 503, 505, 507, 509, 511]) , self.EYES_IDXS0))
            self.LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((self.LIPS_IDXS0, self.LEFT_HAND_IDXS0, self.LEFT_POSE_IDXS0))
            self.LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((self.LIPS_IDXS0, self.RIGHT_HAND_IDXS0, self.RIGHT_POSE_IDXS0))
            self.HAND_IDXS0 = np.concatenate((self.LEFT_HAND_IDXS0, self.RIGHT_HAND_IDXS0), axis=0)
            self.N_COLS = self.LANDMARK_IDXS_LEFT_DOMINANT0.size

            # Landmark indices in processed data
            self.LIPS_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LIPS_IDXS0)).squeeze()
            self.LEFT_HAND_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_HAND_IDXS0)).squeeze()
            self.RIGHT_HAND_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.RIGHT_HAND_IDXS0)).squeeze()
            self.HAND_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.HAND_IDXS0)).squeeze()
            self.POSE_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_POSE_IDXS0)).squeeze()

            self.LIPS_START = 0
            self.LEFT_HAND_START = self.LIPS_IDXS.size
            self.RIGHT_HAND_START = self.LEFT_HAND_START + self.LEFT_HAND_IDXS.size
            self.POSE_START = self.RIGHT_HAND_START + self.RIGHT_HAND_IDXS.size

        elif file in ["gislr-24", "gislr-16"]:
            if file == "gislr-24":
                self.INPUT_SIZE=24
            elif file == "gislr-16":
                self.INPUT_SIZE=16

            # option: V26 (85 landmarks)
            self.LEFT_LIPS_IDXS0 = np.array([
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                ])
            self.RIGHT_LIPS_IDXS0 = np.array([
                    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
                    61, 375, 321, 405, 314, 17, 84, 181, 91, 146,
                    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
                    324, 318, 402, 317, 14, 87, 178, 88, 95, 78
                ])

            self.LEFT_EYES_IDXS0 = np.array([33,159,133,145,263,386,362,374,1,2,98,327])
            self.RIGHT_EYES_IDXS0 = np.array([263,386,362,374,33,159,133,145,1,2,327,98])

            # Landmark indices in original data
            self.LEFT_HAND_IDXS0 = np.arange(468,489)
            self.RIGHT_HAND_IDXS0 = np.arange(522,543)
            self.LEFT_POSE_IDXS0 = np.concatenate((np.array([500, 501, 502, 504, 506, 508, 510, 489, 491, 494, 496, 497]), self.LEFT_EYES_IDXS0))
            self.RIGHT_POSE_IDXS0 = np.concatenate((np.array([500, 501, 503, 505, 507, 509, 511, 489, 494, 491, 497, 496]) , self.RIGHT_EYES_IDXS0))
            self.LANDMARK_IDXS_LEFT_DOMINANT0 = np.concatenate((self.LEFT_LIPS_IDXS0, self.LEFT_HAND_IDXS0, self.LEFT_POSE_IDXS0))
            self.LANDMARK_IDXS_RIGHT_DOMINANT0 = np.concatenate((self.RIGHT_LIPS_IDXS0, self.RIGHT_HAND_IDXS0, self.RIGHT_POSE_IDXS0))
            self.HAND_IDXS0 = np.concatenate((self.LEFT_HAND_IDXS0, self.RIGHT_HAND_IDXS0), axis=0)
            self.N_COLS = self.LANDMARK_IDXS_LEFT_DOMINANT0.size

            # Landmark indices in processed data
            self.LIPS_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_LIPS_IDXS0)).squeeze()
            self.LEFT_HAND_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_HAND_IDXS0)).squeeze()
            self.RIGHT_HAND_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.RIGHT_HAND_IDXS0)).squeeze()
            self.HAND_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.HAND_IDXS0)).squeeze()
            self.POSE_IDXS = np.argwhere(np.isin(self.LANDMARK_IDXS_LEFT_DOMINANT0, self.LEFT_POSE_IDXS0)).squeeze()

            self.LIPS_START = 0
            self.LEFT_HAND_START = self.LIPS_IDXS.size
            self.RIGHT_HAND_START = self.LEFT_HAND_START + self.LEFT_HAND_IDXS.size
            self.POSE_START = self.RIGHT_HAND_START + self.RIGHT_HAND_IDXS.size