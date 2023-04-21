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
            self.TRIPLET_DATA = self.LOG_DATA_DIR + "gislr-triplet-data/"

            with open("config.json", "r+") as f:
                wandb_key = json.load(f)['wandb_key']

        else:
            self.LOG_DATA_DIR = '/kaggle/working/'
            self.MY_DATA_DIR = '/kaggle/input/my-gislr-data/'
            self.COMP_DATA_DIR = "/kaggle/input/asl-signs/"
            self.MW_DATA_DIR = "/kaggle/input/gislr-mw-16/"
            self.WEIGHTS_DIR = "/kaggle/input/my-gislr-data/"

            from kaggle_secrets import UserSecretsClient # type: ignore
            wandb_key = UserSecretsClient().get_secret("wandb") # type: ignore
        
        wandb.login(key=wandb_key)
        
        self.MODEL_PATH = self.LOG_DATA_DIR + "mymodel.h5"
        self.INFER_PATH = self.LOG_DATA_DIR + "tf_model_infer"
        self.TFLITE_PATH = self.LOG_DATA_DIR + "model.tflite"
        
        # Used in Kaggle env
        self.SUBMIT = True

        # ---------- Data Config ----------
        self.N_ROWS = 543
        self.N_DIMS = 2
        self.INPUT_SIZE = 64

        # Dense layer units for landmarks
        self.LIPS_UNITS = 384
        self.HANDS_UNITS = 384
        self.POSE_UNITS = 384

        # Initiailizers
        self.INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
        self.INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
        self.INIT_ZEROS = tf.keras.initializers.constant(0.0)
        # Activations
        self.GELU = tf.keras.activations.gelu

        # OPTIONAL file PATH
        if file == "gislr-mw-16a":
            # Option: V20 (79 landmarks)
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
            self.LEFT_POSE_IDXS0 = np.concatenate((np.array([502, 504, 506, 508, 510]), self.EYES_IDXS0))
            self.RIGHT_POSE_IDXS0 = np.concatenate((np.array([503, 505, 507, 509, 511]) , self.EYES_IDXS0))
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

        elif file == "gislr-mw-16b":
            # optiion: V26 (85 landmarks)
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

        elif file == "gislr-mw-16c":

            self.LIPS_IDXS0 = np.array([
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
            ])

            self.REYE_IDXS0  = np.array([
                    33, 7, 163, 144, 145, 153, 154, 155, 133,
                    246, 161, 160, 159, 158, 157, 173,
                    ])
            self.LEYE_IDXS0  = np.array([
                263, 249, 390, 373, 374, 380, 381, 382, 362,
                466, 388, 387, 386, 385, 384, 398,
            ])    

            # Landmark indices in original data
            self.LEFT_HAND_IDXS0 = np.arange(468,489)
            self.RIGHT_HAND_IDXS0 = np.arange(522,543)
            self.LEFT_POSE_IDXS0 = np.concatenate((np.array([490, 491, 492, 496, 498, 500, 502, 504, 506, 508, 510, 512]), self.LEYE_IDXS0))
            self.RIGHT_POSE_IDXS0 = np.concatenate((np.array([493, 494, 495, 497, 499, 501, 503, 505, 507, 509, 511, 513]) , self.REYE_IDXS0))
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
    

def set_specific_gpu(ID):
    gpus = tf.config.list_physical_devices(device_type='GPU')    
    tf.config.set_visible_devices(gpus[ID], 'GPU')