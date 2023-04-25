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
            self.MW_DATA_DIR = "/kaggle/input/gislr-mw-16/"
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
        self.INPUT_SIZE = 64

        if file in ["gislr-mw-16b", "gislr-mw-16f"]:
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

            # FOR ORIGINAL B
            # self.statsdict = {'POSE_MEAN': np.array([[0.4728136 , 0.4240536 ],[0.536576  , 0.373784  ],[0.40942368, 0.37736   ],[0.5997658 , 0.40010867],[0.35893154, 0.40480456],[0.7798274 , 0.64561284],[0.20726942, 0.6598391 ],[0.48128158, 0.89589083],[0.47540927, 0.68182313],[0.47280234, 0.6161903 ],[0.47247988, 0.5797547 ],[0.47407588, 0.6113952 ],[0.36968395, 0.3722532 ],[0.39401838, 0.3643502 ],[0.42822745, 0.37152776],[0.39657575, 0.37733042],[0.5022916 , 0.37017274],[0.5364219 , 0.3611874 ],[0.5629489 , 0.367665  ],[0.53461444, 0.37427425],[0.46230426, 0.4237136 ],[0.46400854, 0.43631318],[0.42751116, 0.4359091 ],[0.5033123 , 0.4342242 ]], dtype=np.float32), 'POSE_STD': np.array([[0.06185431, 0.07617903],[0.06488567, 0.07624143],[0.06678798, 0.07563455],[0.06733617, 0.07600936],[0.07485308, 0.07561885],[0.08024707, 0.07615038],[0.08732421, 0.07509337],[0.48750842, 0.0933991 ],[0.2785731 , 0.13016148],[0.23597713, 0.15089056],[0.20231341, 0.1394862 ],[0.19836532, 0.13098423],[0.07362793, 0.07461724],[0.07181355, 0.07450472],[0.06945913, 0.07406247],[0.07160134, 0.07419308],[0.0675202 , 0.07435886],[0.06730472, 0.07509894],[0.06703633, 0.07499121],[0.06676407, 0.07452413],[0.06717017, 0.07596151],[0.0665914 , 0.07586987],[0.06808676, 0.0755309 ],[0.06549962, 0.07555062]], dtype=np.float32), 'LEFT_HANDS_MEAN': np.array([[0.75869346, 0.66998696],[0.7097719 , 0.63058126],[0.65930927, 0.5900772 ],[0.618904  , 0.56465566],[0.59087056, 0.549599  ],[0.66726446, 0.544086  ],[0.6067027 , 0.51483697],[0.57830566, 0.5121041 ],[0.56221086, 0.51145816],[0.6791375 , 0.5559241 ],[0.6078051 , 0.5352497 ],[0.5863684 , 0.54267365],[0.5791031 , 0.5469511 ],[0.69115645, 0.576975  ],[0.62416214, 0.56305933],[0.60877424, 0.5725781 ],[0.6057572 , 0.577219  ],[0.7028756 , 0.603207  ],[0.65143657, 0.59304   ],[0.63851345, 0.5969334 ],[0.63490546, 0.59807736]], dtype=np.float32), 'LEFT_HANDS_STD': np.array([[0.10106158, 0.12571585],[0.10476863, 0.12234896],[0.1093974 , 0.12539777],[0.11426649, 0.13273634],[0.1234008 , 0.14140362],[0.11315253, 0.13402379],[0.12135054, 0.14970353],[0.12469217, 0.16204847],[0.12982582, 0.17308722],[0.1132836 , 0.14246114],[0.12367266, 0.16114126],[0.12492438, 0.17307244],[0.13037677, 0.18315522],[0.11868669, 0.15085693],[0.13010697, 0.16768062],[0.13008882, 0.17569257],[0.1342971 , 0.18288803],[0.12936541, 0.15792547],[0.14111309, 0.17060271],[0.14181338, 0.17685519],[0.14492628, 0.18267772]], dtype=np.float32), 'LIPS_MEAN': np.array([[0.41525415, 0.47939426],[0.41882986, 0.4750107 ],[0.4253797 , 0.47045675],[0.43502998, 0.4652801 ],[0.45024556, 0.4603687 ],[0.46643794, 0.4619916 ],[0.4826309 , 0.4595575 ],[0.49905086, 0.46370402],[0.50966954, 0.46836886],[0.5172951 , 0.47256923],[0.52186495, 0.47683564],[0.42011452, 0.48348522],[0.42698193, 0.48780724],[0.43791977, 0.49284703],[0.4521003 , 0.49624422],[0.46875396, 0.4970006 ],[0.48545423, 0.49564663],[0.49979982, 0.49166977],[0.5105381 , 0.48608747],[0.51730937, 0.4812727 ],[0.42040178, 0.47893593],[0.4277283 , 0.47716606],[0.43454722, 0.4756335 ],[0.4434654 , 0.47431198],[0.45452997, 0.47371727],[0.46734703, 0.4737115 ],[0.48040727, 0.47312412],[0.491821  , 0.47317424],[0.50132763, 0.47401685],[0.50891286, 0.4752054 ],[0.42795196, 0.47932437],[0.4346833 , 0.47945398],[0.44365668, 0.47955698],[0.4547623 , 0.47988176],[0.46778014, 0.48014253],[0.48093408, 0.47932917],[0.49227762, 0.4784758 ],[0.50165343, 0.47791994],[0.5087401 , 0.47744554],[0.5166792 , 0.47662103]], dtype=np.float32), 'LIPS_STD': np.array([[0.06906798, 0.07659707],[0.06854177, 0.07673516],[0.06809565, 0.07690708],[0.06770112, 0.07709396],[0.06728175, 0.07725324],[0.06666845, 0.07750451],[0.06607704, 0.07712494],[0.06544961, 0.07679018],[0.06496318, 0.07646379],[0.064566  , 0.07615448],[0.06415609, 0.07590489],[0.06858069, 0.07682684],[0.06807555, 0.07725308],[0.06744229, 0.07791283],[0.06674504, 0.07845675],[0.06603798, 0.07858105],[0.06536768, 0.07832868],[0.06486291, 0.07766014],[0.06449548, 0.07683925],[0.06428463, 0.0762542 ],[0.0688288 , 0.07666568],[0.06821401, 0.07687151],[0.06778651, 0.07708108],[0.06731777, 0.07731194],[0.06683893, 0.07751493],[0.06631962, 0.0776047 ],[0.06572415, 0.07737922],[0.065234  , 0.07702701],[0.06481469, 0.07666281],[0.0645657 , 0.07635052],[0.06820136, 0.07669736],[0.06777402, 0.07682188],[0.06728584, 0.07704714],[0.06678381, 0.07727893],[0.06626177, 0.07739697],[0.06568881, 0.0771628 ],[0.06520426, 0.07679975],[0.06479978, 0.07643656],[0.0645258 , 0.07619682],[0.06424969, 0.07603587]], dtype=np.float32)}