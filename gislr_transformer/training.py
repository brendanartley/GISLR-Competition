import wandb
import numpy as np
import time
import pickle

from gislr_transformer.callbacks import *
from gislr_transformer.models import *
from gislr_transformer.helpers import *
from gislr_transformer.config import *
from gislr_transformer.triplet import *

def train(config):

    # # Set specific GPU for training - For sweeps this is set via bash environment variables
    # set_specific_gpu(ID=config.device)

    # Init wandb/seed
    if config.no_wandb == False:
        wandb.init(project=config.project, config=config)
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get Landmark Statistics
    # statsdict = get_all_stats(X_train) # NOTE: Move below get_data to calc
    # statsdict = {'POSE_MEAN': np.array([[0.48128158, 0.89589083],[0.47540927, 0.68182313],[0.47280234, 0.6161903 ],[0.47247988, 0.5797547 ],[0.47407588, 0.6113952 ],[0.36968395, 0.3722532 ],[0.39401838, 0.3643502 ],[0.42822745, 0.37152776],[0.39657575, 0.37733042],[0.5022916 , 0.37017274],[0.5364219 , 0.3611874 ],[0.5629489 , 0.367665  ],[0.53461444, 0.37427425],[0.46230426, 0.4237136 ],[0.46400854, 0.43631318],[0.42751116, 0.4359091 ],[0.5033123 , 0.4342242 ]], dtype=np.float32), 'POSE_STD': np.array([[0.48750842, 0.0933991 ],[0.2785731 , 0.13016148],[0.23597713, 0.15089056],[0.20231341, 0.1394862 ],[0.19836532, 0.13098423],[0.07362793, 0.07461724],[0.07181355, 0.07450472],[0.06945913, 0.07406247],[0.07160134, 0.07419308],[0.0675202 , 0.07435886],[0.06730472, 0.07509894],[0.06703633, 0.07499121],[0.06676407, 0.07452413],[0.06717017, 0.07596151],[0.0665914 , 0.07586987],[0.06808676, 0.07553091],[0.06549963, 0.07555062]], dtype=np.float32), 'LEFT_HANDS_MEAN': np.array([[0.75869346, 0.66998696],[0.7097719 , 0.63058126],[0.65930927, 0.5900772 ],[0.618904  , 0.56465566],[0.59087056, 0.549599  ],[0.66726446, 0.544086  ],[0.6067027 , 0.51483697],[0.57830566, 0.5121041 ],[0.56221086, 0.51145816],[0.6791375 , 0.5559241 ],[0.6078051 , 0.5352497 ],[0.5863684 , 0.54267365],[0.5791031 , 0.5469511 ],[0.69115645, 0.576975  ],[0.62416214, 0.56305933],[0.60877424, 0.5725781 ],[0.6057572 , 0.577219  ],[0.7028756 , 0.603207  ],[0.65143657, 0.59304   ],[0.63851345, 0.5969334 ],[0.63490546, 0.59807736]], dtype=np.float32), 'LEFT_HANDS_STD': np.array([[0.10106158, 0.12571585],[0.10476863, 0.12234896],[0.1093974 , 0.12539777],[0.11426649, 0.13273634],[0.1234008 , 0.14140362],[0.11315253, 0.13402379],[0.12135054, 0.14970353],[0.12469217, 0.16204847],[0.12982582, 0.17308722],[0.1132836 , 0.14246114],[0.12367266, 0.16114126],[0.12492438, 0.17307244],[0.13037677, 0.18315522],[0.11868669, 0.15085693],[0.13010697, 0.16768062],[0.13008882, 0.17569257],[0.1342971 , 0.18288803],[0.12936541, 0.15792547],[0.14111309, 0.17060271],[0.14181338, 0.17685519],[0.14492628, 0.18267772]], dtype=np.float32), 'LIPS_MEAN': np.array([[0.41525415, 0.47939426],[0.41882986, 0.4750107 ],[0.4253797 , 0.47045675],[0.43502998, 0.4652801 ],[0.45024556, 0.4603687 ],[0.46643794, 0.4619916 ],[0.4826309 , 0.4595575 ],[0.49905086, 0.46370402],[0.50966954, 0.46836886],[0.5172951 , 0.47256923],[0.52186495, 0.47683564],[0.42011452, 0.48348522],[0.42698193, 0.48780724],[0.43791977, 0.49284703],[0.4521003 , 0.49624422],[0.46875396, 0.4970006 ],[0.48545423, 0.49564663],[0.49979982, 0.49166977],[0.5105381 , 0.48608747],[0.51730937, 0.4812727 ],[0.42040178, 0.47893593],[0.4277283 , 0.47716606],[0.43454722, 0.4756335 ],[0.4434654 , 0.47431198],[0.45452997, 0.47371727],[0.46734703, 0.4737115 ],[0.48040727, 0.47312412],[0.491821  , 0.47317424],[0.50132763, 0.47401685],[0.50891286, 0.4752054 ],[0.42795196, 0.47932437],[0.4346833 , 0.47945398],[0.44365668, 0.47955698],[0.4547623 , 0.47988176],[0.46778014, 0.48014253],[0.48093408, 0.47932917],[0.49227762, 0.4784758 ],[0.50165343, 0.47791994],[0.5087401 , 0.47744554],[0.5166792 , 0.47662103]], dtype=np.float32), 'LIPS_STD': np.array([[0.06906798, 0.07659707],[0.06854177, 0.07673516],[0.06809565, 0.07690708],[0.06770112, 0.07709396],[0.06728175, 0.07725324],[0.06666845, 0.07750451],[0.06607704, 0.07712494],[0.06544961, 0.07679018],[0.06496318, 0.07646379],[0.064566  , 0.07615448],[0.06415609, 0.07590489],[0.06858069, 0.07682684],[0.06807555, 0.07725308],[0.06744229, 0.07791283],[0.06674504, 0.07845675],[0.06603798, 0.07858105],[0.06536768, 0.07832868],[0.06486291, 0.07766014],[0.06449548, 0.07683925],[0.06428463, 0.0762542 ],[0.0688288 , 0.07666568],[0.06821401, 0.07687151],[0.06778651, 0.07708108],[0.06731777, 0.07731194],[0.06683893, 0.07751493],[0.06631962, 0.0776047 ],[0.06572415, 0.07737922],[0.065234  , 0.07702701],[0.06481469, 0.07666281],[0.0645657 , 0.07635052],[0.06820136, 0.07669736],[0.06777402, 0.07682188],[0.06728584, 0.07704714],[0.06678381, 0.07727893],[0.06626177, 0.07739697],[0.06568881, 0.0771628 ],[0.06520426, 0.07679975],[0.06479978, 0.07643656],[0.0645258 , 0.07619682],[0.06424969, 0.07603587]], dtype=np.float32)}    
    
    # Train Triplet weights
    if config.triplet and config.triplet_fname == "":
        triplet_embedding_layer = get_triplet_weights(
            config=config, 
            statsdict=statsdict
            )
        if wandb.run is None:
            emb_fname = str(int(time.time())) + '_embeddings.pkl'
        else:
            emb_fname = wandb.run.name + '_embeddings.pkl'
        config.triplet_fname = emb_fname
        
        # Save as pickle
        with open(CFG.WEIGHTS_DIR + config.triplet_fname, 'wb') as f:
            pickle.dump(triplet_embedding_layer.weights, f)

    # Only train triplet option
    if config.no_train:
        return

    # Get data
    print('-'*15 + " Classifier Training " + "-"*15)
    X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data = load_data(
        val_fold=config.val_fold,
        train_all=config.train_all,
    )

    #TODO: TEMP
    statsdict = get_all_stats(X_train) # NOTE: Move below get_data to calc

    # Clear all models in GPU
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get Model
    model = get_model(
        num_blocks=config.num_blocks,
        num_heads=config.num_heads, 
        units=config.units, 
        mlp_dropout_ratio=config.mlp_dropout_ratio, 
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        classifier_drop_rate=config.classifier_drop_rate,
        label_smoothing=config.label_smoothing,
        learning_rate=config.learning_rate,
        clip_norm=config.clip_norm,
        statsdict=statsdict,
    )

    # Set weights trained with triplet loss
    if config.triplet:
        # Set weights
        embedding_layer = model.get_layer(name='embedding')

        # Load weights from pickle
        with open(CFG.WEIGHTS_DIR + config.triplet_fname, 'rb') as f:
            triplet_emb_weights = pickle.load(f)

        for i, val in enumerate(triplet_emb_weights):
            embedding_layer.weights[i] = val

        print("Loaded embedding weights: {}.".format(config.triplet_fname))

        # Freeze weights
        embedding_layer.trainable=False
        print("Frozen embedding weights.")

    # Get callbacks
    callbacks = get_callbacks(
        model=model,
        epochs=config.max_epochs,
        warmup_epochs=config.warmup_epochs,
        lr_max=config.learning_rate,
        wd_ratio=config.weight_decay,
        do_early_stopping=config.do_early_stopping,
        min_delta=config.min_delta,
        patience=config.patience,
        no_wandb=config.no_wandb,
    )

    # Actual Training
    history=model.fit(
            x=get_train_batch_all_signs(
                X_train, 
                y_train, 
                NON_EMPTY_FRAME_IDXS_TRAIN, 
                n=config.batch_all_signs_n,
                num_classes=config.num_classes,
                ),
            steps_per_epoch=len(X_train) // (config.num_classes * config.batch_all_signs_n),
            epochs=config.max_epochs,
            # Only used for validation data since training data is a generator
            batch_size=config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=config.verbose,
        )

    # Log results w/ WandB
    log_classification_report(
        model=model,
        history=history,
        validation_data=validation_data,
        num_classes=config.num_classes,
        no_wandb=config.no_wandb,
    )
    print('Training complete.')