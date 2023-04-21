import wandb
import numpy as np
import time
import pickle

from gislr_transformer.callbacks import *
from gislr_transformer.models import *
from gislr_transformer.helpers import *
from gislr_transformer.triplet import *

from gislr_transformer.config import RUN_CFG
from gislr_transformer.namespace import default_config
CFG = RUN_CFG(file=default_config.file)

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
    # statsdict = {'POSE_MEAN': np.array([[-4.2469689e-01,  1.9685563e-02],[-8.2846320e-01, -1.8719117e-01],[-4.4610986e-01, -2.9509419e-03],[ 3.4990938e+00, -2.6961926e-01],[ 1.7954150e+00, -1.2685601e-01],[ 1.3621850e+00, -7.9327062e-02],[ 1.1261504e+00, -9.2261508e-02],[ 1.1277575e+00, -9.1014482e-02],[-4.7001469e-01,  3.4201611e-02],[-4.7559217e-01,  3.3099849e-02],[-4.8793250e-01,  3.6580600e-02],[-4.7694829e-01,  3.8583126e-02],[-5.2552646e-01,  3.3656415e-02],[-5.4674816e-01,  3.0525660e-02],[-5.6143796e-01,  2.9423552e-02],[-5.4157466e-01,  3.3873301e-02],[-4.7050792e-01,  4.1657038e-02],[-4.7687134e-01,  4.0038593e-02],[-4.6484327e-01,  4.2470284e-02],[-5.0544226e-01,  3.6376033e-02]], dtype=np.float32), 'POSE_STD': np.array([[0.2694868 , 0.84901047],[2.5365689 , 1.5032262 ],[2.1292727 , 1.6387156 ],[1.079823  , 3.8626137 ],[0.67501795, 1.8359071 ],[0.75872785, 1.4709048 ],[0.68607724, 1.1983284 ],[0.63508135, 1.313114  ],[0.8320121 , 1.3137467 ],[0.6490803 , 1.3883748 ],[0.40855542, 1.3193169 ],[0.62928694, 1.26478   ],[0.3650274 , 1.3316368 ],[0.59673226, 1.4168916 ],[0.79510915, 1.3557605 ],[0.5819536 , 1.2928668 ],[0.27099854, 0.84218466],[0.25544205, 0.73168486],[0.412268  , 0.73332137],[0.36727083, 0.7479979 ]], dtype=np.float32), 'LEFT_HANDS_MEAN': np.array([[ 1.7636867 , -0.15413481],[ 1.3866236 , -0.13777296],[ 0.9924917 , -0.13112816],[ 0.67163825, -0.13669544],[ 0.44745353, -0.14297184],[ 1.0463611 , -0.11443031],[ 0.5621253 , -0.13247016],[ 0.3379121 , -0.14792511],[ 0.2125271 , -0.1590922 ],[ 1.1286608 , -0.1264144 ],[ 0.5561015 , -0.14665554],[ 0.39062616, -0.16207196],[ 0.33853763, -0.17368878],[ 1.2120829 , -0.14131294],[ 0.6737869 , -0.16069108],[ 0.55620307, -0.17216225],[ 0.53712696, -0.17980817],[ 1.2935036 , -0.15649651],[ 0.8785858 , -0.16848831],[ 0.77929455, -0.17302689],[ 0.754631  , -0.17529348]], dtype=np.float32), 'LEFT_HANDS_STD': np.array([[0.7102953 , 1.7318809 ],[0.7441402 , 1.415622  ],[0.7784534 , 1.1668953 ],[0.8181207 , 1.1041076 ],[0.9011545 , 1.1380155 ],[0.78031677, 1.0647936 ],[0.85211086, 1.2022802 ],[0.88670355, 1.3222059 ],[0.93664867, 1.4343975 ],[0.7708177 , 1.1552042 ],[0.8628156 , 1.2922746 ],[0.8784067 , 1.4068521 ],[0.9300908 , 1.5124283 ],[0.81058735, 1.2811633 ],[0.9063074 , 1.378376  ],[0.90886813, 1.4707701 ],[0.9493442 , 1.5542362 ],[0.89459914, 1.4420831 ],[0.9879081 , 1.4909658 ],[0.9928109 , 1.5588056 ],[1.0209457 , 1.6176544 ]], dtype=np.float32), 'LIPS_MEAN': np.array([[-0.4610736 ,  0.03977412],[-0.45939192,  0.04332713],[-0.45995882,  0.04652261],[-0.46310902,  0.04906869],[-0.47008356,  0.04966832],[-0.47741753,  0.04884699],[-0.4876453 ,  0.04579594],[-0.4992756 ,  0.0408153 ],[-0.5066582 ,  0.0359198 ],[-0.51192564,  0.03093616],[-0.5138945 ,  0.0262721 ],[-0.46101514,  0.03853977],[-0.4616283 ,  0.03790056],[-0.4637499 ,  0.03747473],[-0.46808317,  0.03668457],[-0.47465992,  0.03494386],[-0.48450598,  0.0328032 ],[-0.49435338,  0.03047679],[-0.5032053 ,  0.02818069],[-0.50952536,  0.02669997],[-0.4636373 ,  0.04021266],[-0.46467757,  0.04288124],[-0.46614254,  0.04477581],[-0.46861058,  0.04609859],[-0.47268313,  0.04631609],[-0.4779922 ,  0.04554776],[-0.4864158 ,  0.04324572],[-0.49458906,  0.04009613],[-0.5015794 ,  0.03635624],[-0.50746083,  0.03252267],[-0.46494046,  0.03757388],[-0.46627876,  0.03623826],[-0.46857545,  0.03510614],[-0.4723717 ,  0.03392679],[-0.47779828,  0.03270812],[-0.48627025,  0.03087266],[-0.4944053 ,  0.02934098],[-0.5014861 ,  0.02811049],[-0.5067303 ,  0.02749097],[-0.51124   ,  0.02788016]], dtype=np.float32), 'LIPS_STD': np.array([[0.49408177, 0.4375811 ],[0.469586  , 0.45752564],[0.42640996, 0.48238638],[0.36738202, 0.51454675],[0.2915363 , 0.5482084 ],[0.25110173, 0.5382221 ],[0.27044377, 0.55398   ],[0.34157655, 0.5249966 ],[0.40379015, 0.49470425],[0.45313415, 0.47005898],[0.48388705, 0.44854048],[0.46049055, 0.42394856],[0.41524816, 0.41525087],[0.34942663, 0.41208255],[0.2816546 , 0.41462773],[0.24768786, 0.41631225],[0.2778324 , 0.41506138],[0.3447301 , 0.41391686],[0.40858388, 0.41987404],[0.4527423 , 0.4318086 ],[0.4586965 , 0.43958634],[0.41013807, 0.44753325],[0.36803824, 0.45554212],[0.31886286, 0.4633094 ],[0.27181703, 0.46754235],[0.24696033, 0.46824738],[0.26009926, 0.47060108],[0.30185822, 0.46908742],[0.35131726, 0.4634521 ],[0.3971806 , 0.45666876],[0.4087654 , 0.4396416 ],[0.36716625, 0.44147456],[0.31777927, 0.44422463],[0.270894  , 0.44563356],[0.24661388, 0.4459104 ],[0.2616168 , 0.44822127],[0.3040959 , 0.4492262 ],[0.35324737, 0.44842783],[0.39615262, 0.4479967 ],[0.44821712, 0.44971973]], dtype=np.float32)}
    statsdict = None

    # Train Triplet weights
    if config.triplet and config.triplet_fname == "":
        triplet_embedding_layer = get_triplet_weights(
            config=config, 
            statsdict=statsdict,
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

    # TEMP: STATS CALC
    statsdict = get_all_stats(X_train)
    
    # Clear all models in GPU
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get Model
    loss, optimizer, metrics, model = get_model(
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

    # IMPORTANT: Compile model after freezing weights
    # Source: ttps://www.tensorflow.org/guide/keras/transfer_learning#fine-tuning
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

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