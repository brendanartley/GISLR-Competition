import wandb
import numpy as np
import time
import pickle

from gislr_transformer.callbacks import *
from gislr_transformer.models import *
from gislr_transformer.helpers import *
from gislr_transformer.triplet import *

def train(config, CFG):

    # # Set specific GPU for training - For sweeps this is set via bash environment variables
    # set_specific_gpu(ID=config.device)

    # Init wandb/seed
    if config.no_wandb == False:
        wandb.init(project=config.project, config=config)
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Train Triplet weights
    if config.triplet and config.triplet_fname == "":
        triplet_embedding_layer = get_triplet_weights(
            config=config, 
            CFG=CFG,
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
        CFG=CFG,
    )


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
        CFG=CFG,
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
                CFG=CFG,
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
        CFG=CFG,
    )
    print('Training complete.')