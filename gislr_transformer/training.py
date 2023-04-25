import wandb
import pickle, time

from gislr_transformer.callbacks import *
from gislr_transformer.models import *
from gislr_transformer.helpers import *
from gislr_transformer.pre_train import *

def train(config, CFG):

    # # Set specific GPU for training - For sweeps this is set via bash environment variables
    # set_specific_gpu(ID=config.device)

    # Init wandb/seed
    if config.no_wandb == False:
        wandb.init(project=config.project, config=config)
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get data
    print('-'*15 + " Classifier Training " + "-"*15)
    X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data = load_data(
        val_fold=config.val_fold,
        train_all=config.train_all,
        CFG=CFG,
    )

    # StatsDict: already computed
    CFG.statsdict = get_all_stats(X_train=X_train, CFG=CFG)
    print("statsdict sample:", CFG.statsdict['POSE_MEAN'][0])

    # Pre-train embeddings
    if config.emb_train == True:
        embedding_layer = pre_train_embeddings(
            config=config,
            CFG=CFG,
            X_train=X_train,
            y_train=y_train,
            NON_EMPTY_FRAME_IDXS_TRAIN=NON_EMPTY_FRAME_IDXS_TRAIN,
            validation_data=validation_data,
        )
        if wandb.run is None:
            emb_fname = str(int(time.time())) + '_embeddings.pkl'
        else:
            emb_fname = wandb.run.name + '_embeddings.pkl'
        config.emb_fname=emb_fname

        #TODO: Add embedding_fname, pre_train variables to config
        with open(CFG.WEIGHTS_DIR + config.emb_fname, 'wb') as f:
            pickle.dump(embedding_layer.weights, f)

    # Clear all models in GPU
    tf.keras.backend.clear_session()
    set_seeds(seed=config.seed)

    # Get Model
    model, loss, optimizer, metrics = get_model(
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

    # Set pre-trained embedding weights
    if config.emb_fname != "":

        # Select embedding layer
        embedding_layer = model.get_layer(name='embedding')

        # Load Weights
        with open(CFG.WEIGHTS_DIR + config.emb_fname, 'rb') as f:
            embedding_weights = pickle.load(f)

        # Set weights
        for i, val in enumerate(embedding_weights):
            embedding_layer.weights[i].assign(val)
        print("Loaded embedding weights: {}.".format(config.emb_fname))
    
        # Freeze weights
        embedding_layer.trainable=False
        print("Frozen embedding weights.")

    # Compile must come after freezing weights
    # Source: https://www.tensorflow.org/guide/keras/transfer_learning#fine-tuning
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    # Get callbacks
    callbacks = get_callbacks(
        model=model,
        epochs=config.max_epochs,
        warmup_epochs=config.warmup_epochs,
        lr_max=config.learning_rate,
        lr_decay=config.lr_decay,
        num_cycles=config.num_cycles,
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