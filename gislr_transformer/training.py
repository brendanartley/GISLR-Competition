import wandb

from gislr_transformer.callbacks import *
from gislr_transformer.models import *
from gislr_transformer.helpers import *

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
                augment=config.augment,
                augment_ratio=config.augment_ratio,
                augment_degrees=config.augment_degrees,
                augment_sampling=config.augment_sampling,
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