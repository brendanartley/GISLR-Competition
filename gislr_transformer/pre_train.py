from gislr_transformer.models import *
from gislr_transformer.callbacks import *

def pre_train_embeddings(config, CFG, X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data):

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
    
    # Save as pickle
    embedding_layer = model.get_layer(name='embedding')
    return embedding_layer