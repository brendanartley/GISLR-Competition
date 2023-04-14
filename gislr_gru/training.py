from gislr_gru.helpers import *
from gislr_gru.classes import *
from gislr_gru.models import *

import wandb
import gc, os
import numpy as np
import pandas as pd

def train(
        max_epochs: int = 150,
        val_fold: int = 2,
        cross_validate: bool = False,
        train_all: bool = False,
        batch_size: int = 1024,
        device: int = 0,
        drop_rate: float = 0.4,
        learning_rate: float = 3e-4,
        seed: int = 0,
        model_name: str = "baseline",
        in_shape: tuple = (15, 252),
        patience: int = 15,
        min_delta: float = 0.001,
        project: str = "gislr_test", #GISLR-keras, gislr_test
        verbose: int = 1, # 0,1,2 -> silent,progress,epoch
):
    # ----- Misc -----
    set_seeds(seed=seed)
    wandb.init(project=project)
    
    # ----- Build Model -----
    if model_name == 'baseline':
        model = get_gru_baseline(
            in_shape=in_shape,
            learning_rate=learning_rate,
        )

    # ----- Get Data -----
    meta_df = pd.read_csv(CFG.OTS_DATA_DIR + 'train.csv')
    ds = CompleteDataset(
        df=meta_df,
        train_all=train_all,
        val_fold=val_fold,
    )

    # ----- Train Model -----
    if train_all == True:
        model.fit(
            x=ds.train_x, 
            y=ds.train_y,
            batch_size=batch_size, 
            epochs=max_epochs,
            validation_freq=1, # run validation at the end of every epoch
            callbacks=[
                wandb.keras.WandbMetricsLogger(log_freq="epoch"),
            ],
            verbose=verbose,
        )
    elif cross_validate == False:
        model.fit(
            x=ds.train_x, 
            y=ds.train_y,
            validation_data=(ds.val_x, ds.val_y),
            batch_size=batch_size, 
            epochs=max_epochs,
            validation_freq=1, # run validation at the end of every epoch
            callbacks=[
                wandb.keras.WandbMetricsLogger(log_freq="epoch"),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    restore_best_weights=True, 
                    min_delta=min_delta, 
                    patience=patience)
            ],
            verbose=verbose,
        )

    elif cross_validate == True:
        val_accs = []
        val_losses = []
        for i in range(7):
            ds = CompleteDataset(
                df=meta_df,
                train_all=train_all,
                val_fold=val_fold,
            )
            history = model.fit(
                x=ds.train_x, 
                y=ds.train_y,
                validation_data=(ds.val_x, ds.val_y),
                batch_size=batch_size, 
                epochs=max_epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', 
                        restore_best_weights=True, 
                        min_delta=min_delta, 
                        patience=patience)
                ],
                verbose=verbose,
            )
            val_accs.append(history.history['val_accuracy'][np.argmin(history.history['val_loss'])])
            val_losses.append(np.min(history.history['val_loss']))

            # Log Acc when val_loss is minimized
            wandb.log({"f{}-Val-Loss".format(i): val_losses[-1]}, commit=False)
            wandb.log({"f{}-Val-Acc".format(i): val_accs[-1]}, commit=True)
            gc.collect()
        
        # Log overall stats
        wandb.log({"Val-Loss".format(i): np.mean(val_losses)}, commit=False)
        wandb.log({"Val-Acc".format(i): np.mean(val_accs)}, commit=True)
    return