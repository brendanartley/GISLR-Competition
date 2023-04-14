from gislr_lightning.classes import *
from gislr_lightning.helpers import *

def train(
    accelerator: str = "gpu",
    batch_size: int = 256,
    devices: int = 1,
    drop_rate: float = 0.4,
    eta_min: float = 1e-6,
    fast_dev_run: bool = False,
    first_out_features: int = 2048,
    in_features: int = 5796,
    learning_rate: float = 3e-4,
    loss: str = "CELoss",
    max_epochs: int = 200,
    model_name: str = "linear",
    num_blocks: int = 3,
    num_classes: int = 250,
    num_workers: int = 2,
    overfit_batches: int = 0,
    optimizer: str = "AdamW",
    patience: int = 20,
    precision: int = 16,
    project: str = "GISLR",
    scheduler: str = "CosineAnnealingLR",
    val_fold: float = 0,
    weight_decay: float = 1e-6,
):
    pl.seed_everything(0, workers=True)
    
    data_module = GislrDataModule(
        batch_size=batch_size,
        meta_data_path=CFG.MY_DATA_DIR + "train.csv",
        in_features=in_features,
        num_workers=num_workers,
        val_fold=val_fold,
    )
    
    module = GISLRModule(
        drop_rate=drop_rate,
        eta_min=eta_min,
        first_out_features=first_out_features,
        in_features=in_features,
        learning_rate=learning_rate,
        loss=loss,
        max_epochs=max_epochs,
        model_name=model_name,
        num_blocks=num_blocks,
        num_classes=num_classes,
        optimizer=optimizer,
        scheduler=scheduler,
        weight_decay=weight_decay,
    )
    
    logger, callbacks = load_logger_and_callbacks(
        fast_dev_run=fast_dev_run,
        metrics={"val_loss": "min", "val_acc": "max", "val_f1": "max"},
        overfit_batches=overfit_batches,
        patience=patience,
        project=project,
        val_fold=val_fold,
    )
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        benchmark=True,
        devices=devices,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run,
        logger=logger,
        log_every_n_steps=5,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        overfit_batches=overfit_batches,
        precision=precision,
        # strategy="ddp" if devices > 1 else None,
    )
    
    trainer.fit(module, datamodule=data_module)
    return module