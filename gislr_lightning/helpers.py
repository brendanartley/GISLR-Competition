import pytorch_lightning as pl
from gislr_lightning.classes import CFG

def load_logger_and_callbacks(
    fast_dev_run,
    metrics,
    overfit_batches,
    patience,
    project,
    val_fold,
):
    """
    Function that loads logger and callbacks.
    
    Returns:
        logger: lighting logger
        callbacks: lightning callbacks
    """
    # Params used to check for Bugs/Errors in Implementation
    if fast_dev_run or overfit_batches > 0:
        logger, callbacks = None, None
    else:
        logger, id_ = get_logger(metrics=metrics, project=project)
        callbacks = get_callbacks(
            id_ = id_,
            mode = list(metrics.values())[0], # monitor first metric in dict
            monitor = list(metrics.values())[0],
            patience = patience,
            val_fold = val_fold,
        )
    return logger, callbacks

def get_logger(metrics, project):
    """
    Function to load logger.
    
    Returns:
        logger: lighting logger
        id_: experiment id
    """
    logger = pl.loggers.WandbLogger(project = project, save_dir = CFG.LOG_DATA_DIR)
    id_ = logger.experiment.id
    
    for metric, summary in metrics.items():
        logger.experiment.define_metric(metric, summary=summary)
    
    return logger, id_

def get_callbacks(id_, mode, monitor, patience, val_fold):
    callbacks = [
        pl.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        pl.callbacks.LearningRateMonitor(),
    ]