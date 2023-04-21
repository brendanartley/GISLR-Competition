import math
import tensorflow as tf
import wandb

class WeightDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, wd_ratio):
        self.step_counter = 0
        self.wd_ratio = wd_ratio
        self.model = model
    
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.weight_decay = self.model.optimizer.learning_rate * self.wd_ratio
        print(f'learning rate: {self.model.optimizer.learning_rate.numpy():.2e}, weight decay: {self.model.optimizer.weight_decay.numpy():.2e}')

def get_callbacks(model, epochs, warmup_epochs, lr_max, wd_ratio, do_early_stopping, min_delta, patience, no_wandb):
    """
    lr, weight decay, earlystopping, wandblogger
    """
    callbacks = [
        get_lr_callback(epochs, warmup_epochs, lr_max),
        WeightDecayCallback(model=model, wd_ratio=wd_ratio),
    ]
    # Optional callbacks
    if do_early_stopping == True:
        callbacks.append(get_earlystopping(min_delta, patience))
    if no_wandb == False:
        callbacks.append(get_wandblogger())
    return callbacks

def lrfn(current_step, num_warmup_steps, lr_max, num_training_steps, num_cycles=0.50):
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return (max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr_max)
    
def get_lr_callback(epochs, warmup_epochs, lr_max):
    LR_SCHEDULE = [lrfn(step, num_warmup_steps=warmup_epochs, lr_max=lr_max,  num_training_steps=epochs, num_cycles=0.50) for step in range(epochs)]
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda step: LR_SCHEDULE[step], verbose=1)
    return lr_callback

def get_wandblogger():
    return wandb.keras.WandbMetricsLogger(log_freq="epoch")

def get_earlystopping(min_delta, patience):
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        restore_best_weights=True, 
        min_delta=min_delta, 
        patience=patience),
