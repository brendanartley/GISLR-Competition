from gislr_transformer.config import RUN_CFG
from types import SimpleNamespace
import argparse

from gislr_transformer.training import train

# defaults
default_config = SimpleNamespace(
    # device=0, # for sweeps this is set via CUDA_VISIBLE_DEVICES
    file="gislr-mw-16b",
    num_blocks=2,
    num_heads=8,
    units=512,
    mlp_dropout_ratio=0.20,
    mlp_ratio=2,
    classifier_drop_rate=0.0,
    learning_rate=5e-3,
    clip_norm=1.0,
    weight_decay=0.06,
    warmup_epochs=0,
    max_epochs=100,
    batch_size=256,
    num_classes=250,
    label_smoothing=0.67,
    batch_all_signs_n=4,
    do_early_stopping=False,
    no_wandb=False,
    patience=25,
    min_delta=1e-3,
    project="GISLR-keras",
    val_fold=2,
    train_all=False,
    verbose=2,
    seed=0,
    no_train=False,
    lr_decay=True,
    num_cycles=4.5,
    aug=True,
    aug_rotate_ratio=0.15,
    aug_rotate_degrees=15.0,
    aug_expand_ratio=0.30,
    aug_expand_pct=0.075, # ex. 0.10 = [0.9 - 1.10]
    aug_sampling='uniform',
)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--device", type=int, default=default_config.device, help="GPU device number.")
    parser.add_argument("--file", type=str, default=default_config.file, help="GISLR data folder to train on.")
    parser.add_argument("--num_blocks", type=int, default=default_config.num_blocks, help="Number of transformer blocks in the model.")
    parser.add_argument("--num_heads", type=int, default=default_config.num_heads, help="Number of attention heads per transformer block.")
    parser.add_argument("--units", type=int, default=default_config.units, help="Final embedding size.")
    parser.add_argument("--mlp_dropout_ratio", type=float, default=default_config.mlp_dropout_ratio, help="Dropout ratio for MLP layers.")
    parser.add_argument("--mlp_ratio", type=int, default=default_config.mlp_ratio, help="MLP expansion ratio.")
    parser.add_argument("--classifier_drop_rate", type=float, default=default_config.classifier_drop_rate, help="Dropout rate for the classifier layer.")
    parser.add_argument("--learning_rate", type=float, default=default_config.learning_rate, help="Learning rate for the optimizer.")
    parser.add_argument("--lr_decay", type=bool, default=default_config.lr_decay, help="Bool to decay LR or not.")
    parser.add_argument("--num_cycles", type=float, default=default_config.num_cycles, help="Number of cycles for the learning rate scheduler.")
    parser.add_argument("--clip_norm", type=float, default=default_config.clip_norm, help="Maximum norm for gradient clipping.")
    parser.add_argument("--weight_decay", type=float, default=default_config.weight_decay, help="L2 regularization weight.")
    parser.add_argument("--warmup_epochs", type=int, default=default_config.warmup_epochs, help="Number of warmup epochs for learning rate schedule.")
    parser.add_argument("--max_epochs", type=int, default=default_config.max_epochs, help="Maximum number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=default_config.batch_size, help="Batch size for training and evaluation.")
    parser.add_argument("--num_classes", type=int, default=default_config.num_classes, help="Number of classes for classification task.")
    parser.add_argument("--label_smoothing", type=float, default=default_config.label_smoothing, help="Amount of label smoothing.")
    parser.add_argument("--batch_all_signs_n", type=int, default=default_config.batch_all_signs_n, help="Number of signs to include in each batch for training.")
    parser.add_argument("--do_early_stopping", action="store_true", help="Whether to use early stopping based on validation loss.")
    parser.add_argument("--patience", type=int, default=default_config.patience, help="Number of epochs to wait for improvement before early stopping.")
    parser.add_argument("--min_delta", type=float, default=default_config.min_delta, help="Minimum change in validation loss to be considered an improvement.")
    parser.add_argument("--no_wandb", action="store_true", help="Whether to use early stopping based on validation loss.")
    parser.add_argument("--project", type=str, default=default_config.project, help="Project name for wandb logging.")
    parser.add_argument("--val_fold", type=float, default=default_config.val_fold, help="Fold number to use for validation.")
    parser.add_argument("--train_all", action="store_true", help="Whether to train on all the data.")
    parser.add_argument("--verbose", type=int, default=default_config.verbose, help="Verbosity level (0 = silent, 1 = progress bar, 2 = one line per epoch).")
    parser.add_argument("--seed", type=int, default=default_config.seed, help="Seed for reproducability.")
    parser.add_argument("--no_train", action="store_true", help="Do training or not.")
    parser.add_argument("--aug", type=bool, default=default_config.aug, help="Do augmentation on hands.")
    parser.add_argument("--aug_rotate_ratio", type=float, default=default_config.aug_rotate_ratio, help="How often to randomly apply rotation.")
    parser.add_argument("--aug_rotate_degrees", type=float, default=default_config.aug_rotate_degrees, help="Degree range of rotation.")
    parser.add_argument("--aug_expand_ratio", type=float, default=default_config.aug_expand_ratio, help="How often to randomly apply expansion.")
    parser.add_argument("--aug_expand_pct", type=float, default=default_config.aug_expand_pct, help="How much to expand/compress hands (0.10 = 10%).")
    parser.add_argument("--aug_sampling", type=str, default=default_config.aug_sampling, help="GISLR data folder to train on.")
    args = parser.parse_args()
    return args

def main(config, CFG):
    module = train(config, CFG)
    pass

if __name__ == "__main__":
    config = parse_args()
    CFG = RUN_CFG(file=config.file)
    main(config, CFG)
