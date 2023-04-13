import numpy as np
import pandas as pd

import torch
from torch import nn
import pytorch_lightning as pl
from timm.optim import create_optimizer_v2
import torchmetrics

from gislr_lightning.config import CFG

class GISLRDataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df, in_features, transform=None):
        self.df = df
        self.transform = transform

        print("Loading data...")
        self.X = np.load(CFG.OTS_DATA_DIR + "feature_data.npy")
        self.y = np.load(CFG.OTS_DATA_DIR + "feature_labels.npy")
        print(f"X shape: {self.X.shape}, y shape: {self.y.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use df_index as idx due to folds splitting
        df_index = self.df.index.values[idx]
        x = self.X[df_index]
        y = self.y[df_index]

        x = torch.Tensor(x)
        y = torch.Tensor([y]).long()

        if self.transform:
            x = self.transform(x)
        return x, y
    
class GislrDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        meta_data_path: str,
        in_features: int,
        num_workers: int,
        val_fold: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.df = pd.read_csv(meta_data_path)

        self.train_transform, self.val_transform = self._init_transforms()
    
    def _init_transforms(self):
        train_transform = None
        val_transform = None
        return train_transform, val_transform
    
    def setup(self, stage=None):
        val_fold = self.hparams.val_fold
        train_df = self.df[self.df.fold != val_fold] # DO NOT reset index here
        val_df = self.df[self.df.fold == val_fold]
        
        if stage == "fit" or stage == None:
            self.train_dataset = self._dataset(train_df, self.train_transform)
            self.val_dataset = self._dataset(val_df, self.val_transform)
            
    def _dataset(self, df, transform):
        # Custom Torch dataset class
        return GISLRDataFrameDataset(df, self.hparams.in_features, transform=transform)
    
    def train_dataloader(self):
        return self._dataloader(self.train_dataset, train=True)
    
    def val_dataloader(self):
        return self._dataloader(self.val_dataset)

    def _dataloader(self, dataset, train=False):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            drop_last=train, # Would train still work with drop_last == False
        )
    
class ASLLinearModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        first_out_features: int,
        num_classes: int,
        num_blocks: int,
        drop_rate: float,
    ):
        super().__init__()

        blocks = []
        out_features = first_out_features
        for idx in range(num_blocks):
            if idx == num_blocks - 1:
                out_features = num_classes

            blocks.append(self._make_block(in_features, out_features, drop_rate))

            in_features = out_features
            out_features = out_features // 2

        self.model = nn.Sequential(*blocks)
        print(self.model)

    def _make_block(self, in_features, out_features, drop_rate):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        return self.model(x)
    
class GISLRModule(pl.LightningModule):
    def __init__(
        self,
        drop_rate: float,
        eta_min: float, # min learning rate
        first_out_features: int,
        learning_rate: float,
        loss: str,
        in_features: int,
        max_epochs: int,
        model_name: str,
        num_blocks: int,
        num_classes: int,
        optimizer: str,
        scheduler: str,
        weight_decay: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = self._init_model()
        self.loss_fn = self._init_loss_fn()
        self.metrics, self.classwise_accuracy_metric = self._init_metrics()

    def _init_model(self):
        if self.hparams.model_name == "linear":
            return ASLLinearModel(
                in_features = self.hparams.in_features,
                first_out_features = self.hparams.first_out_features,
                num_classes = self.hparams.num_classes,
                num_blocks = self.hparams.num_blocks,
                drop_rate = self.hparams.drop_rate,
            )
        else:
            raise ValueError(f"{self.hparams.model_name} is not a valid model")
        
    def _init_loss_fn(self):
        if self.hparams.loss == "CELoss":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"{self.hparams.model_name} is not a valid loss function")
            
    def _init_metrics(self):
        # -- Scalar Value Metrics --
        metrics = {
            "acc": torchmetrics.classification.MulticlassAccuracy(
                num_classes = self.hparams.num_classes,
            ),
        }
        metric_collection = torchmetrics.MetricCollection(metrics)
        metric_collection = torch.nn.ModuleDict({
                "train_metrics": metric_collection.clone(prefix="train_"),
                "val_metrics": metric_collection.clone(prefix="val_")
            })

        # -- By Class Metric for Training End --
        classwise_accuracy_metric = torchmetrics.classification.MulticlassAccuracy(
            num_classes = self.hparams.num_classes,
            average = None,
        )
        return metric_collection, classwise_accuracy_metric
    
    def configure_optimizers(self):
        optimizer = self._init_optimizer()

        scheduler = self._init_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def _init_optimizer(self):
        return create_optimizer_v2(
            self.parameters(),
            opt = self.hparams.optimizer,
            lr = self.hparams.learning_rate,
            weight_decay = self.hparams.weight_decay,
        )
    
    def _init_scheduler(self, optimizer):
        if self.hparams.scheduler == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.eta_min,
            )
        elif self.hparams.scheduler == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.max_epochs // 5,
                gamma=0.95,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
        return scheduler
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        return self._shared_step(batch, "train")
    
    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")
    
    def _shared_step(self, batch, stage):
        x, y, y_hat = self._forward_pass(batch)
        loss = self.loss_fn(y_hat, y)
        
        self.metrics[f"{stage}_metrics"](y_hat, y)
        self._log(stage, loss, batch_size=len(x))
        if stage == "val":
            self.classwise_accuracy_metric(y_hat, y)
        return loss
    
    def _forward_pass(self, batch):
        x, y = batch
        y = y.view(-1) # flattens tensor
        y_hat = self(x)
        return x, y, y_hat
    
    def _log(self, stage, loss, batch_size):
        self.log(f"{stage}_loss", loss, batch_size=batch_size)
        self.log_dict(self.metrics[f"{stage}_metrics"], batch_size=batch_size) # log_dict - logs all metrics at once

    def on_train_end(self):
        accs = self.classwise_accuracy_metric.compute().tolist()
        self.logger.log_table(
            key = "class-accuracy", 
            columns = ["label", "val_acc"],
            data = [[i, x] for i, x in enumerate(accs)], 
            )