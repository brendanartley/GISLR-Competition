import numpy as np
import tensorflow as tf

from gislr_gru.config import CFG


class CompleteDataset:
    def __init__(self, df, train_all, val_fold):
        self.train_all = train_all
        
        # Load Data
        print("Loading Data.. ")
        X = np.load(CFG.OTS_DATA_DIR + "feature_data.npy")
        y = np.load(CFG.OTS_DATA_DIR + "feature_labels.npy")
        
        if self.train_all == False:
            self.train_df = df[df.fold!=val_fold]
            self.val_df = df[df.fold==val_fold]
            
            # Create Train/Val Sets
            self.train_x = X[self.train_df.index]
            self.train_y = y[self.train_df.index]
            self.val_x = X[self.val_df.index]
            self.val_y = y[self.val_df.index]
            
        else:
            self.train_df = df
            self.val_df = None
            
            self.train_x = X
            self.train_y = y
            self.val_x = np.zeros(0)
            self.val_y = np.zeros(0)
        
        print("Complete..")
        print(self.train_x.shape, self.train_y.shape, self.val_x.shape, self.val_y.shape)