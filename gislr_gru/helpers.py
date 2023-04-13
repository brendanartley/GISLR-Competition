import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import os

from gislr_gru.config import CFG

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

# ROWS_PER_FRAME = 543
# def load_relevant_data_subset(pq_path):
#     data_columns = ["x", "y", "z"]
#     data = pd.read_parquet(pq_path, columns=data_columns)
#     n_frames = int(len(data) / ROWS_PER_FRAME)
#     data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
#     return data.astype(np.float32)

# def convert_row(row, feature_converter, right_handed=True):
#     x = load_relevant_data_subset(os.path.join("/kaggle/input/asl-signs", row[1].path))
#     x = feature_converter(tf.convert_to_tensor(x)).cpu().numpy()
#     return x, row[1].label

# def convert_and_save_data(label_map, test=False):
#     df = pd.read_csv(CFG.COMP_DATA_DIR + 'train.csv')
#     df['label'] = df['sign'].map(label_map)
#     total = df.shape[0]
#     npdata = np.zeros((total, 15, 252))
#     nplabels = np.zeros(total)

#     for i, row in tqdm(enumerate(df.iterrows()), total=total):
#         (x,y) = convert_row(row)
#         npdata[i,:] = x
#         nplabels[i] = y

#     np.save("feature_data.npy", npdata)
#     np.save("feature_labels.npy", nplabels)