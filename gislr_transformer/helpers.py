import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import wandb
from sklearn.metrics import classification_report as skl_cr

def set_seeds(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def load_data(val_fold, train_all, CFG):
    meta_df = pd.read_csv(CFG.MY_DATA_DIR + "train.csv")
    print(meta_df.shape)
    meta_df.head()

    X_train = np.load(CFG.MW_DATA_DIR + 'X.npy')
    y_train = np.load(CFG.MW_DATA_DIR + 'y.npy')
    NON_EMPTY_FRAME_IDXS_TRAIN = np.load(CFG.MW_DATA_DIR + '/NON_EMPTY_FRAME_IDXS.npy')

    if train_all == True:
        validation_data = None
    else:
        train_idx = meta_df[meta_df.fold != val_fold].index
        val_idx = meta_df[meta_df.fold == val_fold].index

        # Load Val
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        NON_EMPTY_FRAME_IDXS_VAL = NON_EMPTY_FRAME_IDXS_TRAIN[val_idx]

        # Define validation Data
        validation_data = ({'frames': X_val, 'non_empty_frame_idxs': NON_EMPTY_FRAME_IDXS_VAL }, y_val)

        # Load Val
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        NON_EMPTY_FRAME_IDXS_TRAIN = NON_EMPTY_FRAME_IDXS_TRAIN[train_idx]

    # Train 
    print_shape_dtype([X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN], ['X_train', 'y_train', 'NON_EMPTY_FRAME_IDXS_TRAIN'])
    # Val
    if train_all == False:
        print_shape_dtype([X_val, y_val, NON_EMPTY_FRAME_IDXS_VAL], ['X_val', 'y_val', 'NON_EMPTY_FRAME_IDXS_VAL'])
    # Sanity Check
    print(f'# NaN Values X_train: {np.isnan(X_train).sum()}')
    return X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data

# Prints Shape and Dtype For List Of Variables
def print_shape_dtype(l, names):
    for e, n in zip(l, names):
        print(f'{n} shape: {e.shape}, dtype: {e.dtype}')


def get_all_stats(X_train, CFG):
    POSE_MEAN, POSE_STD = get_pose_mean_std(X_train, CFG)
    LEFT_HANDS_MEAN, LEFT_HANDS_STD = get_left_right_hand_mean_std(X_train, CFG)
    LIPS_MEAN, LIPS_STD = get_lips_mean_std(X_train, CFG)
    stats_dict = {
        "POSE_MEAN": POSE_MEAN,
        "POSE_STD": POSE_STD,
        "LEFT_HANDS_MEAN": LEFT_HANDS_MEAN,
        "LEFT_HANDS_STD": LEFT_HANDS_STD,
        "LIPS_MEAN": LIPS_MEAN,
        "LIPS_STD": LIPS_STD,
    }
    return stats_dict

def get_lips_mean_std(X_train, CFG):
    # LIPS
    LIPS_MEAN_X = np.zeros([CFG.LIPS_IDXS.size], dtype=np.float32)
    LIPS_MEAN_Y = np.zeros([CFG.LIPS_IDXS.size], dtype=np.float32)
    LIPS_STD_X = np.zeros([CFG.LIPS_IDXS.size], dtype=np.float32)
    LIPS_STD_Y = np.zeros([CFG.LIPS_IDXS.size], dtype=np.float32)

    for col, ll in enumerate(tqdm( np.transpose(X_train[:,:,CFG.LIPS_IDXS], [2,3,0,1]).reshape([CFG.LIPS_IDXS.size, CFG.N_DIMS, -1]) )):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0: # X
                LIPS_MEAN_X[col] = v.mean()
                LIPS_STD_X[col] = v.std()
            if dim == 1: # Y
                LIPS_MEAN_Y[col] = v.mean()
                LIPS_STD_Y[col] = v.std()

    LIPS_MEAN = np.array([LIPS_MEAN_X, LIPS_MEAN_Y]).T
    LIPS_STD = np.array([LIPS_STD_X, LIPS_STD_Y]).T
    
    return LIPS_MEAN, LIPS_STD

def get_left_right_hand_mean_std(X_train, CFG):
    # LEFT HAND
    LEFT_HANDS_MEAN_X = np.zeros([CFG.LEFT_HAND_IDXS.size], dtype=np.float32)
    LEFT_HANDS_MEAN_Y = np.zeros([CFG.LEFT_HAND_IDXS.size], dtype=np.float32)
    LEFT_HANDS_STD_X = np.zeros([CFG.LEFT_HAND_IDXS.size], dtype=np.float32)
    LEFT_HANDS_STD_Y = np.zeros([CFG.LEFT_HAND_IDXS.size], dtype=np.float32)

    for col, ll in enumerate(tqdm( np.transpose(X_train[:,:,CFG.LEFT_HAND_IDXS], [2,3,0,1]).reshape([CFG.LEFT_HAND_IDXS.size, CFG.N_DIMS, -1]) )):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0: # X
                LEFT_HANDS_MEAN_X[col] = v.mean()
                LEFT_HANDS_STD_X[col] = v.std()
            if dim == 1: # Y
                LEFT_HANDS_MEAN_Y[col] = v.mean()
                LEFT_HANDS_STD_Y[col] = v.std()

    LEFT_HANDS_MEAN = np.array([LEFT_HANDS_MEAN_X, LEFT_HANDS_MEAN_Y]).T
    LEFT_HANDS_STD = np.array([LEFT_HANDS_STD_X, LEFT_HANDS_STD_Y]).T
    
    return LEFT_HANDS_MEAN, LEFT_HANDS_STD

def get_pose_mean_std(X_train, CFG):
    # POSE
    POSE_MEAN_X = np.zeros([CFG.POSE_IDXS.size], dtype=np.float32)
    POSE_MEAN_Y = np.zeros([CFG.POSE_IDXS.size], dtype=np.float32)
    POSE_STD_X = np.zeros([CFG.POSE_IDXS.size], dtype=np.float32)
    POSE_STD_Y = np.zeros([CFG.POSE_IDXS.size], dtype=np.float32)

    for col, ll in enumerate(tqdm( np.transpose(X_train[:,:,CFG.POSE_IDXS], [2,3,0,1]).reshape([CFG.POSE_IDXS.size, CFG.N_DIMS, -1]) )):
        for dim, l in enumerate(ll):
            v = l[np.nonzero(l)]
            if dim == 0: # X
                POSE_MEAN_X[col] = v.mean()
                POSE_STD_X[col] = v.std()
            if dim == 1: # Y
                POSE_MEAN_Y[col] = v.mean()
                POSE_STD_Y[col] = v.std()

    POSE_MEAN = np.array([POSE_MEAN_X, POSE_MEAN_Y]).T
    POSE_STD = np.array([POSE_STD_X, POSE_STD_Y]).T
    
    return POSE_MEAN, POSE_STD

# Custom sampler to get a batch containing N times all signs
def get_train_batch_all_signs(
        X, 
        y, 
        NON_EMPTY_FRAME_IDXS, 
        n, 
        num_classes, 
        CFG, 
        aug, 
        aug_sampling,
        hand_aug_rotate_ratio, 
        hand_aug_rotate_degrees, 
        hand_aug_expand_ratio,
        hand_aug_expand_pct,
        lips_aug_rotate_ratio, 
        lips_aug_rotate_degrees, 
        lips_aug_expand_ratio,
        lips_aug_expand_pct,
        face_aug_rotate_ratio,
        face_aug_rotate_degrees,
        eyes_aug_shift_ratio,
        eyes_aug_shift,
        eyes_aug_expand_ratio,
        eyes_aug_expand_pct,
        ):
    # Arrays to store batch in
    X_batch = np.zeros([num_classes*n, CFG.INPUT_SIZE, CFG.N_COLS, CFG.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, num_classes, step=1/n, dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([num_classes*n, CFG.INPUT_SIZE], dtype=np.float32)
    
    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(num_classes):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
            
    while True:
        # Fill batch arrays
        for i in range(num_classes):
            idxs = np.random.choice(CLASS2IDXS[i], n)
            X_batch[i*n:(i+1)*n] = X[idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n] = NON_EMPTY_FRAME_IDXS[idxs]

            # Augmentation on the hands
            if aug == True:
                for j in range(i*n, (i+1)*n):
                    
                    ## ----- HANDS AUGMENTATION -----
                    # Random rotation
                    hand_rotate_val = np.random.random()
                    if hand_rotate_val < hand_aug_rotate_ratio:
                        X_batch[j] = rotate_sequence(X_batch[j], hand_aug_rotate_degrees, aug_sampling, part="hands")
                    
                    # Random expansion/compression
                    hand_expand_val = np.random.random()
                    if hand_expand_val < hand_aug_expand_ratio:
                        X_batch[j] = expand_sequence(X_batch[j], hand_aug_expand_pct, aug_sampling, part="hands")

                    ## ----- LIPS AUGMENTATION -----
                    # Random rotation
                    lips_rotate_val = np.random.random()
                    if lips_rotate_val < lips_aug_rotate_ratio:
                        X_batch[j] = rotate_sequence(X_batch[j], lips_aug_rotate_degrees, aug_sampling, part="lips")
                    
                    # Random expansion/compression
                    lips_expand_val = np.random.random()
                    if lips_expand_val < lips_aug_expand_ratio:
                        X_batch[j] = expand_sequence(X_batch[j], lips_aug_expand_pct, aug_sampling, part="lips")

                    ## ----- EYES AUGMENTATION -----
                    # Random rotation
                    eyes_shift_val = np.random.random()
                    if eyes_shift_val < eyes_aug_shift_ratio:
                        X_batch[j] = shift_lr_sequence(X_batch[j], eyes_aug_shift, aug_sampling, part="eyes")

                    eyes_expand_val = np.random.random()
                    if eyes_expand_val < eyes_aug_expand_ratio:
                        X_batch[j] = shift_lr_sequence(X_batch[j], eyes_aug_expand_pct, aug_sampling, part="eyes")

                    ## ----- FACE AUGMENTATION (ALL FACE) -----
                    # Random rotation
                    face_rotate_val = np.random.random()
                    if face_rotate_val < face_aug_rotate_ratio:
                        X_batch[j] = rotate_sequence(X_batch[j], face_aug_rotate_degrees, aug_sampling, part="face")

        yield { 'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch }, y_batch

def expand(points, origin=(0, 0), factor=1):
    o = np.atleast_2d(origin)
    p = np.atleast_2d(points)
    v = factor * (p - o)
    return (v + o).squeeze()

def rotate(points, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(points)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def move_points_away_from_point(points, origin, factor=0.01):
    displacements = points - origin
    distances = np.sqrt(np.sum(displacements**2, axis=1))
    unit_vectors = displacements / distances[:, np.newaxis]
    unit_vectors = np.nan_to_num(unit_vectors, 0)
    moved_points = points + (unit_vectors*factor)
    return moved_points

def rotate_sequence(X, aug_rotate_degrees, aug_sampling, part):
    # Random sample between range
    if aug_sampling == 'uniform':
        degrees = np.random.uniform(-aug_rotate_degrees, aug_rotate_degrees)
    # Sample w/ center around 0
    elif aug_sampling == 'gaussian':
        mu = 0
        sigma = aug_rotate_degrees/3 # 99.7% within 3 std deviations or [-aug_rotate_degrees, aug_rotate_degrees]
        degrees = np.random.normal(mu, sigma)
    else:
        raise ValueError("Incorrect sampling method (either uniform or gaussian).")

    # Hands
    if part == "hands":
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, 40:61, :] = rotate(X[i, 40:61, :], origin=X[i, 40, :], degrees=degrees)

    # Lips
    elif part == "lips":
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, :40, :] = rotate(X[i, :40, :], origin=X[i, 25, :], degrees=degrees)

    # Face (including Lips) NOTE: This is for the updated DATA
    elif part == "face":
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, :40, :] = rotate(X[i, :40, :], origin=X[i, 68, :], degrees=degrees)
                X[i, 68:, :] = rotate(X[i, 68:, :], origin=X[i, 68, :], degrees=degrees)

    else: raise ValueError("Not a valid body part")

    return X

def expand_sequence(X, aug_expand_pct, aug_sampling, part):
    # Random sample between range
    if aug_sampling == 'uniform':
        factor = np.random.uniform(1-aug_expand_pct, 1+aug_expand_pct)
    # Sample w/ center around 0
    elif aug_sampling == 'gaussian':
        mu = 0
        sigma = aug_expand_pct/3 # 99.7% within 3 std deviations or [-aug_rotate_degrees, aug_rotate_degrees]
        factor = np.random.normal(mu, sigma)
    else:
        raise ValueError("Incorrect sampling method (either uniform or gaussian).")

    if part == "hands":
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, 40:61, :] = expand(points=X[i, 40:61, :], origin=X[i, 40, :], factor=factor)

    elif part == "lips":
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, :40, :] = expand(X[i, :40, :], origin=X[i, 25, :], factor=factor)

    elif part == "eyes":
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, 68:, :] = move_points_away_from_point(X[i, 68:, :], origin=X[i, 15, :], factor=factor)

    else: raise ValueError("Not a valid body part")

    return X

def shift_lr_sequence(X, aug_shift, aug_sampling, part):
    # Random sample between range
    if aug_sampling == 'uniform':
        amount_lr = np.random.uniform(-aug_shift, aug_shift)
    # Sample w/ center around 0
    elif aug_sampling == 'gaussian':
        mu = 0
        sigma = aug_shift/3 # 99.7% within 3 std deviations or [-aug_rotate_degrees, aug_rotate_degrees]
        amount_lr = np.random.normal(mu, sigma)
    else:
        raise ValueError("Incorrect sampling method (either uniform or gaussian).")
    
    # Hands
    if part == 'eyes':
        for i in range(X.shape[0]):
            if (X[i, 40, :] == 0).all():
                break
            else:
                X[i, 73:77, 0] += amount_lr
                X[i, 77:81, 0] -= amount_lr
    
    else: raise ValueError("Not a valid body part")

    return X

def log_classification_report(model, history, validation_data, num_classes, no_wandb, CFG):
    if no_wandb == True:
        return
    
    # Log overall stats
    wandb.log({
        "VLoss": np.min(history.history['val_loss']),
        "VAcc": np.max(history.history['val_accuracy']),
        "T5_Vacc": history.history['val_top_5_acc'][np.argmax(history.history['val_accuracy'])],
        "T10_Vacc": history.history['val_top_10_acc'][np.argmax(history.history['val_accuracy'])],
    }, commit=False)

    # Make predictions on valid data
    y_val_pred = model.predict(validation_data[0], verbose=1).argmax(axis=1)
    y_val = validation_data[1]

    # Meta data
    meta_df = pd.read_csv(CFG.MY_DATA_DIR + "train.csv")

    # Get Label-Sign Maps
    ORD2SIGN = meta_df[['label', 'sign']].drop_duplicates().set_index('label')['sign'].to_dict()
    SIGN2ORD = meta_df[['sign', 'label']].drop_duplicates().set_index('sign')['label'].to_dict()

    # Get label names
    labels = [ORD2SIGN.get(i).replace(' ', '_') for i in range(num_classes)]

    # Classification report for all signs
    classification_report = skl_cr(
            y_val,
            y_val_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
    # Round Data for better readability
    classification_report = pd.DataFrame(classification_report).T
    classification_report = classification_report.round(2)
    classification_report = classification_report.astype({
            'support': np.uint16,
        })
    # Add signs
    classification_report['sign_ord'] = classification_report.index.map(SIGN2ORD.get).fillna(-1).astype(np.int16)

    # Sort on F1-score
    classification_report = pd.concat((
        classification_report.head(num_classes).sort_values('f1-score', ascending=False),
    ))
    
    # Log to Wandb
    classification_report=classification_report.reset_index().rename(columns={'index': 'label'})
    table = wandb.Table(dataframe=classification_report, columns=list(classification_report.columns))
    wandb.log({"classification_report": table}, commit=False)
    print('Logged classification report.')
    
    # ---------- Log raw predictions ---------- 
    df = pd.DataFrame({
        'true': np.vectorize(ORD2SIGN.get)(y_val),
        'pred': np.vectorize(ORD2SIGN.get)(y_val_pred),
    })

    table = wandb.Table(dataframe=df, columns=list(df.columns))
    wandb.log({"validation_predictions": table}, commit=True)
    print('Logged validation predictions.')
    return