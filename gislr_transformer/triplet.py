import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
import wandb

from gislr_transformer.config import *
from gislr_transformer.helpers import *
from gislr_transformer.models import Embedding, Transformer
from gislr_transformer.callbacks import *

def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    alpha, mult = 1, 1e5
    anchor, positive, negative = inputs
    positive_distance = tf.square(anchor - positive)
    negative_distance = tf.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = tf.sqrt(tf.reduce_sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = tf.sqrt(tf.reduce_sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = tf.reduce_sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = tf.reduce_sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = tf.maximum(0.0, alpha + loss*mult)
    elif margin == 'softplus':
        loss = tf.math.log(alpha + tf.math.exp(loss))
    loss = tf.reduce_mean(loss, axis=-2) # mean across all 64 frames
    loss = tf.math.l2_normalize(loss, axis=0) # L2 normalization
    return tf.reduce_mean(loss)*1e5

def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    alpha, mult = 1, 1e5
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, alpha + loss*mult)
    elif margin == 'softplus':
        loss = np.log(alpha + np.exp(loss))
    return np.mean(loss)

def get_validation_data(X, NON_EMPTY_FRAME_IDXS, meta_df, train_all, val_fold):
    
    if train_all == True:
        return None
    
    val_size = len(meta_df[meta_df.fold == val_fold])

    # Arrays to store validation_data
    X_val = np.zeros([val_size, CFG.INPUT_SIZE*3, CFG.N_COLS, CFG.N_DIMS], dtype=np.float32)
    non_empty_frame_idxs_val = np.zeros([val_size, CFG.INPUT_SIZE*3], dtype=np.float32)

    # Sample Maps
    P_MAP = np.load(CFG.TRIPLET_DATA + 'pos_map.npy')
    N_MAP = np.load(CFG.TRIPLET_DATA + 'neg_map.npy')

    # Triplet Mapping
    for i, anchor_idx in enumerate(meta_df.index[(meta_df.fold == val_fold)].values):

        positive_idx = P_MAP[anchor_idx, 0] # Selecting hardest for now
        neg_idx = N_MAP[anchor_idx, 0]

        X_val[i, :CFG.INPUT_SIZE] = X[anchor_idx]
        X_val[i, CFG.INPUT_SIZE:CFG.INPUT_SIZE*2] = X[positive_idx]
        X_val[i, CFG.INPUT_SIZE*2:CFG.INPUT_SIZE*3] = X[neg_idx]

        # Anchor mask was used in triplet_mining
        non_empty_frame_idxs_val[i, :CFG.INPUT_SIZE] = NON_EMPTY_FRAME_IDXS[anchor_idx]
        non_empty_frame_idxs_val[i, CFG.INPUT_SIZE:CFG.INPUT_SIZE*2] = NON_EMPTY_FRAME_IDXS[anchor_idx]
        non_empty_frame_idxs_val[i, CFG.INPUT_SIZE*2:CFG.INPUT_SIZE*3] = NON_EMPTY_FRAME_IDXS[anchor_idx]

    return {'frames': X_val, 'non_empty_frame_idxs': non_empty_frame_idxs_val}


# Custom sampler to get a batch containing N times all signs
def triplet_get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, n, num_classes, meta_df, train_all, val_fold):
    
    # Arrays to store batch in
    X_batch = np.zeros([num_classes*n, CFG.INPUT_SIZE*3, CFG.N_COLS, CFG.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, num_classes, step=1/(n), dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([num_classes*n, CFG.INPUT_SIZE*3], dtype=np.float32)
    
    # Sample Maps
    P_MAP = np.load(CFG.TRIPLET_DATA + 'pos_map.npy')
    N_MAP = np.load(CFG.TRIPLET_DATA + 'neg_map.npy')

    # Triplet Mapping
    if train_all == False:
        # removes OOF
        CLASS2IDXS = {}
        for value in range(num_classes):
            indices = meta_df.index[(meta_df['label'] == value) & (meta_df.fold != val_fold)].tolist()
            CLASS2IDXS[value] = indices
    else:
        # Select all
        CLASS2IDXS = {}
        for i in range(num_classes):
            CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)

    while True:

        # Select Anchors
        anchor_idxs = np.zeros([num_classes*n], dtype=np.int32)
        for i in range(num_classes):
            anchor_idxs[i*n:(i+1)*n] = np.random.choice(CLASS2IDXS[i], n)
            
        # Fill batch arrays
        for i in range(num_classes*n):

            # Positive
            mask = np.isin(P_MAP[anchor_idxs[i]], anchor_idxs)
            mask = np.where(mask == True)[0]
            if mask.size != 0:
                positive_idx = P_MAP[anchor_idxs[i], mask[0]]
            else:
                positive_idx = P_MAP[anchor_idxs[i], 0]
                
            # Neg
            mask = np.isin(N_MAP[anchor_idxs[i]], anchor_idxs)
            mask = np.where(mask == True)[0]
            if mask.size != 0:
                neg_idx = N_MAP[anchor_idxs[i], mask[0]]
            else:
                neg_idx = N_MAP[anchor_idxs[i], 0]

            X_batch[i, :CFG.INPUT_SIZE] = X[anchor_idxs[i]]
            X_batch[i, CFG.INPUT_SIZE:CFG.INPUT_SIZE*2] = X[positive_idx]
            X_batch[i, CFG.INPUT_SIZE*2:CFG.INPUT_SIZE*3] = X[neg_idx]

            non_empty_frame_idxs_batch[i, :CFG.INPUT_SIZE] = NON_EMPTY_FRAME_IDXS[anchor_idxs[i]]
            non_empty_frame_idxs_batch[i, CFG.INPUT_SIZE:CFG.INPUT_SIZE*2] = NON_EMPTY_FRAME_IDXS[anchor_idxs[i]]
            non_empty_frame_idxs_batch[i, CFG.INPUT_SIZE*2:CFG.INPUT_SIZE*3] = NON_EMPTY_FRAME_IDXS[anchor_idxs[i]]
            
            yield {'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch }, y_batch

def triplet_load_data():
    meta_df = pd.read_csv(CFG.MY_DATA_DIR + "train.csv")
    X_train = np.load(CFG.MW_DATA_DIR + 'X.npy')
    y_train = np.load(CFG.MW_DATA_DIR + 'y.npy')
    NON_EMPTY_FRAME_IDXS_TRAIN = np.load(CFG.MW_DATA_DIR + '/NON_EMPTY_FRAME_IDXS.npy')
    return X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, meta_df

def get_triplet_model(
        units, 
        learning_rate,
        clip_norm,
        statsdict,
        ):
    # Inputs
    frames = tf.keras.layers.Input([CFG.INPUT_SIZE*3, CFG.N_COLS, CFG.N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([CFG.INPUT_SIZE*3], dtype=tf.float32, name='non_empty_frame_idxs')
    
    elayer = Embedding(units)
    
    # 3 Embeddings [anchor, pos, neg]
    # tf.slice(input, begin, size)
    x0 = tf.slice(frames, [0,CFG.INPUT_SIZE*0,0,0], [-1, CFG.INPUT_SIZE, 78, 2])
    x1 = tf.slice(frames, [0,CFG.INPUT_SIZE*1,0,0], [-1, CFG.INPUT_SIZE, 78, 2])
    x2 = tf.slice(frames, [0,CFG.INPUT_SIZE*2,0,0], [-1, CFG.INPUT_SIZE, 78, 2])
    non_empty_frame_idxs0 = tf.slice(non_empty_frame_idxs, [0,CFG.INPUT_SIZE*0], [-1, CFG.INPUT_SIZE])
    non_empty_frame_idxs1 = tf.slice(non_empty_frame_idxs, [0,CFG.INPUT_SIZE*1], [-1, CFG.INPUT_SIZE])
    non_empty_frame_idxs2 = tf.slice(non_empty_frame_idxs, [0,CFG.INPUT_SIZE*2], [-1, CFG.INPUT_SIZE])

    # Padding Mask
    mask0 = tf.expand_dims(tf.cast(tf.math.not_equal(non_empty_frame_idxs0, -1), tf.float32), axis=2)
    mask1 = tf.expand_dims(tf.cast(tf.math.not_equal(non_empty_frame_idxs1, -1), tf.float32), axis=2)
    mask2 = tf.expand_dims(tf.cast(tf.math.not_equal(non_empty_frame_idxs2, -1), tf.float32), axis=2)

    # LIPS
    lips0 = tf.slice(x0, [0,0,CFG.LIPS_START,0], [-1,CFG.INPUT_SIZE, 40, 2])
    lips0 = tf.where(tf.math.equal(lips0, 0.0), 0.0, (lips0 - statsdict["LIPS_MEAN"]) / statsdict["LIPS_STD"])
    lips1 = tf.slice(x1, [0,0,CFG.LIPS_START,0], [-1,CFG.INPUT_SIZE, 40, 2])
    lips1 = tf.where(tf.math.equal(lips1, 0.0), 0.0, (lips1 - statsdict["LIPS_MEAN"]) / statsdict["LIPS_STD"])
    lips2 = tf.slice(x2, [0,0,CFG.LIPS_START,0], [-1,CFG.INPUT_SIZE, 40, 2])
    lips2 = tf.where(tf.math.equal(lips2, 0.0), 0.0, (lips2 - statsdict["LIPS_MEAN"]) / statsdict["LIPS_STD"])    

    # LEFT HAND
    left_hand0 = tf.slice(x0, [0,0,CFG.LEFT_HAND_START,0], [-1,CFG.INPUT_SIZE, 21, 2])
    left_hand0 = tf.where(tf.math.equal(left_hand0, 0.0), 0.0, (left_hand0 - statsdict["LEFT_HANDS_MEAN"]) / statsdict["LEFT_HANDS_STD"])
    left_hand1 = tf.slice(x1, [0,0,CFG.LEFT_HAND_START,0], [-1,CFG.INPUT_SIZE, 21, 2])
    left_hand1 = tf.where(tf.math.equal(left_hand1, 0.0), 0.0, (left_hand1 - statsdict["LEFT_HANDS_MEAN"]) / statsdict["LEFT_HANDS_STD"])
    left_hand2 = tf.slice(x2, [0,0,CFG.LEFT_HAND_START,0], [-1,CFG.INPUT_SIZE, 21, 2])
    left_hand2 = tf.where(tf.math.equal(left_hand2, 0.0), 0.0, (left_hand2 - statsdict["LEFT_HANDS_MEAN"]) / statsdict["LEFT_HANDS_STD"])

    # POSE
    pose0 = tf.slice(x0, [0,0,CFG.POSE_START,0], [-1,CFG.INPUT_SIZE, 17, 2])
    pose0 = tf.where(tf.math.equal(pose0, 0.0),0.0,(pose0 - statsdict["POSE_MEAN"]) / statsdict["POSE_STD"])
    pose1 = tf.slice(x1, [0,0,CFG.POSE_START,0], [-1,CFG.INPUT_SIZE, 17, 2])
    pose1 = tf.where(tf.math.equal(pose1, 0.0),0.0,(pose1 - statsdict["POSE_MEAN"]) / statsdict["POSE_STD"])
    pose2 = tf.slice(x2, [0,0,CFG.POSE_START,0], [-1,CFG.INPUT_SIZE, 17, 2])
    pose2 = tf.where(tf.math.equal(pose2, 0.0),0.0,(pose2 - statsdict["POSE_MEAN"]) / statsdict["POSE_STD"])

    # Flatten
    lips0 = tf.reshape(lips0, [-1, CFG.INPUT_SIZE, 40*2])
    lips1 = tf.reshape(lips1, [-1, CFG.INPUT_SIZE, 40*2])
    lips2 = tf.reshape(lips2, [-1, CFG.INPUT_SIZE, 40*2])
    left_hand0 = tf.reshape(left_hand0, [-1, CFG.INPUT_SIZE, 21*2])
    left_hand1 = tf.reshape(left_hand1, [-1, CFG.INPUT_SIZE, 21*2])
    left_hand2 = tf.reshape(left_hand2, [-1, CFG.INPUT_SIZE, 21*2])
    pose0 = tf.reshape(pose0, [-1, CFG.INPUT_SIZE, 17*2])
    pose1 = tf.reshape(pose1, [-1, CFG.INPUT_SIZE, 17*2])
    pose2 = tf.reshape(pose2, [-1, CFG.INPUT_SIZE, 17*2])
    
    # Embedding
    x0 = elayer(lips0, left_hand0, pose0, non_empty_frame_idxs0)
    # Ignoring gradients for neg and pos sample
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        x1 = elayer(lips1, left_hand1, pose1, non_empty_frame_idxs1)
        x2 = elayer(lips2, left_hand2, pose2, non_empty_frame_idxs2)

    outputs = [x0, x1, x2]

    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    model.add_loss(triplet_loss(outputs))
    
    # Adam Optimizer with weight decay
    # weight_decay value is overidden by callback
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clip_norm, clipvalue=0.5, weight_decay=1e-5)
    
    model.compile(loss=None, optimizer=optimizer)
    return model

def get_triplet_weights(config, statsdict):
    # Get data
    print("-"*15 + " Triplet Training " + "-"*15)
    X_all, y_all, NON_EMPTY_FRAME_IDXS_ALL, meta_df = triplet_load_data()

    # Clear all models in GPU
    tf.keras.backend.clear_session()

    model = get_triplet_model(
        units=config.units, 
        learning_rate=config.triplet_learning_rate,
        clip_norm=config.clip_norm,
        statsdict=statsdict,
    )

    # # NOTE: FOR TESTING (DOESNT TRAIN TRIPLET WEIGHTS)
    # return model.get_layer(name='embedding').weights

    # Get callbacks
    callbacks = get_callbacks(
        model=model,
        epochs=config.triplet_epochs,
        warmup_epochs=config.warmup_epochs,
        lr_max=config.triplet_learning_rate,
        wd_ratio=config.weight_decay,
        do_early_stopping=False,
        no_wandb=True,
        min_delta=config.min_delta,
        patience=config.patience,
    )

    # Fix for triplet IDXs
    if config.train_all:
        steps_per_epoch = len(X_all) // (config.num_classes * config.batch_all_signs_n)
    else:
        steps_per_epoch = len(meta_df[meta_df.fold != config.val_fold]) // (config.num_classes * config.batch_all_signs_n)
    
    # Actual Training
    history=model.fit(
            x=triplet_get_train_batch_all_signs(
                X_all, 
                y_all, 
                NON_EMPTY_FRAME_IDXS_ALL, 
                n=config.batch_all_signs_n,
                num_classes=config.num_classes,
                meta_df=meta_df,
                train_all=config.train_all,
                val_fold=config.val_fold,
                ),
            steps_per_epoch=steps_per_epoch,
            epochs=config.triplet_epochs,
            # Only used for validation data since training data is a generator
            validation_data=get_validation_data(X_all, NON_EMPTY_FRAME_IDXS_ALL, meta_df, config.train_all, config.val_fold),
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=config.verbose,
        )
  
    if config.no_wandb == False:
        wandb.log({
            "Triplet_val_loss": history.history['val_loss'][-1],
            "Triplet_loss": history.history['loss'][-1]},
            commit=True)
    
    emb_layer = model.get_layer(name='embedding')
    return emb_layer

