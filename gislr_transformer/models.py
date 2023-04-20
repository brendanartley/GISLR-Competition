import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from gislr_transformer.config import CFG
from gislr_transformer.helpers import *

class PreprocessLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PreprocessLayer, self).__init__()
        normalisation_correction = tf.constant([
                    # Add 0.50 to left hand (original right hand) and substract 0.50 of right hand (original left hand)
                    [0] * len(CFG.LIPS_IDXS) + [0.50] * len(CFG.LEFT_HAND_IDXS) + [0] * len(CFG.POSE_IDXS),
                    # Y coordinates stay intact
                    [0] * len(CFG.LANDMARK_IDXS_LEFT_DOMINANT0),
                ],
                dtype=tf.float32,
            )
        self.normalisation_correction = tf.transpose(normalisation_correction, [1,0])

    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return tf.concat((tf.repeat(t[:1], repeats=repeats, axis=0), t), axis=0)
        elif side == 'RIGHT':
            return tf.concat((t, tf.repeat(t[-1:], repeats=repeats, axis=0)), axis=0)

    @tf.function(
        input_signature=(tf.TensorSpec(shape=[None,543,3], dtype=tf.float32),),
    )
    def call(self, data0):
        # Drop Z
        data0 = data0[:, :, :2]

        # Number of Frames in Video
        N_FRAMES0 = tf.shape(data0)[0]

        # Find dominant hand by comparing summed absolute coordinates
        left_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(data0, CFG.LEFT_HAND_IDXS0, axis=1)), 0, 1))
        right_hand_sum = tf.math.reduce_sum(tf.where(tf.math.is_nan(tf.gather(data0, CFG.RIGHT_HAND_IDXS0, axis=1)), 0, 1))
        left_dominant = left_hand_sum >= right_hand_sum

        # Count non NaN Hand values in each frame for the dominant hand
        if left_dominant:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                    tf.where(tf.math.is_nan(tf.gather(data0, CFG.LEFT_HAND_IDXS0, axis=1)), 0, 1),
                    axis=[1, 2],
                )
        else:
            frames_hands_non_nan_sum = tf.math.reduce_sum(
                    tf.where(tf.math.is_nan(tf.gather(data0, CFG.RIGHT_HAND_IDXS0, axis=1)), 0, 1),
                    axis=[1, 2],
                )

        # Find frames indices with coordinates of dominant hand
        non_empty_frames_idxs = tf.where(frames_hands_non_nan_sum > 0)
        non_empty_frames_idxs = tf.squeeze(non_empty_frames_idxs, axis=1)

        # Filter frames
        data = tf.gather(data0, non_empty_frames_idxs, axis=0)

        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = tf.cast(non_empty_frames_idxs, tf.float32)
        # Normalize to start with 0
        non_empty_frames_idxs -= tf.reduce_min(non_empty_frames_idxs)

        # Number of Frames in Filtered Video
        N_FRAMES = tf.shape(data)[0]

        # Gather Relevant Landmark Columns
        if left_dominant:
            data = tf.gather(data, CFG.LANDMARK_IDXS_LEFT_DOMINANT0, axis=1)
        else:
            data = tf.gather(data, CFG.LANDMARK_IDXS_RIGHT_DOMINANT0, axis=1)
            data = (
                    self.normalisation_correction + (
                        (data - self.normalisation_correction) * tf.where(self.normalisation_correction != 0, -1.0, 1.0))
                )

        # Video fits in INPUT_SIZE
        if N_FRAMES < CFG.INPUT_SIZE:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = tf.pad(non_empty_frames_idxs, [[0, CFG.INPUT_SIZE-N_FRAMES]], constant_values=-1)
            # Pad Data With Zeros
            data = tf.pad(data, [[0, CFG.INPUT_SIZE-N_FRAMES], [0,0], [0,0]], constant_values=0)
            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < CFG.INPUT_SIZE**2:
                repeats = tf.math.floordiv(CFG.INPUT_SIZE * CFG.INPUT_SIZE, N_FRAMES0)
                data = tf.repeat(data, repeats=repeats, axis=0)
                non_empty_frames_idxs = tf.repeat(non_empty_frames_idxs, repeats=repeats, axis=0)
            # Pad To Multiple Of Input Size
            pool_size = tf.math.floordiv(len(data), CFG.INPUT_SIZE)
            if tf.math.mod(len(data), CFG.INPUT_SIZE) > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * CFG.INPUT_SIZE) - len(data)
            else:
                pad_size = (pool_size * CFG.INPUT_SIZE) % len(data)

            # Pad Start/End with Start/End value
            pad_left = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(CFG.INPUT_SIZE, 2)
            pad_right = tf.math.floordiv(pad_size, 2) + tf.math.floordiv(CFG.INPUT_SIZE, 2)
            if tf.math.mod(pad_size, 2) > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')

            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_left, 'LEFT')
            non_empty_frames_idxs = self.pad_edge(non_empty_frames_idxs, pad_right, 'RIGHT')

            # Reshape to Mean Pool
            data = tf.reshape(data, [CFG.INPUT_SIZE, -1, CFG.N_COLS, CFG.N_DIMS])
            non_empty_frames_idxs = tf.reshape(non_empty_frames_idxs, [CFG.INPUT_SIZE, -1])

            # Mean Pool
            data = tf.experimental.numpy.nanmean(data, axis=1)
            non_empty_frames_idxs = tf.experimental.numpy.nanmean(non_empty_frames_idxs, axis=1)

            # Fill NaN Values With 0
            data = tf.where(tf.math.is_nan(data), 0.0, data)
            return data, non_empty_frames_idxs
        
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Code: https://stackoverflow.com/a/67344134/14722297
    Dimension Explanation: https://datascience.stackexchange.com/a/93775/115596
    Illustrated Transformer: https://jalammar.github.io/illustrated-transformer/
    """
    def __init__(self, out_dim, num_of_heads):
        super(MultiHeadAttention,self).__init__()
        self.out_dim = out_dim
        self.num_of_heads = num_of_heads
        self.depth = out_dim//num_of_heads
        self.wq = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth) for i in range(num_of_heads)]
        self.wo = tf.keras.layers.Dense(self.out_dim)
        self.softmax = tf.keras.layers.Softmax()
        
    def call(self, x, attention_mask):
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)
            K = self.wk[i](x)
            V = self.wv[i](x)
            multi_attn.append(self.scaled_dot_product(Q,K,V, self.softmax, attention_mask))
            
        multi_head = tf.concat(multi_attn, axis=-1)
        multi_head_attention = self.wo(multi_head)
        return multi_head_attention

    def scaled_dot_product(self,q,k,v, softmax, attention_mask):
        """
        Softmax layer supports softmax w/ mask.
        
        Could modify to not use softmax layer: https://stackoverflow.com/a/65745327/14722297
        """
        # calculates Q . K(transpose)
        qkt = tf.matmul(q,k,transpose_b=True)
        # Scale factor
        dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=tf.float32))
        scaled_qkt = qkt/dk
        # Softmax * V
        softmax = softmax(scaled_qkt, mask=attention_mask)
        z = tf.matmul(softmax, v)
        # softmax(Q*K / dk) * V
        return z
    
# Full Transformer
class Transformer(tf.keras.Model):
    def __init__(self, num_blocks, num_heads, units, mlp_dropout_ratio, mlp_ratio):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.units = units
        self.mlp_dropout_ratio = mlp_dropout_ratio
        self.mlp_ratio = mlp_ratio
    
    def build(self, input_shape):
        self.ln_1s = []
        self.mhas = []
        self.ln_2s = []
        self.mlps = []
        # Make Transformer Blocks
        for i in range(self.num_blocks):
            # Multi Head Attention
            self.mhas.append(MultiHeadAttention(self.units, self.num_heads))
            # Multi Layer Perception
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(self.units * self.mlp_ratio, activation=CFG.GELU, kernel_initializer=CFG.INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(self.mlp_dropout_ratio),
                tf.keras.layers.Dense(self.units, kernel_initializer=CFG.INIT_HE_UNIFORM),
            ]))
        
    def call(self, x, attention_mask):
        # Iterate input over transformer blocks
        for mha, mlp in zip(self.mhas, self.mlps):
            x = x + mha(x, attention_mask)
            x = x + mlp(x)
        return x
    
class LandmarkEmbedding(tf.keras.Model):
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units
        
    def build(self, input_shape):
        # Embedding for missing landmark in frame, initizlied with zeros
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=CFG.INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=CFG.INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(CFG.GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=CFG.INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x):
        return tf.where(
                # Checks whether landmark is missing in frame
                tf.reduce_sum(x, axis=2, keepdims=True) == 0,
                # If so, the empty embedding is used
                self.empty_embedding,
                # Otherwise the landmark data is embedded
                self.dense(x),
            )
    
class Embedding(tf.keras.Model):
    def __init__(self, units):
        super(Embedding, self).__init__()
        self.units = units

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(CFG.INPUT_SIZE+1, self.units, embeddings_initializer=CFG.INIT_ZEROS)
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(CFG.LIPS_UNITS, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(CFG.HANDS_UNITS, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(CFG.POSE_UNITS, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable([0.01, 0.95, 0.01], dtype=tf.float32, name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name='fully_connected_1', use_bias=False, kernel_initializer=CFG.INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(CFG.GELU),
            tf.keras.layers.Dense(self.units, name='fully_connected_2', use_bias=False, kernel_initializer=CFG.INIT_HE_UNIFORM),
        ], name='fc')


    def call(self, lips0, left_hand0, pose0, non_empty_frame_idxs):
        # Lips
        lips_embedding = self.lips_embedding(lips0)
        # Left Hand
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        # Pose
        pose_embedding = self.pose_embedding(pose0)
        # Merge Embeddings of all landmarks with mean pooling
        x = tf.stack((
            lips_embedding, left_hand_embedding, pose_embedding,
        ), axis=3)
        x = x * tf.nn.softmax(self.landmark_weights)
        x = tf.reduce_sum(x, axis=3)
        # Fully Connected Layers
        x = self.fc(x)
        # Add Positional Embedding
        max_frame_idxs = tf.clip_by_value(
                tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True),
                1,
                np.PINF,
            )
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            CFG.INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / max_frame_idxs * CFG.INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        return x
    
def get_model(
        num_blocks,
        num_heads, 
        units, 
        mlp_dropout_ratio, 
        mlp_ratio,
        num_classes,
        classifier_drop_rate,
        label_smoothing,
        learning_rate,
        clip_norm,
        statsdict,
        ):
    # Inputs
    frames = tf.keras.layers.Input([CFG.INPUT_SIZE, CFG.N_COLS, CFG.N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([CFG.INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')
    
    # Padding Mask
    mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)
    mask = tf.expand_dims(mask, axis=2)

    """
        left_hand: 468:489
        pose: 489:522
        right_hand: 522:543
    """
    x = frames
    x = tf.slice(x, [0,0,0,0], [-1,CFG.INPUT_SIZE, CFG.N_COLS, 2])
    # LIPS
    lips = tf.slice(x, [0,0,CFG.LIPS_START,0], [-1,CFG.INPUT_SIZE, 40, 2])
    lips = tf.where(
            tf.math.equal(lips, 0.0),
            0.0,
            (lips - statsdict["LIPS_MEAN"]) / statsdict["LIPS_STD"],
        )
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,40,0], [-1,CFG.INPUT_SIZE, 21, 2])
    left_hand = tf.where(
            tf.math.equal(left_hand, 0.0),
            0.0,
            (left_hand - statsdict["LEFT_HANDS_MEAN"]) / statsdict["LEFT_HANDS_STD"],
        )
    # POSE
    pose = tf.slice(x, [0,0,61,0], [-1,CFG.INPUT_SIZE, 17, 2])
    pose = tf.where(
            tf.math.equal(pose, 0.0),
            0.0,
            (pose - statsdict["POSE_MEAN"]) / statsdict["POSE_STD"],
        )
    
    # Flatten
    lips = tf.reshape(lips, [-1, CFG.INPUT_SIZE, 40*2])
    left_hand = tf.reshape(left_hand, [-1, CFG.INPUT_SIZE, 21*2])
    pose = tf.reshape(pose, [-1, CFG.INPUT_SIZE, 17*2])
        
    # Embedding
    x = Embedding(units)(lips, left_hand, pose, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    x = Transformer(num_blocks, num_heads, units, mlp_dropout_ratio, mlp_ratio)(x, mask)
    
    # Pooling
    x = tf.reduce_sum(x * mask, axis=1) / tf.reduce_sum(mask, axis=1)
    # Classifier Dropout
    x = tf.keras.layers.Dropout(classifier_drop_rate)(x)
    # Classification Layer
    x = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax, kernel_initializer=CFG.INIT_GLOROT_UNIFORM)(x)
    
    outputs = x
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)

    # Sparse Categorical Cross Entropy With Label Smoothing
    # source:: https://stackoverflow.com/questions/60689175/label-smoothing-for-sparse-categorical-crossentropy
    def scce_with_ls(y_true, y_pred):
        # One Hot Encode Sparsely Encoded Target Sign
        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.one_hot(y_true, num_classes, axis=1)
        y_true = tf.squeeze(y_true, axis=2)
        # Categorical Crossentropy with native label smoothing support
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    loss = scce_with_ls
    
    # Adam Optimizer with weight decay
    # weight_decay value is overidden by callback
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clip_norm, weight_decay=1e-5)
    
    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model