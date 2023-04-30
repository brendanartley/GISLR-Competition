import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from gislr_transformer.helpers import *

class ModelCFG:
    # Assorted ModelCFG
    N_ROWS = 543
    N_DIMS = 2
    INPUT_SIZE = 64

    # Initiailizers
    INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
    INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
    INIT_ZEROS = tf.keras.initializers.constant(0.0)

    # Activations
    GELU = tf.keras.activations.gelu


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
                tf.keras.layers.Dense(self.units * self.mlp_ratio, activation=ModelCFG.GELU, kernel_initializer=ModelCFG.INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(self.mlp_dropout_ratio),
                tf.keras.layers.Dense(self.units, kernel_initializer=ModelCFG.INIT_HE_UNIFORM),
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
            initializer=ModelCFG.INIT_ZEROS,
        )
        # Embedding
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=ModelCFG.INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(ModelCFG.GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=ModelCFG.INIT_HE_UNIFORM),
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
    def __init__(self, units, landmark_units):
        super(Embedding, self).__init__()
        self.units = units
        self.landmark_units = landmark_units

    def build(self, input_shape):
        # Positional Embedding, initialized with zeros
        self.positional_embedding = tf.keras.layers.Embedding(ModelCFG.INPUT_SIZE+1, self.units, embeddings_initializer=ModelCFG.INIT_ZEROS)
        # Embedding layer for Landmarks
        self.lips_embedding = LandmarkEmbedding(self.landmark_units, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(self.landmark_units, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(self.landmark_units, 'pose')
        # Landmark Weights
        self.landmark_weights = tf.Variable([0.01, 0.95, 0.01], dtype=tf.float32, name='landmark_weights')
        # Fully Connected Layers for combined landmarks
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name='fully_connected_1', use_bias=False, kernel_initializer=ModelCFG.INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(ModelCFG.GELU),
            tf.keras.layers.Dense(self.units, name='fully_connected_2', use_bias=False, kernel_initializer=ModelCFG.INIT_HE_UNIFORM),
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
            ModelCFG.INPUT_SIZE,
            tf.cast(
                non_empty_frame_idxs / max_frame_idxs * ModelCFG.INPUT_SIZE,
                tf.int32,
            ),
        )
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)
        return x
    
def get_model(
        num_blocks,
        num_heads, 
        units,
        landmark_units,
        mlp_dropout_ratio, 
        mlp_ratio,
        num_classes,
        label_smoothing,
        learning_rate,
        clip_norm,
        CFG,
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
    lips = tf.slice(x, [0,0,CFG.LIPS_START,0], [-1,CFG.INPUT_SIZE, CFG.LIPS_IDXS.size, 2])
    lips = tf.where(tf.math.equal(lips, 0.0), 0.0, (lips - CFG.statsdict["LIPS_MEAN"]) / CFG.statsdict["LIPS_STD"])
    # LEFT HAND
    left_hand = tf.slice(x, [0,0,CFG.LEFT_HAND_START,0], [-1,CFG.INPUT_SIZE, CFG.LEFT_HAND_IDXS.size, 2])
    left_hand = tf.where(tf.math.equal(left_hand, 0.0), 0.0, (left_hand - CFG.statsdict["LEFT_HANDS_MEAN"]) / CFG.statsdict["LEFT_HANDS_STD"])
    # POSE
    pose = tf.slice(x, [0,0,CFG.POSE_START,0], [-1,CFG.INPUT_SIZE, CFG.POSE_IDXS.size, 2])
    pose = tf.where(tf.math.equal(pose, 0.0), 0.0, (pose - CFG.statsdict["POSE_MEAN"]) / CFG.statsdict["POSE_STD"])
    
    # Flatten
    lips = tf.reshape(lips, [-1, CFG.INPUT_SIZE, CFG.LIPS_IDXS.size*2])
    left_hand = tf.reshape(left_hand, [-1, CFG.INPUT_SIZE, CFG.LEFT_HAND_IDXS.size*2])
    pose = tf.reshape(pose, [-1, CFG.INPUT_SIZE, CFG.POSE_IDXS.size*2])
        
    # Embedding
    x = Embedding(units, landmark_units)(lips, left_hand, pose, non_empty_frame_idxs)
    
    # Encoder Transformer Blocks
    x = Transformer(num_blocks, num_heads, units, mlp_dropout_ratio, mlp_ratio)(x, mask)
    
    # OP2: Max Pooling
    x = tf.reduce_max(x * mask, axis=1)

    # Classification Layer
    x = tf.keras.layers.Dense(num_classes, activation=tf.keras.activations.softmax, kernel_initializer=ModelCFG.INIT_GLOROT_UNIFORM)(x)
    
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
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clip_norm, clipvalue=1.0, weight_decay=1e-5)
    
    # TopK Metrics
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model