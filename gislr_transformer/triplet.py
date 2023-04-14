import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd

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
    return tf.reduce_mean(loss)

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

# Custom sampler to get a batch containing N times all signs
def triplet_get_train_batch_all_signs(X, y, NON_EMPTY_FRAME_IDXS, n, num_classes, hard_classes, meta_df):
    # Arrays to store batch in
    X_batch = np.zeros([num_classes*n, CFG.INPUT_SIZE*3, CFG.N_COLS, CFG.N_DIMS], dtype=np.float32)
    y_batch = np.arange(0, num_classes, step=1/(n), dtype=np.float32).astype(np.int64)
    non_empty_frame_idxs_batch = np.zeros([num_classes*n, CFG.INPUT_SIZE*3], dtype=np.float32)

    # Dictionary mapping ordinally encoded sign to corresponding sample indices
    CLASS2IDXS = {}
    for i in range(num_classes):
        CLASS2IDXS[i] = np.argwhere(y == i).squeeze().astype(np.int32)
        
    # Difficult class mapping (which classes are commonly mistaken)
    DIFFICULT_CLASS_MAP = {0: [17, 29, 104, 106, 107, 109, 113, 136, 152, 180], 1: [21, 28, 46, 84, 85, 87, 95, 104, 105, 188], 2: [24, 70, 97, 112, 166, 187, 190, 225, 227, 239], 3: [28, 29, 46, 74, 104, 160, 161, 179, 188, 208], 4: [37, 40, 43, 48, 107, 112, 121, 137, 159, 195], 5: [15, 108, 113, 125, 181, 189, 248], 6: [7, 13, 29, 36, 46, 62, 104, 106, 133, 249], 7: [6, 53, 71, 104, 106, 107, 113, 116, 157, 180], 8: [10, 29, 37, 65, 92, 158, 159, 220, 226, 245], 9: [1, 84, 104, 113, 117, 121, 132, 140, 187, 248], 10: [61, 94, 116, 133, 137, 168, 186, 221, 233, 237], 11: [104, 113, 131, 146, 175, 230, 233], 12: [10, 83, 133, 141, 147, 154, 160, 164, 186, 196], 13: [17, 26, 28, 75, 107, 113, 116, 120, 160, 188], 14: [27, 29, 48, 62, 66, 79, 99, 104, 112, 162], 15: [5, 6, 20, 84, 106, 108, 125, 139, 249], 16: [20, 24, 52, 94, 100, 104, 106, 109, 117, 146], 17: [3, 18, 19, 21, 26, 29, 101, 126, 227, 241], 18: [1, 9, 17, 20, 26, 69, 179, 181, 188, 241], 19: [26, 38, 74, 77, 78, 79, 89, 104, 107, 112], 20: [29, 46, 52, 61, 84, 113, 125, 129, 151, 195], 21: [5, 7, 20, 33, 35, 56, 71, 142, 231, 246], 22: [6, 13, 20, 21, 27, 34, 39, 71, 85, 104], 23: [98, 104, 164, 194, 239], 24: [72, 73, 83, 197, 249], 25: [14, 29, 48, 59, 74, 98, 104, 128, 159, 203], 26: [21, 28, 46, 84, 104, 106, 133, 147, 177, 188], 27: [21, 29, 53, 58, 84, 104, 188, 192, 234, 241], 28: [20, 27, 29, 42, 46, 74, 106, 161, 188, 231], 29: [14, 60, 69, 100, 104, 166, 189], 30: [3, 16, 29, 51, 52, 58, 71, 73, 88, 94], 31: [1, 17, 26, 46, 113, 159, 241], 32: [104, 107, 184, 233, 239, 249], 33: [18, 42, 80, 97, 104, 113, 124, 126, 175, 207], 34: [20, 106, 123, 166, 243], 35: [0, 1, 21, 64, 106, 109, 167, 170, 176, 244], 36: [1, 45, 46, 58, 62, 64, 107, 182, 196, 231], 37: [44, 45, 48, 65, 68, 94, 102, 115, 116, 224], 38: [1, 75, 104, 107, 113, 120, 129, 133, 137, 163], 39: [20, 46, 104, 113, 165, 172, 179, 188, 214, 220], 40: [35, 74, 84, 112, 126, 133, 184, 186, 187, 237], 41: [8, 38, 68, 76, 94, 102, 164, 201, 223, 226], 42: [20, 28, 35, 45, 66, 84, 85, 104, 113, 117], 43: [102, 113, 133, 185, 191, 218, 222, 226, 249], 44: [4, 40, 48, 58, 104, 106, 114, 125, 153, 195], 45: [9, 28, 29, 42, 53, 104, 106, 160, 175, 227], 46: [9, 20, 27, 58, 84, 104, 161, 179, 188, 208], 47: [6, 18, 40, 104, 133, 179, 180, 187, 220, 231], 48: [4, 27, 60, 100, 105, 111, 112, 133, 183, 197], 49: [69, 104, 138], 50: [12, 29, 51, 60, 64, 106, 126, 207, 231, 245], 51: [12, 29, 50, 56, 58, 60, 133, 135, 206, 207], 52: [16, 30, 41, 47, 71, 169, 177, 223, 230, 241], 53: [6, 79, 82, 104, 122, 133, 155, 175, 192, 196], 54: [14, 20, 36, 74, 83, 104, 126], 55: [4, 25, 29, 48, 82, 100, 104, 140, 188, 220], 56: [35, 45, 53, 58, 61, 77, 83, 106, 121, 192], 57: [104, 113, 124, 139, 160, 171, 208, 211, 227, 238], 58: [18, 29, 104, 113, 115, 117, 131, 179, 188, 237], 59: [104, 162, 188, 220, 239], 60: [29, 33, 51, 100, 104, 133, 135, 140, 160, 207], 61: [29, 35, 58, 64, 97, 102, 104, 106, 109, 236], 62: [29, 36, 42, 64, 74, 97, 105, 113, 142, 231], 63: [19, 48, 111, 128, 159, 215, 218, 227, 239, 245], 64: [35, 50, 61, 97, 101, 104, 105, 106, 196, 238], 65: [8, 59, 66, 91, 104, 133, 220, 227], 66: [8, 65, 91, 92, 104, 106, 109, 180, 220, 226], 67: [25, 69, 98, 171, 181, 204, 208, 222], 68: [14, 38, 41, 76, 78, 88, 97, 104, 106, 111], 69: [19, 28, 31, 45, 46, 47, 74, 160, 181, 208], 70: [41, 45, 74, 91, 104, 106, 107, 113, 225, 232], 71: [6, 7, 15, 19, 22, 52, 76, 221, 231, 246], 72: [52, 53, 88, 104, 155, 221, 231], 73: [30, 81, 83, 104, 115, 148, 149, 166, 189, 220], 74: [3, 26, 29, 53, 56, 89, 104, 117, 133, 184], 75: [4, 14, 48, 57, 99, 104, 112, 133, 145, 203], 76: [36, 92, 104, 107, 109, 113, 115, 124, 125, 142], 77: [6, 9, 14, 29, 35, 53, 62, 89, 106, 107], 78: [17, 19, 35, 77, 104, 107, 133, 177, 181, 225], 79: [1, 21, 28, 29, 35, 46, 53, 140, 188, 212], 80: [3, 5, 28, 29, 81, 92, 164, 223, 231, 232], 81: [27, 28, 46, 70, 71, 97, 141, 203, 206, 232], 82: [19, 78, 100, 106, 113, 117, 126, 133, 140], 83: [58, 62, 102, 113, 164, 166, 179, 227, 233, 237], 84: [80, 104, 105, 106, 121, 129, 133, 173, 179, 188], 85: [28, 91, 96, 104, 124, 126, 152, 163, 188, 232], 86: [87, 104, 128], 87: [29, 39, 104, 128, 188], 88: [29, 36, 165, 169, 207], 89: [25, 29, 48, 74, 77, 78, 104, 106, 117, 175], 90: [0, 104, 112, 133, 161], 91: [9, 27, 47, 74, 97, 104, 138, 140, 174, 188], 92: [29, 47, 95, 115, 187, 227, 231, 237], 93: [18, 27, 39, 58, 62, 101, 102, 121, 151, 214], 94: [10, 64, 91, 106, 116, 180, 187, 239, 246], 95: [5, 27, 36, 70, 92, 104, 105, 106, 124, 179], 96: [20, 29, 43, 84, 85, 142, 149, 179, 188, 191], 97: [27, 58, 104, 109, 164, 179, 196, 235, 237, 238], 98: [29, 42, 47, 64, 67, 104, 113, 135, 161, 196], 99: [29, 75, 104, 133, 145, 153, 188, 241], 100: [29, 91, 106, 133, 140, 152], 101: [5, 7, 17, 27, 39, 46, 62, 99, 151, 211], 102: [20, 29, 58, 61, 83, 106, 117, 133, 164, 220], 103: [35, 104, 136, 180, 192], 104: [18, 42, 48, 51, 85, 133, 147, 154, 182], 105: [20, 82, 85, 106, 133, 173, 179, 188, 195, 248], 106: [29, 30, 53, 85, 88, 104, 108, 133, 165, 180], 107: [1, 53, 70, 71, 75, 78, 92, 113, 132, 247], 108: [1, 5, 21, 106, 126, 135, 195, 211], 109: [35, 53, 104, 108, 113, 115, 116, 124, 125, 220], 110: [20, 85, 133, 151, 154, 197, 201, 217, 240], 111: [14, 33, 36, 68, 109, 135, 137, 182], 112: [2, 29, 51, 82, 104, 106, 111, 119, 131, 133], 113: [26, 29, 133, 241], 114: [23, 32, 53, 63, 67, 106, 112, 119, 136, 192], 115: [29, 58, 66, 92, 109, 131, 132, 133, 179, 237], 116: [3, 10, 15, 25, 35, 73, 108, 152, 158, 231], 117: [53, 65, 74, 104, 133, 180, 192], 118: [86, 104, 207, 231], 119: [29, 104, 146, 207], 120: [35, 48, 72, 80, 104, 105, 106, 121, 142, 189], 121: [29, 104, 113, 164, 208, 231, 238], 122: [37, 158, 159], 123: [29, 40, 78, 104, 106, 107, 132, 196, 240, 243], 124: [46, 107, 123, 126, 132, 135, 152, 160, 175, 195], 125: [5, 15, 36, 42, 104, 106, 131, 226, 243, 249], 126: [1, 12, 17, 29, 46, 104, 121, 166, 195, 231], 127: [29, 40, 47, 104, 113, 165, 184, 187, 204, 231], 128: [25, 42, 86, 87, 95, 104, 118, 124, 129, 207], 129: [38, 42, 53, 74, 97, 106, 117, 136, 137, 193], 130: [3, 14, 42, 48, 64, 128, 194, 197, 207, 210], 131: [3, 8, 27, 68, 79, 106, 107, 109, 133, 167], 132: [53, 106, 107, 121, 133, 143, 175, 184, 190, 238], 133: [35, 182, 226, 228, 230], 134: [70, 83, 104, 113, 149, 154, 155, 164, 185, 223], 135: [11, 14, 44, 48, 60, 111, 119, 133, 170, 182], 136: [14, 24, 29, 33, 63, 73, 74, 83, 184, 193], 137: [3, 17, 35, 91, 97, 104, 107, 109, 131, 135], 138: [19, 49, 113, 133, 182, 219], 139: [36, 92, 106, 142, 170, 186, 202, 227, 231, 232], 140: [29, 30, 42, 55, 69, 86, 99, 100, 106, 113], 141: [9, 17, 18, 29, 62, 76, 84, 97, 106, 111], 142: [21, 36, 71, 76, 117, 139, 188, 216, 227, 231], 143: [9, 100, 105, 106, 113, 121, 173, 238, 248, 249], 144: [4, 12, 28, 64, 80, 81, 95, 106, 147, 232], 145: [70, 75, 99, 104, 112, 113, 159, 170], 146: [11, 97, 104, 119, 170, 233], 147: [6, 20, 21, 22, 27, 28, 62, 157, 179, 227], 148: [11, 23, 25, 29, 59, 104, 115, 131, 159, 224], 149: [43, 52, 81, 91, 104, 130, 154, 155, 167, 185], 150: [6, 17, 104, 136, 153, 160, 181, 184, 199, 200], 151: [7, 20, 54, 84, 85, 87, 104, 121, 188, 227], 152: [1, 5, 10, 33, 35, 42, 64, 106, 128, 135], 153: [104, 132, 133, 136, 175, 184, 210, 237], 154: [6, 17, 26, 74, 97, 117, 148, 165, 187], 155: [24, 29, 154], 156: [3, 7, 36, 104, 116, 125, 133, 158, 168, 180], 157: [34, 104, 106, 108, 113, 117, 200, 227, 231, 232], 158: [21, 94, 104, 106, 113, 116, 140], 159: [3, 14, 133, 174, 208], 160: [6, 36, 40, 57, 81, 97, 104, 106, 107, 237], 161: [14, 17, 21, 26, 27, 28, 29, 46, 106, 147], 162: [106, 117, 159], 163: [29, 36, 70, 85, 97, 104, 111, 113, 130, 235], 164: [9, 10, 36, 58, 62, 104, 106, 115, 231, 236], 165: [64, 106, 133, 178], 166: [9, 15, 56, 106, 128, 133, 242, 248], 167: [3, 14, 29, 40, 43, 104, 149, 150, 168, 188], 168: [9, 56, 75, 77, 97, 102, 167, 179, 202, 242], 169: [24, 29, 88, 106, 135, 146, 167, 197, 217, 230], 170: [1, 21, 27, 47, 100, 104, 106, 119, 136, 188], 171: [17, 42, 57, 106, 110, 143, 201, 208, 211, 227], 172: [0, 35, 104, 107, 112, 114, 117, 133, 135, 153], 173: [21, 29, 104, 105, 133, 143, 166, 248], 174: [29, 44, 45, 113, 211, 227], 175: [12, 29, 45, 53, 62, 74, 97, 124, 160, 181], 176: [4, 6, 29, 30, 36, 58, 61, 106, 113, 246], 177: [26, 81, 85, 99, 104, 113, 140, 148, 157, 220], 178: [5, 81, 93, 104, 138, 151, 188, 218, 235, 242], 179: [3, 53, 58, 74, 103, 104, 106, 107, 115, 231], 180: [40, 69, 74, 84, 104, 116, 117, 133, 187, 231], 181: [1, 14, 25, 35, 69, 106, 113, 194, 208, 212], 182: [106, 111, 117, 154], 183: [4, 25, 33, 48, 79, 81, 104, 112, 154, 203], 184: [0, 7, 40, 64, 104, 106, 112, 113, 117, 187], 185: [17, 54, 104, 114, 130, 149, 192, 223], 186: [36, 64, 106, 108, 109, 116, 136, 139, 142, 231], 187: [12, 29, 40, 56, 58, 74, 99, 112, 125, 184], 188: [3, 46, 58, 104, 105, 113, 133, 142, 179, 238], 189: [1, 14, 31, 42, 69, 77, 101, 104, 106, 128], 190: [21, 47, 66, 74, 82, 106, 113, 115, 135, 205], 191: [29, 43, 50, 66, 104, 110, 133, 134, 136, 164], 192: [1, 53, 104, 106, 117, 133, 193, 218, 248], 193: [53, 111, 117, 136, 192, 204, 227], 194: [14, 133, 181, 227], 195: [42, 104, 113], 196: [0, 29, 61, 104, 113, 117, 124, 126, 133, 142], 197: [29, 56, 93, 163, 182, 207, 210], 198: [35, 106, 189, 213, 227], 199: [75, 104, 106, 113, 124, 150, 178, 200, 235, 242], 200: [22, 101, 104, 114, 122, 150, 151, 199, 235, 242], 201: [24, 25, 52, 67, 72, 102, 110, 111, 118, 164], 202: [11, 30, 37, 74, 104, 122, 156, 158, 180, 192], 203: [44, 77, 80, 81, 84, 104, 135, 154, 160, 163], 204: [0, 17, 21, 24, 90, 91, 112, 113, 127, 187], 205: [6, 7, 74, 78, 106, 116, 124, 152, 190, 215], 206: [38, 71, 106, 107, 132, 141, 143, 173, 179, 227], 207: [3, 9, 20, 29, 38, 68, 102, 104, 106, 195], 208: [46, 79, 104, 108, 113, 120, 124, 133, 162, 195], 209: [54, 113, 192, 199, 227, 231], 210: [64, 85, 91, 97, 133, 154, 163, 197, 207, 228], 211: [19, 29, 31, 74, 76, 92, 140, 141, 143, 160], 212: [46, 74, 79, 104, 113, 133, 160, 181, 195, 237], 213: [22, 27, 80, 101, 104, 120, 232], 214: [13, 35, 58, 79, 104, 105, 113, 151, 179, 188], 215: [7, 35, 66, 80, 106, 125, 126, 133, 190, 205], 216: [1, 9, 27, 28, 46, 84, 92, 104, 115, 131], 217: [12, 24, 68, 88, 100, 102, 106, 111, 169, 198], 218: [5, 15, 21, 185, 227, 249], 219: [39, 46, 70, 104, 106, 133, 147], 220: [28, 91, 92, 106, 113, 131, 133, 135, 136, 231], 221: [27, 29, 64, 71, 94, 106, 113, 158, 180, 227], 222: [43, 115, 133, 198], 223: [41, 66, 83, 97, 103, 104, 134, 149, 155, 222], 224: [66, 94, 102, 115, 227], 225: [13, 25, 28, 29, 33, 74, 104, 133, 160, 208], 226: [29, 36, 58, 62, 104, 164, 176, 237], 227: [17, 29, 100, 133, 147, 154, 161, 163, 232, 240], 228: [17, 165, 186], 229: [0, 21, 40, 53, 75, 84, 97, 113, 141, 142], 230: [58, 73, 83, 88, 104, 106, 115, 131, 169, 237], 231: [5, 12, 26, 35, 56, 61, 74, 104, 107, 108], 232: [27, 28, 29, 71, 104, 157, 179], 233: [7, 11, 29, 42, 97, 104, 131, 133, 146, 239], 234: [104, 106, 113, 136, 159, 212], 235: [3, 27, 58, 62, 95, 104, 125, 130, 152, 163], 236: [5, 15, 36, 61, 91, 104, 106, 125, 220, 231], 237: [3, 17, 97, 104, 109, 133, 165, 170, 176, 188], 238: [29, 104, 106, 138, 141, 175, 180, 189, 202, 235], 239: [29, 102, 104, 159, 170, 211, 227, 235], 240: [26, 34, 85, 97, 104, 133, 154, 206, 207], 241: [1, 18, 26, 27, 28, 29, 46, 71, 181, 188], 242: [101, 104, 113, 117, 164, 177, 178, 235], 243: [7, 12, 15, 34, 84, 113, 157, 190, 205, 215], 244: [35, 61, 64, 109, 113, 126, 137, 196, 204, 227], 245: [10, 12, 17, 50, 64, 119, 124, 146, 152, 180], 246: [6, 104, 106, 116, 117, 125, 157, 179, 180, 221], 247: [4, 45, 79, 81, 89, 107, 125, 188, 208, 225], 248: [17, 26, 104, 106, 120, 132, 140, 195, 211, 238], 249: [5, 9, 27, 35, 36, 46, 56, 77, 180, 195]}
    if hard_classes > 10: 
        raise ValueError('hard_classes must be <= 10.')

    while True:
        # Fill batch arrays
        for i in range(num_classes):
            anchor_idxs = np.random.choice(CLASS2IDXS[i], n)
            positive_idxs = np.zeros(n, dtype=np.int64)
            negative_idxs = np.zeros(n, dtype=np.int64)

            for z in range(n):
                participant_id_pos = meta_df[(meta_df.participant_id == meta_df.iloc[anchor_idxs[z]].participant_id) & (meta_df.label==z) & (meta_df.index != anchor_idxs[z])].index
                participant_id_neg = meta_df[(meta_df.participant_id == meta_df.iloc[anchor_idxs[z]].participant_id) & (meta_df.label.isin(DIFFICULT_CLASS_MAP[i][:hard_classes]))].index
                if len(participant_id_pos) == 0 or len(participant_id_neg) == 0:
                    positive_idxs[z] = np.random.choice(CLASS2IDXS[i][0], 1)
                    neg_idxs[z] = np.random.choice(CLASS2IDXS[DIFFICULT_CLASS_MAP[i][0]], 1)
                else:
                    positive_idxs[z] = np.random.choice(participant_id_pos)
                    negative_idxs[z] = np.random.choice(participant_id_neg)

            anchor_idxs = np.random.choice(CLASS2IDXS[i], n)
            positive_idxs = np.random.choice(CLASS2IDXS[i], n)
            neg_idxs = np.random.choice(CLASS2IDXS[i], n)

            X_batch[i*n:(i+1)*n, :CFG.INPUT_SIZE] = X[anchor_idxs]
            X_batch[i*n:(i+1)*n, CFG.INPUT_SIZE:CFG.INPUT_SIZE*2] = X[positive_idxs]
            X_batch[i*n:(i+1)*n, CFG.INPUT_SIZE*2:CFG.INPUT_SIZE*3] = X[neg_idxs]

            non_empty_frame_idxs_batch[i*n:(i+1)*n, :CFG.INPUT_SIZE] = NON_EMPTY_FRAME_IDXS[anchor_idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n, CFG.INPUT_SIZE:CFG.INPUT_SIZE*2] = NON_EMPTY_FRAME_IDXS[positive_idxs]
            non_empty_frame_idxs_batch[i*n:(i+1)*n, CFG.INPUT_SIZE*2:CFG.INPUT_SIZE*3] = NON_EMPTY_FRAME_IDXS[neg_idxs]
            yield {'frames': X_batch, 'non_empty_frame_idxs': non_empty_frame_idxs_batch }, y_batch

def triplet_load_data(val_fold, train_all):
    
    meta_df = pd.read_csv(CFG.MY_DATA_DIR + "train.csv")
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
    return X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data, meta_df

def get_triplet_model( 
        units, 
        learning_rate,
        clip_norm,
        statsdict,
        ):
    # Inputs
    frames = tf.keras.layers.Input([CFG.INPUT_SIZE*3, CFG.N_COLS, CFG.N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([CFG.INPUT_SIZE*3], dtype=tf.float32, name='non_empty_frame_idxs')
    
    outputs = []
    elayer = Embedding(units)
    
    # 3 Embeddings [anchor, pos, neg]
    for i in range(0, 192, 64):
        x = tf.slice(frames, [0,0,0,0], [-1, 64, 78, 2])
        non_empty_frame_idxs2 = tf.slice(non_empty_frame_idxs, [0,0], [-1, 64])
    
        # Padding Mask
        mask = tf.cast(tf.math.not_equal(non_empty_frame_idxs2, -1), tf.float32)
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
        x = elayer(lips, left_hand, pose, non_empty_frame_idxs2)
        outputs.append(x)
    
    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    model.add_loss(tf.reduce_mean(triplet_loss(outputs)))

    # Adam Optimizer with weight decay
    # weight_decay value is overidden by callback
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clip_norm, weight_decay=1e-5)
    
    model.compile(loss=None, optimizer=optimizer)
    return model

def get_triplet_transformer(
        num_blocks,
        num_heads, 
        units, 
        mlp_dropout_ratio, 
        mlp_ratio,
        learning_rate,
        clip_norm,
        statsdict,
        ):
    # Inputs
    frames = tf.keras.layers.Input([CFG.INPUT_SIZE*3, CFG.N_COLS, CFG.N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([CFG.INPUT_SIZE*3], dtype=tf.float32, name='non_empty_frame_idxs')
    
    elayer = Embedding(units)
    tlayer = Transformer(num_blocks, num_heads, units, mlp_dropout_ratio, mlp_ratio)
    
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
    # Ignoring gradients for neg and pos sample
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        x1 = elayer(lips1, left_hand1, pose1, non_empty_frame_idxs1)
        x2 = elayer(lips2, left_hand2, pose2, non_empty_frame_idxs2)
        x1 = tlayer(x1, mask1)
        x2 = tlayer(x2, mask2)
    x0 = elayer(lips0, left_hand0, pose0, non_empty_frame_idxs0)
    x0 = tlayer(x0, mask0)

    outputs = [x0, x1, x2]

    # Create Tensorflow Model
    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)
    model.add_loss(tf.reduce_mean(triplet_loss(outputs)))
    
    # Adam Optimizer with weight decay
    # weight_decay value is overidden by callback
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, clipnorm=clip_norm, weight_decay=1e-5)
    
    model.compile(loss=None, optimizer=optimizer)
    return model

def get_triplet_weights(config, statsdict):

    # Get data
    X_train, y_train, NON_EMPTY_FRAME_IDXS_TRAIN, validation_data, meta_df = triplet_load_data(
        val_fold=config.val_fold,
        train_all=config.train_all,
    )

    # Clear all models in GPU
    tf.keras.backend.clear_session()

    # Get Model
    if config.triplet_transformer == True:
        model = get_triplet_transformer(
            num_blocks=config.num_blocks,
            num_heads=config.num_heads, 
            units=config.units, 
            mlp_dropout_ratio=config.mlp_dropout_ratio, 
            mlp_ratio=config.mlp_ratio,
            learning_rate=config.learning_rate,
            clip_norm=config.clip_norm,
            statsdict=statsdict,
        )
    else:
        model = get_triplet_model(
            units=config.units, 
            learning_rate=config.triplet_learning_rate,
            clip_norm=config.clip_norm,
            statsdict=statsdict,
        )

    emb_weights = model.get_layer(name='embedding').weights
    if config.triplet_transformer == False:
        return emb_weights, None
    else:
        tfr_weights = model.get_layer(name='transformer').weights
        return emb_weights, tfr_weights

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

    # Actual Training
    history=model.fit(
            x=triplet_get_train_batch_all_signs(
                X_train, 
                y_train, 
                NON_EMPTY_FRAME_IDXS_TRAIN, 
                n=config.batch_all_signs_n,
                num_classes=config.num_classes,
                hard_classes=config.triplet_hard_class_n,
                meta_df=meta_df,
                ),
            steps_per_epoch=len(X_train) // (config.num_classes * config.batch_all_signs_n),
            epochs=config.triplet_epochs,
            # Only used for validation data since training data is a generator
            batch_size=config.batch_size,
            callbacks=callbacks,
            verbose=config.verbose,
        )
    
    emb_weights = model.get_layer(name='embedding').weights
    if config.triplet_transformer == False:
        return emb_weights, None
    else:
        tfr_weights = model.get_layer(name='transformer').weights
        return emb_weights, tfr_weights

