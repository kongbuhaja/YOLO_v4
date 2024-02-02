import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer, Reshape
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import ConvLayer

class Detect(Layer):
    def __init__(self, decode, scales, row_anchors, col_anchors, num_classes, kernel_initializer=glorot):
        super().__init__()
        self.decode = decode
        self.detects = []
        for l in range(row_anchors):
            self.detects += [[ConvLayer(col_anchors * (num_classes + 5), 1, activate=False, bn=False, kernel_initializer=kernel_initializer),
                              Reshape((*scales[l][::-1], col_anchors, num_classes + 5))]]

    @tf.function
    def call(self, x, training=False):
        branch = []
        for l, (detect, reshape) in enumerate(self.detects):
            r = reshape(detect(x[l], training))
            branch += [self.decode(r)]

        return branch