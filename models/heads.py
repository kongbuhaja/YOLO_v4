import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from models.blocks import ConvLayer

class Detect(Layer):
    def __init__(self, row_anchors, col_anchors, num_classes, kernel_initializer=None):
        super().__init__()
        self.col_anchors = col_anchors
        self.detects = []
        for l in range(row_anchors):
            self.detects += [ConvLayer(col_anchors * (num_classes + 5), 1, activate=False, bn=False, kernel_initializer=kernel_initializer)]

    @tf.function
    def call(self, x, training=False):
        branch = []
        for l, detect in enumerate(self.detects):
            r = detect(x[l], training)
            branch += [tf.reshape(r, [*r.shape[:3], self.col_anchors, -1])]

        return branch
    