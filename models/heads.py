import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import YoloHeadBlock

class YoloHead(Layer):
    def __init__(self, unit, decode, scales, row_anchors, col_anchors, num_classes, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.decode = decode
        self.scales = scales
        self.row_anchors = row_anchors
        self.col_anchors = col_anchors
        self.num_classes = num_classes
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = []
        for l in range(self.row_anchors):
            self.layers += [YoloHeadBlock(self.unit * min(l+1, 4), self.scales[l], self.col_anchors, self.num_classes,
                                      activate=self.activate, kernel_initializer=self.kernel_initializer)]

    @tf.function
    def call(self, x, training=False):
        branchs = []
        for l in range(self.row_anchors):
            branchs += [self.decode(self.layers[l](x[l], training))]

        return branchs