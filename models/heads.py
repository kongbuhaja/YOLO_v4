import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import YoloHeadBlock

class YoloHead(Layer):
    def __init__(self, unit, scales, col_anchors, num_classes, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.scales = scales
        self.col_anchors = col_anchors
        self.num_classes = num_classes
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = []
        for l in range(len(self.scales)):
            self.layers += [YoloHeadBlock(self.unit * min(l+1, 4), self.scales[l], self.col_anchors, self.num_classes,
                                      activate=self.activate, kernel_initializer=self.kernel_initializer)]

    @tf.function
    def call(self, x, training=False):
        branchs = []
        for l in range(len(self.scales)):
            branchs += [self.layers[l](x[l], training)]

        return branchs