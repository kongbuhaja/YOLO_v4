import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import *

class CSPPANSPP(Layer):
    def __init__(self, unit, row_anchors, block_size, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        self.sppblock = CSPSPPBlock(unit*2**4, activate=activate, kernel_initializer=kernel_initializer)        
        self.upsamples = []
        for l in range(1, row_anchors):
            u = unit*2**min(row_anchors + 1 - l , 4)
            self.upsamples += [[UpsampleLayer(u, activate=activate, kernel_initializer=kernel_initializer),
                                ConvLayer(u, 1, activate=activate, kernel_initializer=kernel_initializer),
                                CSPBlockB(u, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)]]
            
        self.downsamples = []
        for l in range(1, row_anchors):
            u = unit*2**min(2 + l, 4)
            self.downsamples += [[ConvLayer(u, 3, 2, activate=activate, kernel_initializer=kernel_initializer),
                                  CSPBlockB(u, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)]]

        self.convs = []
        for l in range(row_anchors):
            self.convs += [ConvLayer(unit*2**min(3+l, 5), 3, activate=activate, kernel_initializer=kernel_initializer)]

        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        up_branch = [self.sppblock(x[-1], training)]
        for l, (upsample, transition, block) in enumerate(self.upsamples):
            u = upsample(up_branch[-1], training)
            br = transition(x[-l-2], training)
            c = self.concat([u, br])
            up_branch += [block(c, training)]

        down_branch = [up_branch[-1]]
        for l, (downsample, block) in enumerate(self.downsamples):
            d = downsample(down_branch[-1], training)
            c = self.concat([d, up_branch[-l-2]])
            down_branch += [block(c, training)]

        branch = []
        for l, conv in enumerate(self.convs):
            branch += [conv(down_branch[l], training)]
            
        return branch
    
class PANSPP(Layer):
    def __init__(self, unit, row_anchors, block_size, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        self.sppblock = SPPBlock(unit*2**4, activate=activate, kernel_initializer=kernel_initializer)        
        self.upsamples = []
        for l in range(1, row_anchors):
            u = unit*2**min(row_anchors + 1 - l , 4)
            self.upsamples += [[UpsampleLayer(u, activate=activate, kernel_initializer=kernel_initializer),
                                ConvLayer(u, 1, activate=activate, kernel_initializer=kernel_initializer),
                                PlainBlockB(u, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)]]
            
        self.downsamples = []
        for l in range(1, row_anchors):
            u = unit*2**min(2 + l, 4)
            self.downsamples += [[ConvLayer(u, 3, 2, activate=activate, kernel_initializer=kernel_initializer),
                                  PlainBlockB(u, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)]]
        
        self.convs = []
        for l in range(row_anchors):
            self.convs += [ConvLayer(unit*2**min(3+l, 5), 3, activate=activate, kernel_initializer=kernel_initializer)]

        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        up_branch = [self.sppblock(x[-1], training)]
        for l, (upsample, transition, block) in enumerate(self.upsamples):
            u = upsample(up_branch[-1], training)
            br = transition(x[-l-2], training)
            c = self.concat([u, br])
            up_branch += [block(c, training)]

        down_branch = [up_branch[-1]]
        for l, (downsample, block) in enumerate(self.downsamples):
            d = downsample(down_branch[-1], training)
            c = self.concat([d, up_branch[-l-2]])
            down_branch += [block(c, training)]

        branch = []
        for l, conv in enumerate(self.convs):
            branch += [conv(down_branch[l], training)]
            
        return branch

class CSPFPN(Layer):
    def __init__(self, unit, row_anchors, block_size, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        self.block = PlainBlockB(unit*2**4, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)
        self.upsamples = []
        for l in range(1, row_anchors):
            u = unit*2**min(row_anchors + 1 - l , 4)
            self.upsamples += [[UpsampleLayer(u, activate=activate, kernel_initializer=kernel_initializer),
                                ConvLayer(u, activate=activate, kernel_initializer=kernel_initializer),
                                CSPBlockB(u, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)]]

        self.convs = []
        for l in range(row_anchors):
            self.convs += [ConvLayer(unit*2**min(row_anchors + 2 - l, 5), 3, activate=activate, kernel_initializer=kernel_initializer)]
        
        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        up_branch = [self.block(x[-1], training)]
        for l, (upsample, transition, block) in enumerate(self.upsamples):
            u = upsample(up_branch[-1], training)
            b = transition(x[-l-2], training)
            c = self.concat([u, b])
            up_branch += [block(c, training)]

        branch = []
        for l, conv in enumerate(self.convs):
            branch += [conv(up_branch[-l-1], training)]

        return branch

class FPN(Layer):
    def __init__(self, unit, row_anchors, block_size, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        self.block = PlainBlockB(unit*2**4, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)
        self.upsamples = []
        for l in range(1, row_anchors):
            u = unit*2**min(row_anchors + 1 - l , 4)
            self.upsamples += [[UpsampleLayer(u, activate=activate, kernel_initializer=kernel_initializer),
                                PlainBlockB(u, 'Bottle', block_size, activate=activate, kernel_initializer=kernel_initializer)]]

        self.convs = []
        for l in range(row_anchors):
            self.convs += [ConvLayer(unit*2**min(row_anchors + 2 - l, 5), 3, activate=activate, kernel_initializer=kernel_initializer)]
        
        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        up_branch = [self.block(x[-1], training)]
        for l, (upsample, block) in enumerate(self.upsamples):
            u = upsample(up_branch[-1], training)
            c = self.concat([u, x[-l-2]])
            up_branch += [block(c, training)]

        branch = []
        for l, conv in enumerate(self.convs):
            branch += [conv(up_branch[-l-1], training)]

        return branch

class tinyFPN(Layer):
    def __init__(self, unit, row_anchors, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        self.conv = ConvLayer(unit*2**3, 3, activate=activate, kernel_initializer=kernel_initializer)
        
        self.upsamples = []
        for l in range(1, row_anchors):
            self.upsamples += [UpsampleLayer(unit*2**min(row_anchors + 1 - l, 4), activate=activate, kernel_initializer=kernel_initializer)]

        self.convs = []
        for l in range(row_anchors):
            self.convs += [ConvLayer(unit*2**min(4-l, 5), 3, activate=activate, kernel_initializer=kernel_initializer)]
        
        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        up_branch = [self.conv(x[-1], training)]

        for l, upsample in enumerate(self.upsamples):
            u = upsample(up_branch[-1], training)
            up_branch += [self.concat([u, x[-l-2]])]

        branch = []
        for l, conv in enumerate(self.convs):
            branch += [conv(up_branch[-l-1], training)]

        return branch

class reOrg(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot):
        super().__init__()
        self.convm = ConvLayer(unit*2, 3, activate=activate, kernel_initializer=kernel_initializer)
        
        self.convl_1 = ConvLayer(unit*2**5, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.convl_2 = ConvLayer(unit*2**5, 3, activate=activate, kernel_initializer=kernel_initializer)

        self.conv = ConvLayer(unit*2**5, 3, activate=activate, kernel_initializer=kernel_initializer)

        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        m = self.convm(x[0], training)
        medium_branch = self.concat([m[:, ::2, ::2], m[:, 1::2, ::2], m[:, ::2, 1::2], m[:, 1::2, 1::2]], -1)

        l = self.convl_1(x[1], training)
        large_branch = self.convl_2(l, training)
        x = self.concat([large_branch, medium_branch], -1)
        x = self.conv(x, training)

        return x
    