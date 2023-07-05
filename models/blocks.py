from tensorflow.keras.layers import Layer, Reshape, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he

from models.layers import *

class DarknetBlock(Layer):
    def __init__(self, units, block, block_num, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.block = block
        self.block_num = block_num
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.pre_conv = DarknetConv(self.units, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if self.block == 'Resnet':
            self.res_blocks = [DarknetResidual([self.units//2, self.units], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.block_num)]

    def call(self, input, training=False):
        x = self.pre_conv(input, training)

        for r in range(self.block_num):
            x = self.res_blocks[r](x, training)
        
        return x
    
class CSPDarknetBlock(Layer):
    def __init__(self, units, block, block_num, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.block = block
        self.block_num = block_num
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        div = 1 if self.block_num == 1 else 2

        self.pre_conv = DarknetConv(self.units, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)     
        
        self.block_pre_conv = DarknetConv(self.units//div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if self.block == 'Resnet':
            self.block = [DarknetResidual([self.units//2, self.units//div], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.block_num)]
        self.block_transition = DarknetConv(self.units//div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.branch_transition = DarknetConv(self.units//div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.concat_transition = DarknetConv(self.units, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
    def call(self, input, training=False):
        x = self.pre_conv(input, training)
        branch = self.branch_transition(x, training)

        x = self.block_pre_conv(x, training)
        for b in range(self.block_num):
            x = self.block[b](x, training)
        x = self.block_transition(x, training)

        x = Concatenate()([x, branch])
        x = self.concat_transition(x, training)

        return x

class ReverseDarknetBlock(Layer):
    def __init__(self, unit, block=False, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit=unit
        self.kernel_initializer = kernel_initializer
        self.activate = activate
        self.block = block

        self.conv1 = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv3 = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if self.block=='SPP':
            self.block = SPP(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv4 = DarknetConv(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv5 = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        if self.block:
            x = self.block(x, training)
        x = self.conv4(x, training)
        x = self.conv5(x, training)

        return x

class DarknetUpsampleBlock(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.upsample = DarknetUpsample(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_darknet_block = ReverseDarknetBlock(unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
    def call(self, branch1, branch2, training=False):
        branch2 = self.upsample(branch2, training)       
        x = Concatenate()([branch1, branch2])
        x = self.reverse_darknet_block(x, training)

        return x

class DarknetDownsampleBlock(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.downsample = DarknetDownsample(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_darket_block = ReverseDarknetBlock(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)
        x = Concatenate()([branch1, branch2])
        x = self.reverse_darket_block(x, training)

        return x

class GridBlock(Layer):
    def __init__(self, unit, scale, num_anchors, num_classes, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.scale = scale
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv1 = DarknetConv(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.num_anchors * (self.num_classes + 5), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)
        
    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        x = Reshape((self.scale, self.scale, self.num_anchors, self.num_classes + 5))(x)

        return x