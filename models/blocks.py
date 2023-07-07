from tensorflow.keras.layers import Layer, Reshape, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he

from models.layers import *

class DarknetBlock(Layer):
    def __init__(self, unit, layer, layer_num, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.layers = [Identity]
        self.layer_num = layer_num
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.pre_conv = DarknetConv(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if layer == 'Conv':
            self.layers = [DarknetConv(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.layer_num)]
        elif layer == 'Resnet':
            self.layers = [DarknetResidual([self.unit//2, self.unit], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.layer_num)]

    def call(self, input, training=False):
        x = self.pre_conv(input, training)

        for l in range(self.layer_num):
            x = self.layers[l](x, training)
        
        return x
    
class CSPDarknetBlock(Layer):
    def __init__(self, unit, layer, layer_num, div=2, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.layers = [Identity()]
        self.layer_num = layer_num
        self.div = div
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.pre_conv = DarknetConv(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)     
        
        self.block_pre_conv = DarknetConv(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if layer == 'Conv':
            self.layers = [DarknetConv(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.layer_num)]
        elif layer == 'CSPConv':
            self.layers = [CSPDarknetConv(self, unit, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.layer_num)]
        elif layer == 'Resnet':
            self.layers = [DarknetResidual([self.unit//2, self.unit//self.div], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.layer_num)]
        self.block_transition = DarknetConv(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.branch_transition = DarknetConv(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.concat_transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    def call(self, input, training=False):
        x = self.pre_conv(input, training)
        branch = self.branch_transition(x, training)

        x = self.block_pre_conv(x, training)
        for l in range(self.layer_num):
            x = self.layers[l](x, training)
        x = self.block_transition(x, training)

        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x

class ReverseDarknetBlock(Layer):
    def __init__(self, unit, layer=False, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit=unit
        self.kernel_initializer = kernel_initializer
        self.activate = activate
        self.layer = Identity()

        self.conv1 = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv3 = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if layer=='SPP':
            self.layer = SPP(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv4 = DarknetConv(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv5 = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)

        x = self.layer(x, training)

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
        
        self.concat = Concatenate()

    def call(self, branch1, branch2, training=False):
        branch2 = self.upsample(branch2, training)       
        x = self.concat([branch1, branch2])
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
    
        self.concat = Concatenate()

    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)
        x = self.concat([branch1, branch2])
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