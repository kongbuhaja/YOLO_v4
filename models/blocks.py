from tensorflow.keras.layers import Layer, Reshape, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he

from models.layers import *

class DarknetBlock(Layer):
    def __init__(self, unit, block_layer, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = [DarknetConv(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        if block_layer == 'Conv':
            self.layers += [DarknetConv(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        elif block_layer == 'Resnet':
            self.layers += [DarknetResidual([self.unit//2, self.unit], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]

    def call(self, x, training=False):
        x = x

        for l in range(len(self.layers)):
            x = self.layers[l](x, training)
        
        return x
    
class CSPOSABlock(Layer):
    def __init__(self, unit, block_layer, size, branch_transition=True, block_pre_transition=True, block_post_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.block_layers = [Identity()]
        self.size = size
        self.activate = activate
        self.branch_transition = Identity()
        self.block_pre_transition = Identity()
        self.block_post_transition = Identity()
        self.kernel_initializer = kernel_initializer

        self.pre_conv = DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)     
        
        if branch_transition:
            self.branch_transition = DarknetConv(self.unit//4, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        if block_pre_transition:
            self.block_pre_transition = DarknetConv(self.unit//4, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        if block_layer == 'Conv':
            self.block_layers = [DarknetConv(self.unit//4, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        elif block_layer == 'Resnet':
            self.block_layers = [DarknetResidual([self.unit//4, self.unit//4], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        if block_post_transition:
            self.block_post_transition = DarknetConv(self.unit//2, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    def call(self, x, training=False):
        x = self.pre_conv(x, training)
        branch1 = self.branch_transition(x, training)
    
        branch2 = self.block_pre_transition(x, training)
        x = branch2
        branchs = []
        for l in range(self.size):
            x = self.block_layers[l](x, training)
            branchs += [x]
        x = self.concat(branchs)
        x = self.block_post_transition(x, training)

        x = self.concat([x, branch2, branch1])

        return x
    
class CSPDarknetBlock(Layer):
    def __init__(self, unit, block_layer, size, div=2, branch_transition=True, block_pre_transition=True, block_post_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.div = div
        self.activate = activate
        self.branch_transition = Identity()
        self.kernel_initializer = kernel_initializer

        self.pre_conv = DarknetConv(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)     
        
        if branch_transition:
            self.branch_transition = DarknetConv(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        if block_layer == 'Conv':
            self.layers = [DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        elif block_layer == 'Resnet':
            self.layers = [DarknetResidual([self.unit//2, self.unit//self.div], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        if block_pre_transition:
            self.layers = [DarknetConv(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)] + self.layers
        if block_post_transition:
            self.layers += [DarknetConv(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]

        self.concat_transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    def call(self, x, training=False):
        x = self.pre_conv(x, training)
        branch = self.branch_transition(x, training)
    
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)

        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x
    
class DarknetTinyBlock(Layer):
    def __init__(self, unit, block_layer, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.block_layers = [Identity()]
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.split = SplitLayer(2, 1)
        self.pre_conv = DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        if block_layer == 'Conv':
            self.block_layers = [DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        elif block_layer == 'Resnet':
            self.block_layers = [DarknetResidual([self.unit//2, self.unit//self.div], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        
        self.concat = Concatenate()
        self.transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, x, training):
        x = self.split(x)
        x = self.pre_conv(x, training)
        short_cut = x
        for l in range(self.size):
            x = self.block_layers[l](x, training)
        x = self.concat([x, short_cut])
        x = self.transition(x, training)

        return x

class ReverseDarknetBlock(Layer):
    def __init__(self, unit, size, block_layer=False, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.block_layer = block_layer
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = []
        for l in range(self.size):
            self.layers += [DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]
            if l==1 and self.block_layer=='SPP':
                self.layers += [SPP(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)]
            self.layers += [DarknetConv(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        self.layers += [DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]

    def call(self, x, training=False):
        x = x
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)

        return x
    
class ReverseCSPDarknetBlock(Layer):
    def __init__(self, unit, size, block_layer=False, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.block_layer = block_layer
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.branch_transition = DarknetConv(self.unit//2, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.layers = []
        for l in range(self.size):
            self.layers += [DarknetConv(self.unit//2, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]
            if l==1 and self.block_layer=='SPP':
                self.layers += [SPP(self.unit//2, activate=self.activate, kernel_initializer=self.kernel_initializer)]
            self.layers += [DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)]

        self.concat_transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.concat = Concatenate()

    def call(self, x, training=False):
        branch = self.branch_transition(x, training)

        x = x
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)
        
        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x
    
class CSPDarknetUpsampleBlock(Layer):
    def __init__(self, unit, size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.branch_transition = Identity()

        if branch_transition:
            self.branch_transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.upsample = DarknetUpsample(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_csp_darknet_block = ReverseCSPDarknetBlock(unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        self.concat = Concatenate()

    def call(self, branch1, branch2, training=False):
        branch1 = self.upsample(branch1, training)
        branch2 = self.branch_transition(branch2, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_csp_darknet_block(x, training)

        return x

class CSPDarknetDownsampleBlock(Layer):
    def __init__(self, unit, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.downsample = DarknetDownsample(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_csp_darknet_block = ReverseCSPDarknetBlock(self.unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_csp_darknet_block(x, training)

        return x

class DarknetUpsampleBlock(Layer):
    def __init__(self, unit, size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.branch_transition = Identity()

        if branch_transition:
            self.branch_transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.upsample = DarknetUpsample(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_darknet_block = ReverseDarknetBlock(self.unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        self.concat = Concatenate()

    def call(self, branch1, branch2, training=False):
        branch1 = self.upsample(branch1, training)
        branch2 = self.branch_transition(branch2, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_darknet_block(x, training)

        return x

class DarknetDownsampleBlock(Layer):
    def __init__(self, unit, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.downsample = DarknetDownsample(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_darket_block = ReverseDarknetBlock(self.unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_darket_block(x, training)

        return x

class YoloHeadBlock(Layer):
    def __init__(self, unit, scale, col_anchors, num_classes, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.scale = scale
        self.num_classes = num_classes
        self.col_anchors = col_anchors
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv1 = DarknetConv(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.col_anchors * (self.num_classes + 5), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)
        self.reshape = Reshape((self.scale, self.scale, self.col_anchors, self.num_classes + 5))
        
    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.reshape(x)

        return x