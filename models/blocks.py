from tensorflow.keras.layers import Layer, Reshape, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he

from models.layers import *

class ResidualBlockI(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(unit//2, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.add = Add()
    
    @tf.function
    def call(self, x, training=False):
        branch = x
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.add([x, branch])

        return x

class CSPResidualBlockI(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.add = Add()
    
    @tf.function
    def call(self, x, training=False):
        branch = x
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.add([x, branch])

        return x
    
class ResidualBlockC(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.branch = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv1 = ConvLayer(unit//2, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)        
        self.add = Add()
    
    @tf.function
    def call(self, x, training=False):
        branch = self.branch(x, training)
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.add([x, branch])

        return x
    
class CSPResidualBlockC(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.branch = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv1 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.add = Add()
    
    @tf.function
    def call(self, x, training=False):
        branch = self.branch(x, training)
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.add([x, branch])

        return x
    
class BottleBlock(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit*2, 3, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = self.conv2(x, training)

        return x
    
class CSPBottleBlock(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = self.conv2(x, training)

        return x
    
class PlainBlockA(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()        
        block = get_block(block)
        self.blocks = [block(unit, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]

    @tf.function
    def call(self, x, training=False):        
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)

        return x
    
class PlainBlockB(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()        
        block = get_block(block)
        self.blocks = [block(unit, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]
        self.conv = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        x = self.conv(x)

        return x
    
class CSPBlockA(Layer): 
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.branch_transition = ConvLayer(unit//2, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.block_pre_transition = ConvLayer(unit//2, 1, activate=activate, kernel_initializer=kernel_initializer)
        block = get_block(block)
        self.blocks = [block(unit//2, activate=activate, kernel_initializer=kernel_initializer)] * block_size
        self.block_post_transition = ConvLayer(unit//2, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.concat = Concatenate()
        self.concat_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        branch = self.branch_transition(x, training)
        
        x = self.block_pre_transition(x, training)
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        x = self.block_post_transition(x, training)
        
        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x
    
class CSPBlockB(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.transition = ConvLayer(unit, 1, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.branch_transition = ConvLayer(unit//2, 1, activate=activate, kernel_initializer=kernel_initializer)

        block = get_block(block)
        self.blocks = [block(unit//2, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]

        self.concat = Concatenate()
        self.concat_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        x = self.transition(x, training)
        branch = self.branch_transition(x, training)
        
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        
        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x
    
class CSPBlockC(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.branch_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.block_pre_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        block = get_block(block)
        self.blocks = [block(unit, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]
        self.block_post_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.concat = Concatenate()
        self.concat_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        branch = self.branch_transition(x, training)
        
        x = self.block_pre_transition(x, training)
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        x = self.block_post_transition(x, training)
        
        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x

class SPPBlock(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv1 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit*2, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.spp = SPPLayer(unit, activate=activate, kernel_initializer=kernel_initializer)
        self.conv3 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv4 = ConvLayer(unit*2, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.conv5 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.spp(x, training)
        x = self.conv4(x, training)
        x = self.conv5(x, training)

        return x
    
class CSPSPPBlock(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.branch_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        
        self.conv1 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv2 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.spp = SPPLayer(unit, activate=activate, kernel_initializer=kernel_initializer)
        self.conv3 = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.conv4 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)

        self.concat = Concatenate()
        self.concat_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        branch = self.branch_transition(x, training)

        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.spp(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)

        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x

class DarknetBlock(Layer):
    def __init__(self, unit, block_layer, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.block = [ConvLayer(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        if block_layer == 'Conv':
            self.block += [ConvLayer(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        elif block_layer == 'Resnet':
            self.block += [ResidualBlock([self.unit//2, self.unit], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]

    @tf.function
    def call(self, x, training=False):
        x = x

        for l in range(len(self.block)):
            x = self.block[l](x, training)
        
        return x
    
class CSPOSABlock(Layer):
    def __init__(self, unit, growth_rate, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.growth_rate = growth_rate
        self.activate = activate

        self.kernel_initializer = kernel_initializer

        self.pre_conv = ConvLayer(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.branch1_transition = ConvLayer(self.unit//4, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.branch2_transition = ConvLayer(self.unit//4, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)   
        
        self.osa = OSALayer(self.unit//2, self.growth_rate, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    @tf.function
    def call(self, x, training=False):
        x = self.pre_conv(x, training)
        branch1 = self.branch1_transition(x, training)
        branch2 = self.branch2_transition(x, training)
        
        x = self.osa(x, training)
    
        x = self.concat([x, branch1, branch2])

        return x
    
class CSPDarknetBlock(Layer):
    def __init__(self, unit, block_layer, size, div=2, branch_transition=True, block_pre_transition=True, block_post_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.size = size
        self.div = div
        self.activate = activate
        self.branch_transition = IdentityLayer()
        self.kernel_initializer = kernel_initializer

        self.conv_conv = ConvLayer(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)     
        
        self.branch_transition = ConvLayer(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer) if branch_transition else IdentityLayer()

        self.block = ConvLayer(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer) if block_pre_transition else []
        if block_layer == 'Conv':
            self.block += [ConvLayer(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        elif block_layer == 'Resnet':
            self.block += [ResidualBlock([self.unit//2, self.unit//self.div], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.size)]
        self.block += [ConvLayer(self.unit//self.div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)] if block_post_transition else []

        self.concat_transition = ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    @tf.function
    def call(self, x, training=False):
        x = self.pre_conv(x, training)
        branch = self.branch_transition(x, training)
    
        for l in range(len(self.block)):
            x = self.block[l](x, training)

        x = self.concat([x, branch])
        x = self.concat_transition(x, training)

        return x
    
class TinyBlock(Layer):
    def __init__(self, unit, block_layer, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.split = SplitLayer(2, 1)
        
        self.pre_transition = ConvLayer(unit//2, 3, activate=activate, kernel_initializer=kernel_initializer)
        if block_layer == 'Conv':
            self.layers = [ConvLayer(unit//2, 3, activate=activate, kernel_initializer=kernel_initializer) for _ in range(size)]
        elif block_layer == 'Resnet':
            self.layers = [ResidualBlock(unit//2, activate=activate, kernel_initializer=kernel_initializer) for _ in range(size)]
        
        self.concat_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        x = self.split(x, training)

        x = self.pre_transition(x, training)
        branch = x
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)
        x = self.concat([x, branch])
        x = self.concat_transition(x, training)
        
        return x

class ReverseDarknetBlock(Layer):
    def __init__(self, unit, block_size, block_layer='Conv', activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.block_size = block_size//2
        self.block_layer = block_layer
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = [ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer),
                       ConvLayer(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        
        if self.block_layer == 'Conv':
            self.layers += [ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        elif self.block_layer == 'SPP':
            self.layers += [SPPLayer(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        
        for l in range(1, self.block_layer):
            self.layers = [ConvLayer(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer),
                           ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer),]

    @tf.function
    def call(self, x, training=False):
        x = x
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)

        return x
    
class ReverseCSPDarknetBlock(Layer):
    def __init__(self, unit, size, block_layer=False, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.size = size
        self.block_layer = block_layer
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.branch_transition = ConvLayer(self.unit//2, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.layers = [ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer),
                       ConvLayer(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        
        if self.block_layer == 'Conv':
            self.layers += [ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        elif self.block_layer == 'SPP':
            self.layers += [SPPLayer(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        
        for l in range(1, self.block_layer):
            self.layers = [ConvLayer(self.unit * 2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer),
                           ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer),]

        self.layers = []
        for l in range(self.size):
            self.layers += [ConvLayer(self.unit//2, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)]
            if l==1 and self.block_layer=='SPP':
                self.layers += [SPPLayer(self.unit//2, activate=self.activate, kernel_initializer=self.kernel_initializer)]
            self.layers += [ConvLayer(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)]

        self.concat_transition = ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.concat = Concatenate()

    @tf.function
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
        super().__init__()
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.branch_transition = IdentityLayer()

        if branch_transition:
            self.branch_transition = ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.upsample = UpsampleLayer(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_csp_darknet_block = ReverseCSPDarknetBlock(unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        self.concat = Concatenate()

    @tf.function
    def call(self, branch1, branch2, training=False):
        branch1 = self.upsample(branch1, training)
        branch2 = self.branch_transition(branch2, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_csp_darknet_block(x, training)

        return x

class CSPDarknetDownsampleBlock(Layer):
    def __init__(self, unit, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.downsample = DownsampleLayer(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_csp_darknet_block = ReverseCSPDarknetBlock(self.unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    @tf.function
    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_csp_darknet_block(x, training)

        return x

class DarknetUpsampleBlock(Layer):
    def __init__(self, unit, size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        
        self.branch_transition = ConvLayer(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer) if branch_transition else IdentityLayer()

        self.upsample = UpsampleLayer(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_darknet_block = ReverseDarknetBlock(self.unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        self.concat = Concatenate()

    @tf.function
    def call(self, branch1, branch2, training=False):
        branch1 = self.upsample(branch1, training)
        branch2 = self.branch_transition(branch2, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_darknet_block(x, training)

        return x

class DarknetDownsampleBlock(Layer):
    def __init__(self, unit, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.size = size
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.downsample = DownsampleLayer(self.unit, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.reverse_darket_block = ReverseDarknetBlock(self.unit, size=self.size, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.concat = Concatenate()

    @tf.function
    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)
        x = self.concat([branch1, branch2])
        x = self.reverse_darket_block(x, training)

        return x

class YoloHeadBlock(Layer):
    def __init__(self, unit, scale, col_anchors, num_classes, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.unit = unit
        self.scale = scale
        self.num_classes = num_classes
        self.col_anchors = col_anchors
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv1 = ConvLayer(self.unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = ConvLayer(self.col_anchors * (self.num_classes + 5), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)
        self.reshape = Reshape((*self.scale[::-1], self.col_anchors, self.num_classes + 5))
        
    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.reshape(x)

        return x
    
def get_block(block):
    if block == 'Conv':
        return ConvLayer
    elif block == 'Bottle':
        return BottleBlock
    elif block == 'Residual':
        return ResidualBlockI
    elif block == 'ResidualConv':
        return ResidualBlockC
    elif block == 'CSPBottle':
        return CSPBottleBlock
    elif block == 'CSPResidual':
        return CSPResidualBlockI
    elif block == 'CSPResidualConv':
        return CSPResidualBlockC
    elif block == 'SPP':
        return SPPBlock
    elif block == 'CSPSPP':
        return CSPSPPBlock
