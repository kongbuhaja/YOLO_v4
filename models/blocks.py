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

        self.concat = ConcatLayer(unit, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        branch = self.branch_transition(x, training)
        
        x = self.block_pre_transition(x, training)
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        x = self.block_post_transition(x, training)
        
        x = self.concat([x, branch], training)

        return x

class CSPBlockA2(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.branch_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.block_pre_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        block = get_block(block)
        self.blocks = [block(unit, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]
        self.block_post_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.concat = ConcatLayer(unit, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        branch = self.branch_transition(x, training)
        
        x = self.block_pre_transition(x, training)
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        x = self.block_post_transition(x, training)
        
        x = self.concat([x, branch], training)

        return x
    
class CSPBlockB(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        self.branch_transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

        block = get_block(block)
        self.blocks = [block(unit, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]

        self.concat = ConcatLayer(unit, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        x = self.transition(x, training)
        branch = self.branch_transition(x, training)
        
        for l in range(len(self.blocks)):
            x = self.blocks[l](x, training)
        
        x = self.concat([x, branch], training)

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

        self.concat = ConcatLayer(unit, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        branch = self.branch_transition(x, training)

        x = self.conv1(x, training)
        x = self.conv2(x, training)
        x = self.spp(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)

        x = self.concat([x, branch], training)

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
    
class TinyCSPBlock(Layer):
    def __init__(self, unit, block, block_size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.split = SplitLayer(2, 1)
        
        self.pre_transition = ConvLayer(unit//2, 3, activate=activate, kernel_initializer=kernel_initializer)
        block = get_block(block)
        self.block = [block(unit//2, kernel_size=3, activate=activate, kernel_initializer=kernel_initializer) for b in range(block_size)]

        self.concat = ConcatLayer(unit, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training):
        x = self.split(x, training)

        x = self.pre_transition(x, training)
        branch = x
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)
        x = self.concat([x, branch], training)
        
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
