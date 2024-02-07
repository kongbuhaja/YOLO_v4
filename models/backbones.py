from tensorflow.keras.layers import Layer, MaxPool2D
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import *
from models.layers import *

class CSPP(Layer):
    def __init__(self, unit, size, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        units = [unit*2, unit*2**2, unit*2**3, unit*2**4, unit*2**5, unit*2**5, unit*2**5]
        block_sizes = [1, 3, 15, 15, 7, 7, 7]

        self.conv = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)

        self.blocks = []
        for s in range(size):
            self.blocks += [[ConvLayer(units[s], 3, 2, activate=activate, kernel_initializer=kernel_initializer),
                             CSPBlockA(units[s], 'CSPResidual', block_sizes[s], activate=activate, kernel_initializer=kernel_initializer)]]

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)

        branch = []
        for b, (downsample, block) in enumerate(self.blocks):
            x = downsample(x, training)
            x = block(x, training)
            if b > 1:
                branch += [x]
            
        return branch

class CSPDarknet53(Layer):
    def __init__(self, unit, csp=True, activate='Mish', kernel_initializer=glorot):
        super().__init__()        
        self.conv = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
    
        self.downsample1 = ConvLayer(unit*2, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block1 = PlainBlockA(unit*2, 'Residual', 1, activate=activate, kernel_initializer=kernel_initializer) if csp else \
                      CSPBlockA2(unit*2, 'Residual', 1, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample2 = ConvLayer(unit*2**2, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block2 = CSPBlockA(unit*2**2, 'CSPResidual', 2, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample3 = ConvLayer(unit*2**3, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block3 = CSPBlockA(unit*2**3, 'CSPResidual', 8, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample4 = ConvLayer(unit*2**4, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block4 = CSPBlockA(unit*2**4, 'CSPResidual', 8, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample5 = ConvLayer(unit*2**5, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block5 = CSPBlockA(unit*2**5, 'CSPResidual', 4, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)

        x = self.downsample1(x, training)
        x = self.block1(x, training)

        x = self.downsample2(x, training)
        x = self.block2(x, training)

        x = self.downsample3(x, training)
        small_branch = self.block3(x, training)

        x = self.downsample4(small_branch, training)
        medium_branch = self.block4(x, training)

        x = self.downsample5(medium_branch, training)
        large_branch = self.block5(x, training)

        return small_branch, medium_branch, large_branch       

class Darknet53(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot):
        super().__init__()        
        self.conv = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
    
        self.downsample1 = ConvLayer(unit*2, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block1 = PlainBlockA(unit*2, 'Residual', 1, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample2 = ConvLayer(unit*2**2, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block2 = PlainBlockA(unit*2**2, 'Residual', 2, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample3 = ConvLayer(unit*2**3, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block3 = PlainBlockA(unit*2**3, 'Residual', 8, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample4 = ConvLayer(unit*2**4, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block4 = PlainBlockA(unit*2**4, 'Residual', 8, activate=activate, kernel_initializer=kernel_initializer)

        self.downsample5 = ConvLayer(unit*2**5, 3, 2, activate=activate, kernel_initializer=kernel_initializer)
        self.block5 = PlainBlockA(unit*2**5, 'Residual', 4, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)

        x = self.downsample1(x, training)
        x = self.block1(x, training)

        x = self.downsample2(x, training)
        x = self.block2(x, training)

        x = self.downsample3(x, training)
        small_branch = self.block3(x, training)

        x = self.downsample4(small_branch, training)
        medium_branch = self.block4(x, training)

        x = self.downsample5(medium_branch, training)
        large_branch = self.block5(x, training)

        return small_branch, medium_branch, large_branch
    
class CSPDarknet19(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot):
        super().__init__()
        self.activate = activate
        self.kernel_initialier = kernel_initializer

        self.conv1 = ConvLayer(unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.conv2 = ConvLayer(unit*2, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.transition1 = ConvLayer(unit*2, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.block1 = TinyCSPBlock(unit*2, 'Conv', 1, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.transition2 = ConvLayer(unit*2**2, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.block2 = TinyCSPBlock(unit*2**2, 'Conv', 1, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.transition3 = ConvLayer(unit*2**3, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.block3 = TinyCSPBlock(unit*2**3, 'Conv', 1, activate=self.activate, kernel_initializer=self.kernel_initialier)

        self.conv3 = ConvLayer(unit*2**4, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)

        self.maxpool = MaxPool2D(2, 2)
        self.concat = Concatenate()
    
    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        
        branch = self.transition1(x, training)
        x = self.block1(branch, training)
        x = self.concat([x, branch])
        x = self.maxpool(x)

        branch = self.transition2(x, training)
        x = self.block2(branch, training)
        x = self.concat([x, branch])
        x = self.maxpool(x)
        
        branch = self.transition3(x, training)
        medium_branch = self.block3(branch, training)

        x = self.concat([medium_branch, branch])
        x = self.maxpool(x)
        large_branch = self.conv3(x, training)

        return medium_branch, large_branch
    
class Darknet19(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot):
        super().__init__()
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv1 = ConvLayer(unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = ConvLayer(unit*2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv3 = ConvLayer(unit*2**2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv4 = ConvLayer(unit*2**3, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv5 = ConvLayer(unit*2**4, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv6 = ConvLayer(unit*2**5, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv7 = ConvLayer(unit*2**6, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.maxpool2_2 = MaxPool2D(2, 2)
        self.maxpool2_1 = MaxPool2D(2, 1, 'same')

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)

        x = self.maxpool2_2(x)
        x = self.conv2(x, training)

        x = self.maxpool2_2(x)
        x = self.conv3(x, training)

        x = self.maxpool2_2(x)
        x = self.conv4(x, training)

        x = self.maxpool2_2(x)
        medium_branch = self.conv5(x, training)

        x = self.maxpool2_2(medium_branch)
        x = self.conv6(x, training)

        x = self.maxpool2_1(x)
        large_branch = self.conv7(x, training)
        
        return medium_branch, large_branch
    
class Darknet19_v2(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot):
        super().__init__()
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv1 = ConvLayer(unit, 3, activate=activate, kernel_initializer=kernel_initializer)
        
        self.conv2 = ConvLayer(unit*2, 3, activate=activate, kernel_initializer=kernel_initializer)
        
        self.conv3_1 = ConvLayer(unit*2**2, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.conv3_2 = PlainBlockA(unit*2, 'Bottle', 1, activate=activate, kernel_initializer=kernel_initializer)
        
        self.conv4_1 = ConvLayer(unit*2**3, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.conv4_2 = PlainBlockA(unit*2**2, 'Bottle', 1, activate=activate, kernel_initializer=kernel_initializer)
        
        self.conv5_1 = ConvLayer(unit*2**4, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.conv5_2 = PlainBlockA(unit*2**3, 'Bottle', 2, activate=activate, kernel_initializer=kernel_initializer)
        
        self.conv6_1 = ConvLayer(unit*2**5, 3, activate=activate, kernel_initializer=kernel_initializer)
        self.conv6_2 = PlainBlockA(unit*2**4, 'Bottle', 2, activate=activate, kernel_initializer=kernel_initializer)        
        
        self.maxpool2_2 = MaxPool2D(2, 2)

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)

        x = self.maxpool2_2(x)
        x = self.conv2(x, training)

        x = self.maxpool2_2(x)
        x = self.conv3_1(x, training)
        x = self.conv3_2(x, training)

        x = self.maxpool2_2(x)
        x = self.conv4_1(x, training)
        x = self.conv4_2(x, training)

        x = self.maxpool2_2(x)
        x = self.conv5_1(x, training)
        medium_branch = self.conv5_2(x, training)

        x = self.maxpool2_2(medium_branch)
        x = self.conv6_1(x, training)
        large_branch = self.conv6_2(x, training)
        
        return medium_branch, large_branch
    
class Darknet19_v2_tiny(Layer):
    def __init__(self, unit, activate='LeakyReLU', kernel_initializer=glorot):
        super().__init__()
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv1 = ConvLayer(unit, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = ConvLayer(unit*2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv3 = ConvLayer(unit*2**2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv4 = ConvLayer(unit*2**3, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv5 = ConvLayer(unit*2**4, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv6 = ConvLayer(unit*2**5, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv7 = ConvLayer(unit*2**6, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
        self.maxpool2_2 = MaxPool2D(2, 2)
        self.maxpool2_1 = MaxPool2D(2, 1, 'same')

    @tf.function
    def call(self, x, training=False):
        x = self.conv1(x, training)

        x = self.maxpool2_2(x)
        x = self.conv2(x, training)

        x = self.maxpool2_2(x)
        x = self.conv3(x, training)

        x = self.maxpool2_2(x)
        x = self.conv4(x, training)

        x = self.maxpool2_2(x)
        x = self.conv5(x, training)

        x = self.maxpool2_2(x)
        x = self.conv6(x, training)

        x = self.maxpool2_1(x)
        x = self.conv7(x, training)
        
        return x