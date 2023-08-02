import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import *
from config import *

class CSPPANSPP(Layer):
    def __init__(self, unit, layer_size, block_size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.layer_size = layer_size
        self.block_size = block_size
        self.branch_transition = branch_transition
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.up_layers = [ReverseCSPDarknetBlock(self.unit, size=2, block_layer='SPP',
                                               activate=self.activate, kernel_initializer=self.kernel_initializer)]
        for l in range(self.layer_size - 3):
            self.up_layers += [CSPDarknetUpsampleBlock(self.unit, size=self.block_size, branch_transition=self.branch_transition,
                                                       activate=self.activate, kernel_initializer=self.kernel_initializer)]
        for u in range(2):
            self.up_layers += [CSPDarknetUpsampleBlock(self.unit//2**(u+1), size=self.block_size,branch_transition=self.branch_transition,
                                                       activate=self.activate, kernel_initializer=self.kernel_initializer)]
            
        self.down_layers = [CSPDarknetDownsampleBlock(self.unit//2, size=self.block_size,
                                                      activate=self.activate, kernel_initializer=self.kernel_initializer)]
        for l in range(self.layer_size - 2):
            self.down_layers += [CSPDarknetDownsampleBlock(self.unit, size=self.block_size,
                                                           activate=self.activate, kernel_initializer=self.kernel_initializer)]

    @tf.function
    def call(self, x, training):
        branchs = [self.up_layers[0](x[-1], training)]
        
        for l in range(1, self.layer_size):
            branchs += [self.up_layers[l](branchs[-1], x[-l-1], training)]
        
        out_branchs = [branchs[-1]]
        for l in range(self.layer_size - 1):
            out_branchs += [self.down_layers[l](out_branchs[-1], branchs[-l-2], training)]

        return out_branchs
    
class PANSPP(Layer):
    def __init__(self, unit, layer_size, block_size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.layer_size = layer_size
        self.block_size = block_size
        self.branch_transition = branch_transition
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.up_layers = [ReverseDarknetBlock(self.unit, size=2, block_layer='SPP',
                                              activate=self.activate, kernel_initializer=self.kernel_initializer)]
        for l in range(self.layer_size - 3):
            self.up_layers += [DarknetUpsampleBlock(self.unit, self.block_size, branch_transition=self.branch_transition,
                                                    activate=self.activate, kernel_initializer=self.kernel_initializer)]
        for u in range(2):
            self.up_layers += [DarknetUpsampleBlock(self.unit//2**(u+1), size=self.block_size,branch_transition=self.branch_transition,
                                                    activate=self.activate, kernel_initializer=self.kernel_initializer)]
        
        self.down_layers = [DarknetDownsampleBlock(self.unit//2, size=self.block_size,
                                                      activate=self.activate, kernel_initializer=self.kernel_initializer)]
        for l in range(self.layer_size - 2):
            self.down_layers += [DarknetDownsampleBlock(self.unit, size=self.block_size,
                                                           activate=self.activate, kernel_initializer=self.kernel_initializer)]
    
    @tf.function
    def call(self, x, training):
        branchs = [self.up_layers[0](x[-1], training)]
        
        for l in range(1, self.layer_size):
            branchs += [self.up_layers[l](branchs[-1], x[-l-1], training)]
        
        out_branchs = [branchs[-1]]
        for l in range(self.layer_size - 1):
            out_branchs += [self.down_layers[l](out_branchs[-1], branchs[-l-2], training)]

        return out_branchs

class CSPFPNSPP(Layer):
    def __init__(self, unit, layer_size, block_size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.layer_size = layer_size
        self.block_size = block_size
        self.branch_transition = branch_transition
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = []
        for l in range(self.layer_size - 1):
            self.layers += [CSPDarknetUpsampleBlock(self.unit//(l+1), self.block_size, branch_transition=self.branch_transition,
                                                    activate=self.activate, kernel_initializer=self.kernel_initializer)]
            
    @tf.function
    def call(self, x, training):
        branchs = [x[-1]]

        for l in range(self.layer_size - 1):
            branchs += [self.layers[l](branchs[-1], x[-l-2], training)]

        return branchs[::-1]

class FPNSPP(Layer):
    def __init__(self, unit, layer_size, block_size, branch_transition=True, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.layer_size = layer_size
        self.block_size = block_size
        self.branch_transition = branch_transition
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = []
        for l in range(self.layer_size - 1):
            self.layers += [DarknetUpsampleBlock(self.unit//(l+1), self.block_size, branch_transition=self.branch_transition,
                                                 activate=self.activate, kernel_initializer=self.kernel_initializer)]
    
    @tf.function
    def call(self, x, training):
        branchs = [x[-1]]
        
        for l in range(self.layer_size - 1):
            branchs += [self.layers[l](branchs[-1], x[-l-2], training)]

        return branchs[::-1]

class tinyFPN(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.medium_upsample_layer = DarknetUpsample(self.unit//2, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.concat = Concatenate()

    @tf.function
    def call(self, x, training):
        branchs = [x[-1]]

        upsampled_medium_branch = self.medium_upsample_layer(branchs[-1], training)
        branchs += [self.concat([upsampled_medium_branch, x[0]])]

        return branchs[::-1]
