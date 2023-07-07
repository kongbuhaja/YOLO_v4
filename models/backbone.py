from tensorflow.keras.layers import Layer, MaxPool2D
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import *

class CSPDarknet53(Layer):
    def __init__(self, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv = DarknetConv(32, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.csp_resblock1 = CSPDarknetBlock(64, 'Resnet', 1, div=1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_resblock2 = CSPDarknetBlock(128, 'Resnet', 2, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_resblock3 = CSPDarknetBlock(256, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_resblock4 = CSPDarknetBlock(512, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_resblock5 = CSPDarknetBlock(1024, 'Resnet', 4, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)
        x = self.csp_resblock1(x, training)
        x = self.csp_resblock2(x, training)

        small_branch = self.csp_resblock3(x, training)
        medium_branch = self.csp_resblock4(small_branch, training)
        large_branch = self.csp_resblock5(medium_branch, training)

        return small_branch, medium_branch, large_branch
    
class CSPDarknet19(Layer):
    def __init__(self, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initialier = kernel_initializer

        self.conv1 = DarknetConv(32, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.conv2 = DarknetConv(64, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.pre_conv1 = DarknetConv(64, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.csp_conv1 = CSPDarknetConv(64, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.pre_conv2 = DarknetConv(128, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.csp_conv2 = CSPDarknetConv(128, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.pre_conv3 = DarknetConv(256, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.csp_conv3 = CSPDarknetConv(256, activate=self.activate, kernel_initializer=self.kernel_initialier)

        self.maxpool = MaxPool2D(2, 2)
        self.concat = Concatenate()

    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        
        x = self.pre_conv1(x, training)
        short_cut = x
        x = self.csp_conv1(x, training)
        x = self.concat([x, short_cut])
        x = self.maxpool(x)

        x = self.pre_conv2(x, training)
        short_cut = x
        x = self.csp_conv2(x, training)
        x = self.concat([x, short_cut])
        x = self.maxpool(x)
        
        x = self.pre_conv3(x, training)
        short_cut = x
        x = self.csp_conv3(x, training)
        medium_branch = x
        x = self.concat([x, short_cut])
        large_branch = self.maxpool(x)

        return medium_branch, large_branch

class Darknet53(Layer):
    def __init__(self, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.conv = DarknetConv(32, 3, kernel_initializer=self.kernel_initializer)
    
        self.resblock1 = DarknetBlock(64, 'Resnet', 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.resblock2 = DarknetBlock(128, 'Resnet', 2, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.resblock3 = DarknetBlock(256, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.resblock4 = DarknetBlock(512, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.resblock5 = DarknetBlock(1024, 'Resnet', 4, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)
        x = self.resblock1(x, training)
        x = self.resblock2(x, training)

        small_branch = self.resblock3(x, training)
        medium_branch = self.resblock4(small_branch, training)
        large_branch = self.resblock5(medium_branch, training)

        return small_branch, medium_branch, large_branch
    
class Darknet19(Layer):
    def __init__(self, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv1 = DarknetConv(16, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(32, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv3 = DarknetConv(64, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv4 = DarknetConv(128, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv5 = DarknetConv(256, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv6 = DarknetConv(512, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.maxpool1 = MaxPool2D(2, 2)
        self.maxpool2 = MaxPool2D(2, 1, 'same')

    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.maxpool1(x)

        x = self.conv2(x, training)
        x = self.maxpool1(x)

        x = self.conv3(x, training)
        x = self.maxpool1(x)

        x = self.conv4(x, training)
        x = self.maxpool1(x)

        x = self.conv5(x, training)
        medium_branch = x
        x = self.maxpool1(x)

        x = self.conv6(x, training)
        large_branch = self.maxpool2(x)
        
        return medium_branch, large_branch