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

        self.csp_resblock1 = CSPDarknetBlock(64, 'Resnet', 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
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

        self.darknetConv1 = DarknetConv(16, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.darknetConv2 = DarknetConv(32, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.darknetConv3 = DarknetConv(64, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.darknetConv4 = DarknetConv(128, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.darknetConv5 = DarknetConv(256, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.darknetConv6 = DarknetConv(512, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.darknetConv7 = DarknetConv(1024, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.darknetConv1(input, training)
        x = MaxPool2D(2, 2)(x)

        x = self.darknetConv2(x, training)
        x = MaxPool2D(2, 2)(x)

        x = self.darknetConv3(x, training)
        x = MaxPool2D(2, 2)(x)

        x = self.darknetConv4(x, training)
        x = MaxPool2D(2, 2)(x)

        x = self.darknetConv5(x, training)
        m_route = x
        x = MaxPool2D(2, 2)(x)

        x = self.darknetConv6(x, training)
        x = MaxPool2D(2, 1, 'same')(x)

        l_route = self.darknetConv7(x, training)
        
        return m_route, l_route