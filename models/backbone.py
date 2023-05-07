from tensorflow.keras.layers import Layer, MaxPool2D
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.common import *

class CSPDarknet53(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(32, 3, activate='Mish', kernel_initializer=self.kernel_initializer)

        self.csp_resblock1 = CSPDarknetResidualBlock([64, 32], 1, activate='Mish', kernel_initializer=self.kernel_initializer)
        self.csp_resblock2 = CSPDarknetResidualBlock([64, 64], 2, activate='Mish', kernel_initializer=self.kernel_initializer)
        self.csp_resblock3 = CSPDarknetResidualBlock([128, 128], 8, activate='Mish', kernel_initializer=self.kernel_initializer)
        self.csp_resblock4 = CSPDarknetResidualBlock([256, 256], 8, activate='Mish', kernel_initializer=self.kernel_initializer)
        self.csp_resblock5 = CSPDarknetResidualBlock([512, 512], 4, activate='Mish', kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)

        x = self.csp_resblock1(x, training)
        x = self.csp_resblock2(x, training)
        small_branch = self.csp_resblock3(x, training)
        medium_branch = self.csp_resblock4(small_branch, training)
        large_branch = self.csp_resblock5(medium_branch, training)

        return small_branch, medium_branch, large_branch

class Darknet53(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer
        
        self.conv = DarknetConv(32, 3, kernel_initializer=self.kernel_initializer)
    
        self.resblock1 = DarknetResidualBlock([64, 32], 1)
        self.resblock2 = DarknetResidualBlock([128, 64], 2)
        self.resblock3 = DarknetResidualBlock([256, 128], 8)
        self.resblock4 = DarknetResidualBlock([512, 256], 8)
        self.resblock5 = DarknetResidualBlock([1024, 512], 4)

    def call(self, input, training=False):
        x = self.conv(input, training)
        
        x = self.resblock1(x, training)
        x = self.resblock2(x, training)
        small_branch = self.resblock3(x, training)
        medium_branch = self.resblock4(small_branch, training)
        large_branch = self.resblock5(medium_branch, training)

        return small_branch, medium_branch, large_branch
    
class Darknet19(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.darknetConv1 = DarknetConv(16, 3, kernel_initializer=self.kernel_initializer)
        
        self.darknetConv2 = DarknetConv(32, 3, kernel_initializer=self.kernel_initializer)
        
        self.darknetConv3 = DarknetConv(64, 3, kernel_initializer=self.kernel_initializer)

        self.darknetConv4 = DarknetConv(128, 3, kernel_initializer=self.kernel_initializer)
        
        self.darknetConv5 = DarknetConv(256, 3, kernel_initializer=self.kernel_initializer)
        
        self.darknetConv6 = DarknetConv(512, 3, kernel_initializer=self.kernel_initializer)

        self.darknetConv7 = DarknetConv(1024, 3, kernel_initializer=self.kernel_initializer)

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