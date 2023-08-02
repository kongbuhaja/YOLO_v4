from tensorflow.keras.layers import Layer, MaxPool2D
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from models.blocks import *
from models.layers import *

class CSPP(Layer):
    def __init__(self, size, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv = DarknetConv(32, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.csp_blocks = [
            CSPDarknetBlock(64, 'Resnet', 1, activate=self.activate, kernel_initializer=self.kernel_initializer),
            CSPDarknetBlock(128, 'Resnet', 3, activate=self.activate, kernel_initializer=self.kernel_initializer),
            CSPDarknetBlock(256, 'Resnet', 15, activate=self.activate, kernel_initializer=self.kernel_initializer),
            CSPDarknetBlock(512, 'Resnet', 15, activate=self.activate, kernel_initializer=self.kernel_initializer),
            CSPDarknetBlock(1024, 'Resnet', 7, activate=self.activate, kernel_initializer=self.kernel_initializer)]

        if size > 5:
            self.csp_blocks += [CSPDarknetBlock(1024, 'Resnet', 7, activate=self.activate, kernel_initializer=self.kernel_initializer)]
        if size > 6:
            self.csp_blocks += [CSPDarknetBlock(1024, 'Resnet', 7, activate=self.activate, kernel_initializer=self.kernel_initializer)]

    def call(self, x, training=False):
        x = self.conv(x, training)

        branch = []
        for b in range(len(self.csp_blocks)):
            x = self.csp_blocks[b](x, training)
            if b > 1:
                branch += [x]
            
        return branch

class CSPDarknet53(Layer):
    def __init__(self, activate='Mish', scaled=True, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.conv = DarknetConv(32, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)

        if scaled:
            self.csp_block1 = DarknetBlock(64, 'Resnet', 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        else:
            self.csp_block1 = CSPDarknetBlock(64, 'Resnet', 1, div=1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_block2 = CSPDarknetBlock(128, 'Resnet', 2, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_block3 = CSPDarknetBlock(256, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_block4 = CSPDarknetBlock(512, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.csp_block5 = CSPDarknetBlock(1024, 'Resnet', 4, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, x, training=False):
        x = self.conv(x, training)
        x = self.csp_block1(x, training)
        x = self.csp_block2(x, training)

        small_branch = self.csp_block3(x, training)
        medium_branch = self.csp_block4(small_branch, training)
        large_branch = self.csp_block5(medium_branch, training)

        return small_branch, medium_branch, large_branch

class Darknet53(Layer):
    def __init__(self, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.conv = DarknetConv(32, 3, kernel_initializer=self.kernel_initializer)
    
        self.block1 = DarknetBlock(64, 'Resnet', 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.block2 = DarknetBlock(128, 'Resnet', 2, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.block3 = DarknetBlock(256, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.block4 = DarknetBlock(512, 'Resnet', 8, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.block5 = DarknetBlock(1024, 'Resnet', 4, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, x, training=False):
        x = self.conv(x, training)
        x = self.block1(x, training)
        x = self.block2(x, training)

        small_branch = self.block3(x, training)
        medium_branch = self.block4(small_branch, training)
        large_branch = self.block5(medium_branch, training)

        return small_branch, medium_branch, large_branch
    
class CSPDarknet19(Layer):
    def __init__(self, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.activate = activate
        self.kernel_initialier = kernel_initializer

        self.conv1 = DarknetConv(32, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.conv2 = DarknetConv(64, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.pre_conv1 = DarknetConv(64, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.tiny_conv1 = DarknetTinyBlock(64, 'Conv', 1, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.pre_conv2 = DarknetConv(128, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.tiny_conv2 = DarknetTinyBlock(128, 'Conv', 1, activate=self.activate, kernel_initializer=self.kernel_initialier)
        
        self.pre_conv3 = DarknetConv(256, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.tiny_conv3 = DarknetTinyBlock(256, 'Conv', 1, activate=self.activate, kernel_initializer=self.kernel_initialier)

        self.maxpool = MaxPool2D(2, 2)
        
        self.conv3 = DarknetConv(512, 3, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.conv4 = DarknetConv(256, 1, activate=self.activate, kernel_initializer=self.kernel_initialier)
        self.concat = Concatenate()

    def call(self, x, training=False):
        x = self.conv1(x, training)
        x = self.conv2(x, training)
        
        x = self.pre_conv1(x, training)
        branch = x
        x = self.tiny_conv1(x, training)
        x = self.concat([x, branch])
        x = self.maxpool(x)

        x = self.pre_conv2(x, training)
        branch = x
        x = self.tiny_conv2(x, training)
        x = self.concat([x, branch])
        x = self.maxpool(x)
        
        x = self.pre_conv3(x, training)
        branch = x
        x = self.tiny_conv3(x, training)
        medium_branch = x
        x = self.concat([x, branch])
        x = self.maxpool(x)
        x = self.conv3(x, training)
        large_branch = self.conv4(x, training)

        return medium_branch, large_branch
    
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
        
        self.maxpool2_2 = MaxPool2D(2, 2)
        self.maxpool2_1 = MaxPool2D(2, 1, 'same')

        self.conv7 = DarknetConv(1024, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv8 = DarknetConv(256, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

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
        medium_branch = x
        x = self.maxpool2_2(x)

        x = self.conv6(x, training)
        x = self.maxpool2_1(x)
        x = self.conv7(x, training)
        large_branch = self.conv8(x, training)
        
        return medium_branch, large_branch