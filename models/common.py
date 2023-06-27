import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Layer, LeakyReLU, Add, Reshape, MaxPool2D, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras.regularizers import l2

class DarknetUpsample(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input, training=False):
        return tf.image.resize(input, (input.shape[1]*2, input.shape[2]*2), method='nearest')
    
class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        return input * tf.math.tanh(tf.math.softplus(input))

class DarknetConv(Layer):
    def __init__(self, units, kernel_size, strides=1, padding='same', kernel_initializer=glorot, activate='LeakyReLU', bn=True, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.activate = activate
        self.bn = bn
        self.padding = padding
        
        self.conv = Conv2D(self.units, self.kernel_size, padding=self.padding, strides=self.strides,
                           use_bias=not self.bn, kernel_regularizer=l2(0.0005),
                           kernel_initializer=self.kernel_initializer)
        if self.bn:
            self.batchN = BatchNormalization()
        
        if self.activate == 'Mish':
            self.activateF = Mish()
        elif self.activate == 'LeakyReLU':
            self.activateF = LeakyReLU(alpha=0.1)

    def call(self, input, training=False):
        x = self.conv(input)
        if self.bn:
            x = self.batchN(x, training)

        if self.activate:
            x = self.activateF(x)
        
        return x

class DarknetResidual(Layer):
    def __init__(self, units, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.conv1 = DarknetConv(self.units[0], 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.units[1], 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        # self.drop_block = DropBlock(0.9, 3)
    
    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        # x = self.drop_block(x, training)

        return Add()([input, x])

class DarknetResidualBlock(Layer):
    def __init__(self, units, resblock_num, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.resblock_num = resblock_num
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.pre_conv = DarknetConv(self.units, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.res_blocks = [DarknetResidual([self.units//2, self.units], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.resblock_num)]


    def call(self, input, training=False):
        x = self.pre_conv(input, training)

        for r in range(len(self.res_blocks)):
            x = self.res_blocks[r](x, training)
        
        return x

class CSPDarknetResidualBlock(Layer):
    def __init__(self, units, resblock_num, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.resblock_num = resblock_num
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        div = 2 if resblock_num // 2 > 0 else 1

        self.pre_conv = DarknetConv(self.units, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)     
        
        self.res_pre_conv = DarknetConv(self.units//div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.res_blocks = [DarknetResidual([self.units//2, self.units//div], activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.resblock_num)]
        self.res_transition = DarknetConv(self.units//div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        self.short_cut_transition = DarknetConv(self.units//div, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

        self.concat_transition = DarknetConv(self.units, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
    
    def call(self, input, training=False):
        x = self.pre_conv(input, training)
        short_cut = x

        x = self.res_pre_conv(x, training)
        for r in range(len(self.res_blocks)):
            x = self.res_blocks[r](x, training)
        x = self.res_transition(x, training)

        short_cut = self.short_cut_transition(short_cut, training)

        x = Concatenate()([x, short_cut])

        x = self.concat_transition(x, training)

        return x

class SPP(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input, training=False):
        pool1 = MaxPool2D(13, 1, 'same')(input)
        pool2 = MaxPool2D(9, 1, 'same')(input)
        pool3 = MaxPool2D(5, 1, 'same')(input)
        x = Concatenate()([pool1, pool2, pool3, input])

        return x

class SPPBlock(Layer):
    def __init__(self, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = kernel_initializer

        self.pre_conv1 = DarknetConv(512, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.pre_conv2 = DarknetConv(1024, 3, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.pre_conv3 = DarknetConv(512, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        
        self.spp = SPP()

        self.post_conv1 = DarknetConv(512, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.post_conv2 = DarknetConv(1024, 3, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.post_conv3 = DarknetConv(512, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.pre_conv1(input, training)
        x = self.pre_conv2(x, training)
        x = self.pre_conv3(x, training)
        
        x = self.spp(x)

        x = self.post_conv1(x, training)
        x = self.post_conv2(x, training)
        x = self.post_conv3(x, training)
        
        return x

class DarknetConv5(Layer):
    def __init__(self, unit, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit=unit
        self.kernel_initializer = kernel_initializer

        self.conv1 = DarknetConv(self.unit, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.unit * 2, 3, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv3 = DarknetConv(self.unit, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv4 = DarknetConv(self.unit * 2, 3, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv5 = DarknetConv(self.unit, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        x = self.conv5(x, training)

        return x
    
class ConcatConv(Layer):
    def __init__(self, unit, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.kernel_initializer = kernel_initializer

        self.conv = DarknetConv5(unit, kernel_initializer=self.kernel_initializer)

    def call(self, branch1, branch2, training=False):
        x = Concatenate()([branch1, branch2])
        x = self.conv(x, training)

        return x

class UpsampleConcat(Layer):
    def __init__(self, unit, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.kernel_initializer = kernel_initializer

        self.conv1 = DarknetConv(self.unit, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.unit, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        
        self.upsample = DarknetUpsample()
        self.concat_conv = ConcatConv(self.unit, kernel_initializer=self.kernel_initializer)
    
    def call(self, branch1, branch2, training=False):
        branch1 = self.conv1(branch1, training)
        branch2 = self.conv2(branch2, training)
        branch2 = self.upsample(branch2, training)       

        x = self.concat_conv(branch1, branch2, training)

        return x

class DownsampleConcat(Layer):
    def __init__(self, unit, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.kernel_initializer = kernel_initializer

        self.downsample = DarknetConv(self.unit, 3, 2, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

        self.concat_conv = ConcatConv(self.unit, kernel_initializer=self.kernel_initializer)
    
    def call(self, branch1, branch2, training=False):
        branch1 = self.downsample(branch1, training)

        x = self.concat_conv(branch1, branch2, training)

        return x

class GridOut(Layer):
    def __init__(self, unit, scale, num_anchors, num_classes, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.scale = scale
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.kernel_initializer = kernel_initializer

        self.conv1 = DarknetConv(self.unit, 3, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.num_anchors * (self.num_classes + 5), 1, activate=False, bn=False, kernel_initializer=self.kernel_initializer)
        
    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        x = Reshape((self.scale, self.scale, self.num_anchors, self.num_classes + 5))(x)

        return x