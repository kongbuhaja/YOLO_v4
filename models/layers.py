import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, Add, MaxPool2D, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras.regularizers import l2

class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input):
        return input * tf.math.tanh(tf.math.softplus(input))
    
class CSPDarknetConv(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.pre_conv = DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv = DarknetConv(self.unit//2, 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        
        self.concat = Concatenate()
        self.transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training):
        x = self.pre_conv(input, training)
        short_cut = x
        x = self.conv(x, training)
        x = self.concat([x, short_cut])
        x = self.transition(x, training)

        return x

class DarknetConv(Layer):
    def __init__(self, unit, kernel_size, strides=1, padding='same', activate='Mish', bn=True, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activate = Identity()
        self.bn = Identity()
        self.kernel_initializer = kernel_initializer
        
        self.conv = Conv2D(self.unit, self.kernel_size, padding=self.padding, strides=self.strides,
                           use_bias=not self.bn, kernel_regularizer=l2(0.0005),
                           kernel_initializer=self.kernel_initializer)
        
        if bn:
            self.bn = BatchNormalization()
        
        if activate == 'Mish':
            self.activate = Mish()
        elif activate == 'LeakyReLU':
            self.activate = LeakyReLU(alpha=0.1)

    def call(self, input, training=False):
        x = self.conv(input)
        x = self.bn(x, training)
        x = self.activate(x)
        
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
        self.add = Add()
    
    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        # x = self.drop_block(x, training)

        return self.add([input, x])
    
class SPP(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.maxpool1 = MaxPool2D(13, 1, 'same')
        self.maxpool2 = MaxPool2D(9, 1, 'same')
        self.maxpool3 = MaxPool2D(5, 1, 'same')
        self.concat = Concatenate()        
        self.conv = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        pool1 = self.maxpool1(input)
        pool2 = self.maxpool2(input)
        pool3 = self.maxpool3(input)
        x = self.concat([pool1, pool2, pool3, input])
        x = self.conv(x, training)

        return x
    
class DarknetUpsample(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)
        x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='nearest')
        return x
    
class DarknetDownsample(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)
        return x
    
class Identity(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input, training=False):
        return input