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
    
class DarknetConv(Layer):
    def __init__(self, units, kernel_size, strides=1, padding='same', activate='Mish', bn=True, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activate = activate
        self.bn = bn
        self.kernel_initializer = kernel_initializer
        
        self.conv = Conv2D(self.units, self.kernel_size, padding=self.padding, strides=self.strides,
                           use_bias=not self.bn, kernel_regularizer=l2(0.0005),
                           kernel_initializer=self.kernel_initializer)
        if self.bn:
            self.bn = BatchNormalization()
        
        if self.activate == 'Mish':
            self.activate = Mish()
        elif self.activate == 'LeakyReLU':
            self.activate = LeakyReLU(alpha=0.1)

    def call(self, input, training=False):
        x = self.conv(input)
        if self.bn:
            x = self.bn(x, training)
        if self.activate:
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
    
    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)
        # x = self.drop_block(x, training)

        return Add()([input, x])
    
class SPP(Layer):
    def __init__(self, units, activate='LeakyReLU', kernel_initializer='glorot', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.units, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        pool1 = MaxPool2D(13, 1, 'same')(input)
        pool2 = MaxPool2D(9, 1, 'same')(input)
        pool3 = MaxPool2D(5, 1, 'same')(input)
        x = Concatenate()([pool1, pool2, pool3, input])
        x = self.conv(x, training)

        return x
    
class DarknetUpsample(Layer):
    def __init__(self, units, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.units, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)
        x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='nearest')
        return x
    
class DarknetDownsample(Layer):
    def __init__(self, units, activate='Mish', kernel_initializer='glorot', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.units, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        x = self.conv(input, training)
        return x