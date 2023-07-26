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

class DarknetOSA(Layer):
    def __init__(self, unit, growth_rate, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.growth_rate = growth_rate
        self.activate = activate
        self.kernel_initializer = kernel_initializer

        self.layers = [DarknetConv(self.unit//growth_rate, 3, activate=self.activate, kernel_initializer=self.kernel_initializer) for _ in range(self.growth_rate)]
        self.features = []

        self.concat = Concatenate()
        self.transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        for l in range(self.growth_rate):
            input = self.layers[l](input, training)
            self.features += [input]
        
        x = self.concat(self.features)
        x = self.transition(x, training)
        
class SplitLayer(Layer):
    def __init__(self, groups, group, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.group = group
    
    def call(self, input):
        return tf.split(input, self.groups, -1)[self.group]
    
class DarknetResidual(Layer):
    def __init__(self, units, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.conv1 = DarknetConv(self.units[0], 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.units[1], 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.add = Add()
    
    def call(self, input, training=False):
        x = self.conv1(input, training)
        x = self.conv2(x, training)

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