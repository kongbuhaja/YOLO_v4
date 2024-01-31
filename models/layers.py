import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, LeakyReLU, Add, MaxPool2D, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras.regularizers import l2

class Mish(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

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

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x)
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

        self.concat = Concatenate()
        self.transition = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        branchs = []
        for l in range(self.growth_rate):
            x = self.layers[l](x, training)
            branchs += [x]
        
        x = self.concat(branchs)
        x = self.transition(x, training)
        
        return x
        
class SplitLayer(Layer):
    def __init__(self, groups, group, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.group = group
    
    @tf.function
    def call(self, x):
        return tf.split(x, self.groups, -1)[self.group]
    
class DarknetResidual(Layer):
    def __init__(self, units, activate='LeakyReLU', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        
        self.conv1 = DarknetConv(self.units[0], 1, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(self.units[1], 3, activate=self.activate, kernel_initializer=self.kernel_initializer)
        self.add = Add()
    
    @tf.function
    def call(self, x, training=False):
        branch = x
        x = self.conv1(x, training)
        x = self.conv2(x, training)

        return self.add([x, branch])
    
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

    @tf.function
    def call(self, x, training=False):
        pool1 = self.maxpool1(x)
        pool2 = self.maxpool2(x)
        pool3 = self.maxpool3(x)
        x = self.concat([pool1, pool2, pool3, x])
        x = self.conv(x, training)

        return x
    
class DarknetUpsample(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.unit, 1, activate=self.activate, kernel_initializer=self.kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)
        x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='nearest')
        return x
    
class DarknetDownsample(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.unit = unit
        self.activate = activate
        self.kernel_initializer = kernel_initializer
        self.conv = DarknetConv(self.unit, 3, 2, activate=self.activate, kernel_initializer=self.kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)
        return x
    
class Identity(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def call(self, x, training=False):
        return x