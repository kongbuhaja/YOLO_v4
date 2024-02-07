import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, LeakyReLU, Add, MaxPool2D, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras.regularizers import l2

class Mish(Layer):
    def __init__(self):
        super().__init__()

    @tf.function
    def call(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

class ConvLayer(Layer):
    def __init__(self, unit, kernel_size, strides=1, padding='same', activate='Mish', bn=True, kernel_initializer=glorot, **kwargs):
        super().__init__()        
        self.conv = Conv2D(unit, kernel_size, padding=padding, strides=strides,
                           use_bias=not bn, kernel_regularizer=l2(0.0005),
                           kernel_initializer=kernel_initializer)
        
        self.bn = BatchNormalization() if bn else IdentityLayer()
        self.activate = get_activate(activate)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training)
        x = self.activate(x)
        
        return x

class OSALayer(Layer):
    def __init__(self, unit, growth_rate, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.layers = [ConvLayer(unit//growth_rate, 3, activate=activate, kernel_initializer=kernel_initializer)] * growth_rate

        self.concat = Concatenate()
        self.transition = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        branchs = []
        for l in range(len(self.layers)):
            x = self.layers[l](x, training)
            branchs += [x]
        
        x = self.concat(branchs)
        x = self.transition(x, training)
        
        return x
        
class SplitLayer(Layer):
    def __init__(self, groups, group_id, **kwargs):
        super().__init__()
        self.groups = groups
        self.group_id = group_id
    
    @tf.function
    def call(self, x, training=False):
        return tf.split(x, self.groups, -1)[self.group_id]
    
class SPPLayer(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)
        self.maxpool1 = MaxPool2D(5, 1, 'same')
        self.maxpool2 = MaxPool2D(9, 1, 'same')
        self.maxpool3 = MaxPool2D(13, 1, 'same')
        self.concat = Concatenate()        

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)
        pool1 = self.maxpool1(x)
        pool2 = self.maxpool2(x)
        pool3 = self.maxpool3(x)
        x = self.concat([x, pool1, pool2, pool3])

        return x
    
class UpsampleLayer(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv = ConvLayer(unit, 1, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)
        x = tf.image.resize(x, (x.shape[1]*2, x.shape[2]*2), method='nearest')
        return x
    
class DownsampleLayer(Layer):
    def __init__(self, unit, activate='Mish', kernel_initializer=glorot, **kwargs):
        super().__init__()
        self.conv = ConvLayer(unit, 3, 2, activate=activate, kernel_initializer=kernel_initializer)

    @tf.function
    def call(self, x, training=False):
        x = self.conv(x, training)
        return x
    
class IdentityLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    @tf.function
    def call(self, x, training=False):
        return x

def get_activate(activate):
        if activate == 'Mish':
            return Mish()
        elif activate == 'LeakyReLU':
            return LeakyReLU(alpha=0.1)
        elif activate =='ReLU':
            return ReLU()
        else:
            return IdentityLayer()
    
