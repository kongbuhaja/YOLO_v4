import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam

class Optimizer():
    def __init__(self, opt):
        if opt['name'] in ['sgd', 'SGD']:
            self.optimizer = SGD(momentum=opt['momentum'],
                                 weight_decay=opt['decay'])
        elif opt['name'] in ['adam', 'Adam']:
            self.optimizer = Adam(beta_1=opt['momentum'],
                                  weight_decay=opt['decay'])

        self.lr = float(self.optimizer.lr.numpy())

    def __call__(self, zip_data):
        self.optimizer.apply_gradients(zip_data)

    def assign_lr(self, lr):
        self.optimizer.lr.assign(lr)
        self.lr = lr
