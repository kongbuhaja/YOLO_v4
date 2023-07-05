import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.blocks import *
from models.backbone import Darknet19
from config import *
from utils import anchor_utils
from losses import yolo_loss

    
class YOLO(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES,
                 iou_threshold=IOU_THRESHOLD, num_anchors=NUM_ANCHORS, eps=EPS, inf=INF, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.anchors = anchor_utils.get_anchors_xywh(anchors, strides, image_size)
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.scales = (self.image_size // np.array(self.strides)).tolist()
        self.iou_threshold = iou_threshold
        self.num_anchors = num_anchors
        self.eps = eps
        self.inf = inf
        self.kernel_initializer = kernel_initializer

        if LOSS_METRIC == 'YOLOv4Loss':
            self.loss_metric = yolo_loss.v4_loss
        elif LOSS_METRIC == 'YOLOv3Loss':
            self.loss_metric = yolo_loss.v3_loss

        self.darknet19_tiny = Darknet19(activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv_layer = DarknetConv(256, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        
        self.large_grid_block = GridBlock(512, self.scales[1], self.num_anchors, self.num_classes, 
                                          activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        
        self.medium_upsample_layer = DarknetUpsample(128, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.medium_grid_block = GridBlock(256, self.scales[0], self.num_anchors, self.num_classes, 
                                           activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        print('Model: YOLOv3_tiny')
    def call(self, input, training):
        medium_branch, large_branch = self.darknet19_tiny(input, training)
        
        large_branch = self.conv_layer(large_branch, training)
        l_grid = self.large_grid_block(large_branch, training)
                    
        large_branch = self.medium_upsample_layer(large_branch, training)
        medium_branch = Concatenate()([medium_branch, large_branch])
        m_grid = self.medium_grid_block(medium_branch, training)

        return m_grid, l_grid
    
    @tf.function
    def loss(self, labels, preds, batch_size):
        return self.loss_metric(labels, preds, batch_size, self.anchors, self.strides, self.image_size,
                                self.iou_threshold, self.inf, self.eps)