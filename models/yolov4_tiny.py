import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.blocks import *
from models.backbone import CSPDarknet19
from config import *
from utils import anchor_utils
from losses import yolo_loss

    
class YOLO(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES, loss_metric=LOSS_METRIC,
                 iou_threshold=IOU_THRESHOLD, eps=EPS, inf=INF, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.anchors = anchor_utils.get_anchors_xywh(anchors, strides, image_size)
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.scales = (self.image_size // np.array(self.strides)).tolist()
        self.iou_threshold = iou_threshold
        self.col_anchors = len(anchors[0])
        self.eps = eps
        self.inf = inf
        self.kernel_initializer = kernel_initializer

        if loss_metric == 'YOLOv4Loss':
            self.loss_metric = yolo_loss.v4_loss
        elif loss_metric == 'YOLOv3Loss':
            self.loss_metric = yolo_loss.v3_loss

        self.darknet19_tiny = CSPDarknet19(activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv1 = DarknetConv(512, 3, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.conv2 = DarknetConv(256, 1, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        
        self.large_grid_block = GridBlock(512, self.scales[1], self.col_anchors, self.num_classes, 
                                          activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        
        self.medium_upsample_layer = DarknetUpsample(128, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.medium_grid_block = GridBlock(256, self.scales[0], self.col_anchors, self.num_classes, 
                                           activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.concat = Concatenate()

    @tf.function
    def call(self, input, training=False):
        medium_branch, large_branch = self.darknet19_tiny(input, training)
        
        x = self.conv1(large_branch, training)
        large_branch = self.conv2(x, training)
        l_grid = self.large_grid_block(large_branch, training)
                    
        upsampled_medium_branch = self.medium_upsample_layer(large_branch, training)
        medium_branch = self.concat([medium_branch, upsampled_medium_branch])
        m_grid = self.medium_grid_block(medium_branch, training)

        return m_grid, l_grid
    
    @tf.function
    def loss(self, labels, preds, batch_size):
        return self.loss_metric(labels, preds, batch_size, self.anchors, self.strides, self.image_size,
                                self.iou_threshold, self.inf, self.eps)