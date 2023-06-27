import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.common import *
from models.backbone import Darknet53
import numpy as np
from config import *
from utils import anchor_utils
from losses import yolo_loss


class YOLO(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES,
                 iou_threshold=IOU_THRESHOLD, num_anchors=NUM_ANCHORS, eps=EPS, inf=INF, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.image_size = IMAGE_SIZE
        self.strides = strides
        self.anchors = anchor_utils.get_anchors_xywh(anchors, self.strides, image_size)
        self.scales = image_size//np.array(self.strides)
        self.iou_threshold = iou_threshold
        self.num_anchors = num_anchors
        self.kernel_initializer = kernel_initializer
        self.eps = eps
        self.inf = inf

        if LOSS_METRIC == 'YOLOv4Loss':
            self.loss_metric = yolo_loss.v4_loss
        elif LOSS_METRIC == 'YOLOv3Loss':
            self.loss_metric = yolo_loss.v3_loss

        self.darknet53 = Darknet53(kernel_initializer=self.kernel_initializer)
        
        self.conv_layers  = DarknetConv5(512, kernel_initializer=self.kernel_initializer)
        
        self.large_grid_layer = GridOut(1024, self.scales[2], self.num_anchors, self.num_classes, kernel_initializer=self.kernel_initializer)
        
        self.large_upsample_layers = [DarknetConv(256, 1, kernel_initializer=self.kernel_initializer),
                                      DarknetUpsample()]
        
        self.medium_concat_layer  = ConcatConv(256, kernel_initializer=self.kernel_initializer)
        self.medium_grid_layer = GridOut(512, self.scales[1], self.num_anchors, self.num_classes, kernel_initializer=self.kernel_initializer)
        
        self.medium_upsample_layers = [DarknetConv(128, 1, kernel_initializer=self.kernel_initializer),
                                       DarknetUpsample()]
        
        self.small_concat_layer = ConcatConv(128, kernel_initializer=self.kernel_initializer)
        self.small_grid_layer = GridOut(256, self.scales[0], self.num_anchors, self.num_classes, kernel_initializer=self.kernel_initializer)
    
    def call(self, input, training=False):
        small_branch, medium_branch, large_branch = self.darknet53(input, training)
        
        # l_grid
        large_branch = self.conv_layers(large_branch, training)
        
        l_grid = self.large_grid_layer(large_branch, training)

        # m_grid
        for i in range(len(self.large_upsample_layers)):
            large_branch = self.large_upsample_layers[i](large_branch, training)
        
        medium_branch = self.medium_concat_layer(large_branch, medium_branch, training)
        m_grid = self.medium_grid_layer(medium_branch, training)

        # s_grid
        for i in range(len(self.medium_upsample_layers)):
            medium_branch = self.medium_upsample_layers[i](medium_branch, training)
        
        small_branch = self.small_concat_layer(medium_branch, small_branch, training)
        s_grid = self.small_grid_layer(small_branch, training)

        return s_grid, m_grid, l_grid
    
    @tf.function
    def loss(self, labels, preds, batch_size):
        return self.loss_metric(labels, preds, batch_size, self.anchors, self.strides, self.image_size,
                                self.iou_threshold, self.inf, self.eps)