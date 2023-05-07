import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Concatenate
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.common import *
from models.backbone import Darknet19
import numpy as np
from config import *
from utils import anchor_utils
from losses import yolo_loss

    
class YOLO(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES,
                 iou_threshold=IOU_THRESHOLD, num_anchors=NUM_ANCHORS, eps=EPS, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.strides = np.array(strides)
        self.anchors = anchor_utils.get_anchors_xywh(anchors, strides, image_size)
        self.scales = image_size//self.strides
        self.iou_threshold = iou_threshold
        self.num_anchors = num_anchors
        self.eps = eps
        self.inf = 1e+30
        self.kernel_initializer = kernel_initializer

        self.darknet19_tiny = Darknet19(kernel_initializer=self.kernel_initializer)
        
        self.conv_layer = DarknetConv(256, 1, kernel_initializer=self.kernel_initializer)
        
        self.large_grid_layer = GridOut(512, self.scales[1], self.num_anchors, self.num_classes, kernel_initializer=self.kernel_initializer)
        
        self.large_upsample_layers = [DarknetConv(128, 1, kernel_initializer=self.kernel_initializer),
                                      DarknetUpsample()]
        
        self.medium_grid_layer = GridOut(256, self.scales[0], self.num_anchors, self.num_classes, kernel_initializer=self.kernel_initializer)
        print('Model: YOLOv3_tiny')
    def call(self, input, training):
        medium_branch, large_branch = self.darknet19_tiny(input, training)
        
        large_branch = self.conv_layer(large_branch, training)
        
        l_grid = self.large_grid_layer(large_branch, training)
                    
        for i in range(len(self.large_upsample_layers)):
            large_branch = self.large_upsample_layers[i](large_branch, training)
            
        medium_branch = Concatenate()([medium_branch, large_branch])

        m_grid = self.medium_grid_layer(medium_branch)

        return m_grid, l_grid
    
    @tf.function
    def loss(self, labels, preds):
        return yolo_loss.v3_loss(labels, preds, self.anchors, self.iou_threshold, self.inf, self.eps)