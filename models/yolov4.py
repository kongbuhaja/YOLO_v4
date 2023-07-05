import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.blocks import *
from models.backbone import CSPDarknet53
from config import *
from utils import anchor_utils
from losses import yolo_loss


class YOLO(Model):
    def __init__(self, anchors=ANCHORS, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE, strides=STRIDES,
                 iou_threshold=IOU_THRESHOLD, num_anchors=NUM_ANCHORS, eps=EPS, inf=INF, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.strides = strides
        self.anchors = anchor_utils.get_anchors_xywh(anchors, self.strides, image_size)
        self.image_size = image_size
        self.scales = (self.image_size // np.array(self.strides)).tolist()
        self.iou_threshold = iou_threshold
        self.kernel_initializer = kernel_initializer
        self.eps = eps
        self.inf = inf
        
        if LOSS_METRIC == 'YOLOv4Loss':
            self.loss_metric = yolo_loss.v4_loss
        elif LOSS_METRIC == 'YOLOv3Loss':
            self.loss_metric = yolo_loss.v3_loss

        self.backbone = CSPDarknet53(activate='Mish', kernel_initializer = self.kernel_initializer)

        self.spp_block = ReverseDarknetBlock(512, block='SPP', activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.medium_upsample_block = DarknetUpsampleBlock(256, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.small_upsample_block = DarknetUpsampleBlock(128, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

        self.small_grid_block = GridBlock(256, self.scales[0], self.num_anchors, self.num_classes, 
                                          activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

        self.medium_downsample_block = DarknetDownsampleBlock(256, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.medium_grid_block = GridBlock(512, self.scales[1], self.num_anchors, self.num_classes,
                                           activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

        self.large_downsample_block = DarknetDownsampleBlock(512, activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.large_grid_block = GridBlock(1024, self.scales[2], self.num_anchors, self.num_classes,
                                          activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

    def call(self, input, training=False):
        small_branch, medium_branch, large_branch = self.backbone(input, training)

        large_branch = self.spp_block(large_branch, training)
        medium_branch = self.medium_upsample_block(medium_branch, large_branch, training)
        small_branch = self.small_upsample_block(small_branch, medium_branch, training)
        
        s_grid = self.small_grid_block(small_branch, training)
        
        medium_branch = self.medium_downsample_block(small_branch, medium_branch, training)
        m_grid = self.medium_grid_block(medium_branch, training)
        
        large_branch = self.large_downsample_block(medium_branch, large_branch, training)
        l_grid = self.large_grid_block(large_branch, training)
        
        return s_grid, m_grid, l_grid
    
    @tf.function
    def loss(self, labels, preds, batch_size):
        return self.loss_metric(labels, preds, batch_size, self.anchors, self.strides, self.image_size,
                                self.iou_threshold, self.inf, self.eps)