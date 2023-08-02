import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.blocks import *
from models.backbone import Darknet53
from models.necks import FPNSPP
from models.heads import Head
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
        self.scales = self.image_size // np.array(self.strides)
        self.iou_threshold = iou_threshold
        self.col_anchors = len(anchors[0])
        self.eps = eps
        self.inf = inf
        self.kernel_initializer = kernel_initializer

        if loss_metric == 'YOLOv4Loss':
            self.loss_metric = yolo_loss.v4_loss
        elif loss_metric == 'YOLOv3Loss':
            self.loss_metric = yolo_loss.v3_loss

        self.backbone = Darknet53(activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.neck = FPNSPP(512, layer_size=3, block_size=2, branch_transition=False,
                           activate='LeakyReLU', kernel_initializer=self.kernel_initializer)
        self.head = Head(256, self.scales, self.col_anchors, self.num_classes, 
                         activate='LeakyReLU', kernel_initializer=self.kernel_initializer)

    @tf.function    
    def call(self, x, training=False):
        backbone = self.backbone(x, training)
        neck = self.neck(backbone, training)
        head = self.head(neck, training)

        return head
    
    @tf.function
    def loss(self, labels, preds, batch_size):
        return self.loss_metric(labels, preds, batch_size, self.anchors, self.strides, self.image_size,
                                self.iou_threshold, self.inf, self.eps)