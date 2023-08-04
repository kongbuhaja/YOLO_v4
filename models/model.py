import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform as glorot
from tensorflow.keras.initializers import HeUniform as he
from tensorflow.keras import Model
from models.backbones import *
from models.necks import *
from models.heads import *
from utils.anchor_utils import get_anchors_xywh
from losses import yolo_loss

class YOLO(Model):
    def __init__(self, model_type, anchors, num_classes, strides, iou_threshold, 
                 eps, inf, kernel_initializer=glorot, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.eps = eps
        self.inf = inf

        if self.model_type == 'YOLOv3':
            backbone = 'Darknet53'
            neck = 'FPNSPP'
            head = 'Yolo'
            backbone_activate = 'LeakyReLU'
            activate = 'LeakyReLU'
            block_size = 2
            loss_metric = 'YOLOv3_Loss'
            self.input_size = 512
            self.strides = strides[:3]
        elif self.model_type == 'YOLOv3_tiny':
            backbone = 'Darknet19'
            neck = 'tinyFPN'
            head = 'Yolo'
            backbone_activate = 'LeakyReLU'
            activate = 'LeakyReLU'
            loss_metric = 'YOLOv3_Loss'
            self.input_size = 416
            self.strides = strides[1:3]
        elif self.model_type == 'YOLOv4':
            backbone = 'CSPDarknet53'
            neck = 'PANSPP'
            head = 'Yolo'
            scaled = False
            backbone_activate = 'Mish'
            activate = 'LeakyReLU'
            block_size = 2
            loss_metric = 'YOLOv4_Loss'
            self.input_size = 512
            self.strides = strides[:3]
        elif self.model_type == 'YOLOv4_tiny':
            backbone = 'CSPDarknet19'
            neck = 'tinyFPN'
            head = 'Yolo'
            backbone_activate = 'LeakyReLU'
            activate = 'LeakyReLU'
            loss_metric = 'YOLOv4_Loss'
            self.input_size = 416
            self.strides = strides[1:3]
        elif self.model_type == 'YOLOv4_csp':
            backbone = 'CSPDarknet53'
            neck = 'CSPPANSPP'
            head = 'Yolo'
            scaled = True
            backbone_activate = 'Mish'
            activate = 'Mish'
            block_size = 2
            loss_metric = 'YOLOv4_Loss'
            self.input_size = 512
            self.strides = strides[:3]
        elif 'YOLOv4_P' in self.model_type:
            backbone = 'CSPP'
            neck = 'CSPPANSPP'
            head = 'Yolo'
            backbone_activate = 'Mish'
            activate = 'Mish'
            block_size = 3
            loss_metric = 'YOLOv4_Loss'
            size = int(self.model_type[-1])
            self.strides = strides[:size-2]
            if size==5:
                self.input_size = 896
            elif size==6:
                self.input_size = 1280
            elif size==7:
                self.input_size = 1536

        self.row_anchors = len(anchors)
        self.col_anchors = len(anchors[0])
        self.scales = (self.input_size // np.array(self.strides)).tolist()
        self.anchors_xywh = get_anchors_xywh(anchors, self.strides, self.input_size)

        if kernel_initializer == 'glorot':
            self.kernel_initializer = glorot
        elif kernel_initializer == 'he':
            self.kernel_initializer = he
        else:
            self.kernel_initializer = kernel_initializer

        if loss_metric == 'YOLOv3_Loss':
            self.loss_metric = yolo_loss.v3_loss
        elif loss_metric == 'YOLOv4_Loss':
            self.loss_metric = yolo_loss.v4_loss

        if backbone == 'Darknet53':
            self.backbone = Darknet53(activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'Darknet19':
            self.backbone = Darknet19(activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPDarknet53':
            self.backbone = CSPDarknet53(activate=backbone_activate, scaled=scaled, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPDarknet19':
            self.backbone = CSPDarknet19(activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPP':
            self.backbone = CSPP(size=size, activate=backbone_activate, kernel_initializer=self.kernel_initializer)

        if neck == 'FPNSPP':
            self.neck = FPNSPP(512, layer_size=self.row_anchors, block_size=block_size, branch_transition=False,
                               activate=activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'PANSPP':
            self.neck = PANSPP(512, layer_size=self.row_anchors, block_size=block_size, branch_transition=True,
                               activate=activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'tinyFPN':
            self.neck = tinyFPN(256, activate=activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'CSPFPNSPP':
            self.neck = CSPFPNSPP(512, layer_size=self.row_anchors, block_size=block_size, branch_transition=True,
                                  activate=activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'CSPPANSPP':
            self.neck = CSPPANSPP(512, layer_size=self.row_anchors, block_size=block_size, branch_transition=True,
                                  activate=activate, kernel_initializer=self.kernel_initializer)
            
        if head == 'Yolo':
            self.head = YoloHead(256, self.scales, self.col_anchors, self.num_classes,
                                 activate=activate, kernel_initializer=self.kernel_initializer)
            
        print(f'Model: {self.model_type}')
        print(f'Backbone: {backbone}')
        print(f'Neck: {neck}')
        print(f'Head: {head}')
        print(f'Input size: {self.input_size}x{self.input_size}')
        print(f'Loss Metric: {loss_metric}')

    @tf.function
    def call(self, x, training=False):
        backbone = self.backbone(x, training)
        neck = self.neck(backbone, training)
        head = self.head(neck, training)

        return head
    
    @tf.function
    def loss(self, labels, preds, batch_size):
        return self.loss_metric(labels, preds, batch_size, self.anchors_xywh, self.strides, self.input_size,
                                self.iou_threshold, self.inf, self.eps)