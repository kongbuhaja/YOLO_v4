import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, HeUniform, HeNormal, Zeros
from tensorflow.keras import Model
from models.backbones import *
from models.necks import *
from models.heads import *
from utils.bbox_utils import bbox_iou
from utils.io_utils import read_model_info, write_model_info
from losses import yolov3, yolov4

class YOLO(Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.model_name = cfg['model']['name']
        self.num_classes = cfg['data']['labels']['count']
        self.nms = cfg['eval']['nms']['type']
        self.score_th = cfg['eval']['nms']['score_th']
        self.iou_th = cfg['eval']['nms']['iou_th']
        self.sigma = cfg['eval']['nms']['sigma']
        self.seed = cfg['seed']

        if 'YOLOv3' in self.model_name:
            decode = self.v3_decode
            loss = 'YOLOv3_Loss'
            if self.model_name == 'YOLOv3':
                backbone = 'Darknet53'
                neck = 'FPNSPP'
                head = 'Yolo'
                backbone_activate = 'LeakyReLU'
                activate = 'LeakyReLU'
                block_size = 2
                self.input_size = (512, 512)
                self.strides = cfg['model']['strides'][:3]
            elif self.model_name == 'YOLOv3_tiny':
                backbone = 'Darknet19'
                neck = 'tinyFPN'
                head = 'Yolo'
                backbone_activate = 'LeakyReLU'
                activate = 'LeakyReLU'
                self.input_size = (416, 416)
                self.strides = cfg['model']['strides'][1:3]
        elif 'YOLOv4' in self.model_name:
            decode = self.v4_decode
            loss = 'YOLOv4_Loss'
            if self.model_name == 'YOLOv4':
                backbone = 'CSPDarknet53'
                neck = 'PANSPP'
                head = 'Yolo'
                csp = False
                backbone_activate = 'Mish'
                activate = 'LeakyReLU'
                block_size = 2
                self.input_size = (512, 512)
                self.strides = cfg['model']['strides'][:3]
            elif self.model_name == 'YOLOv4_tiny':
                backbone = 'CSPDarknet19'
                neck = 'tinyFPN'
                head = 'Yolo'
                backbone_activate = 'LeakyReLU'
                activate = 'LeakyReLU'
                self.input_size = (416, 416)
                self.strides = cfg['model']['strides'][1:3]
            elif self.model_name == 'YOLOv4_csp':
                backbone = 'CSPDarknet53'
                neck = 'CSPPANSPP'
                head = 'Yolo'
                csp = True
                backbone_activate = 'Mish'
                activate = 'Mish'
                block_size = 2
                self.input_size = (512, 512)
                self.strides = cfg['model']['strides'][:3]
            elif 'YOLOv4_P' in self.model_name:
                backbone = 'CSPP'
                neck = 'CSPPANSPP'
                head = 'Yolo'
                backbone_activate = 'Mish'
                activate = 'Mish'
                block_size = 3
                size = int(self.model_name[-1])
                self.strides = cfg['model']['strides'][:size-2]
                if size==5:
                    self.input_size = (896, 896)
                elif size==6:
                    self.input_size = (1280, 1280)
                elif size==7:
                    self.input_size = (1536, 1536)

        self.input_size = np.array(self.input_size, np.int32)
        self.anchors = np.array(cfg['model']['anchors']) * self.input_size
        self.strides = np.array(self.strides, np.int32)
        self.row_anchors, self.col_anchors = self.anchors.shape[:2]
        self.scales = (self.input_size[None] // self.strides[:, None])
        self.anchors_grid = list(map(lambda x: tf.reshape(x, [-1,4]), get_anchors_grid(self.anchors, self.strides, self.input_size)))

        cfg['model']['input_size'] = self.input_size
        cfg['model']['anchors'] = self.anchors
        cfg['model']['strides'] = self.strides

        if cfg['model']['kernel_init'] == 'glorot_uniform':
            self.kernel_initializer = GlorotUniform(seed=self.seed)
        elif cfg['model']['kernel_init'] == 'glorot_normal':
            self.kernel_initializer = GlorotNormal(seed=self.seed)
        elif cfg['model']['kernel_init'] == 'he_uniform':
            self.kernel_initializer = HeUniform(seed=self.seed)
        elif cfg['model']['kernel_init'] == 'he_normal':
            self.kernel_initializer = HeNormal(seed=self.seed)
        else:
            self.kernel_initializer = Zeros()

        if loss == 'YOLOv3_Loss':
            self.loss = yolov3.loss(self.input_size, self.anchors, self.strides, self.num_classes, cfg['train']['assign'])
        elif loss == 'YOLOv4_Loss':
            self.loss = yolov4.loss(self.input_size, self.anchors, self.strides, self.num_classes, cfg['train']['assign'])

        if backbone == 'Darknet53':
            self.backbone = Darknet53(activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'Darknet19':
            self.backbone = Darknet19(activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPDarknet53':
            self.backbone = CSPDarknet53(activate=backbone_activate, csp=csp, kernel_initializer=self.kernel_initializer)
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
            self.head = YoloHead(256, decode, self.scales, self.row_anchors, self.col_anchors, self.num_classes,
                                 activate=activate, kernel_initializer=self.kernel_initializer)
            
        print(f'Model: {self.model_name}')
        print(f'Backbone: {backbone}')
        print(f'Neck: {neck}')
        print(f'Head: {head}')
        print(f'Input size: {self.input_size[0]}x{self.input_size[1]}')
        print(f'Loss Function: {loss}')

    @tf.function
    def call(self, x, training=False):
        backbone = self.backbone(x, training)
        neck = self.neck(backbone, training)
        head = self.head(neck, training)

        return head
    
    @tf.function
    def v3_decode(self, pred):
        xy = tf.sigmoid(pred[..., :2])
        wh = tf.exp(pred[..., 2:4])
        obj = tf.sigmoid(pred[..., 4:5])
        cls = tf.sigmoid(pred[..., 5:])

        return tf.concat([xy, wh, obj, cls], -1)
    
    @tf.function
    def v4_decode(self, pred):
        xy = tf.sigmoid(pred[..., :2]) * 2. - 0.5
        wh = tf.square(tf.sigmoid(pred[..., 2:4])*2)
        obj = tf.sigmoid(pred[..., 4:5])
        cls = tf.sigmoid(pred[..., 5:])

        return tf.concat([xy, wh, obj, cls], -1)
    
    def output(self, preds):
        batch_size = preds[0].shape[0]
        bboxes = tf.zeros((batch_size, 0, 4))
        scores = tf.zeros((batch_size, 0, 1))
        classes = tf.zeros((batch_size, 0, 1))
        for pred, anchor, stride in zip(preds, self.anchors_grid, self.strides):
            pred = tf.reshape(pred, [batch_size, -1, 5+self.num_classes])

            xy = pred[..., :2] + anchor[..., :2]
            wh = pred[..., 2:4] * anchor[..., 2:4]
            score = pred[..., 4:5]
            probs = pred[..., 5:]

            max_prob_id = tf.cast(tf.argmax(probs, -1)[..., None], tf.float32)
            max_prob = tf.reduce_max(probs, -1)[..., None]

            bboxes = tf.concat([bboxes, tf.concat([xy, wh], -1) * stride], 1)
            scores = tf.concat([scores, score * max_prob], 1)
            classes = tf.concat([classes, max_prob_id], 1)

        bboxes = tf.minimum(tf.maximum(bboxes, [0., 0., 0., 0.]), [*self.input_size, *self.input_size])

        return tf.concat([bboxes, scores, classes], -1)
    
    def NMS(self, preds):
        output = tf.zeros((0, 6), tf.float32)
        valid_mask = preds[..., 4] >= self.score_th
        
        if not tf.reduce_any(valid_mask):
            #return empty
            return tf.zeros((0, 6))

        targets = preds[valid_mask]


        while(targets.shape[0]):
            max_idx = tf.argmax(targets[..., 4], -1)
            max_target = targets[max_idx][None]
            output = tf.concat([output, max_target], 0)

            targets = tf.concat([targets[:max_idx], targets[max_idx+1:]], 0)
            ious = bbox_iou(max_target[:, :4], targets[:, :4], iou_type='diou')

            if self.nms == 'normal':
                new_scores = tf.where(ious >= self.iou_th, 0., targets[:, 4])
            elif self.nms == 'soft_normal':
                new_scores = tf.where(ious >= self.iou_th, targets[..., 4] * (1 - ious), targets[..., 4])
            elif self.nms == 'soft_gaussian':
                new_scores = tf.exp(-(ious)**2/self.sigma) * targets[:, 4]

            valid_mask = new_scores >= self.score_th
            targets = tf.concat([targets[:, :4], new_scores[:, None], targets[:, 5:]], -1)[valid_mask]

        return output
    
def load_model(cfg):
    model = YOLO(cfg)
    if cfg['model']['load']:
        try:
            model.load_weights(cfg['model']['checkpoint'])
            saved = read_model_info(cfg['model']['checkpoint'])
            print(f"succeed to load model| epoch:{saved['epoch']} mAP50:{saved['mAP50']} mAP:{saved['mAP']} total_loss:{saved['total_loss']}")
            return model, saved['epoch'], saved['mAP50'], saved['mAP'], saved['total_loss']
        except:
            print(f'{cfg["model"]["checkpoint"]} is not exist.')
    print('make new model')
    return model, 1, -1, -1., 9999999999

def save_model(model, epoch, mAP50, mAP, loss, checkpoint):
    model.save_weights(checkpoint)
    write_model_info(checkpoint, epoch, mAP50, mAP, loss)
    if 'map' in checkpoint.split('/')[-1]:
        print(f'{checkpoint} epoch:{epoch}, mAP50:{mAP50:.4f}, mAP:{mAP:.4f} best_model is saved')

def get_anchors_grid(anchors, strides, image_size):
    grid_anchors = []
    for i in range(len(strides)):
        scale = image_size // strides[i]
        scale_range = [tf.range(s, dtype=tf.float32) for s in scale]
        x, y = tf.meshgrid(*scale_range)
        xy = tf.concat([x[..., None], y[..., None]], -1)

        wh = tf.constant(anchors[i], dtype=tf.float32) / strides[i]
        xy = tf.tile(xy[:,:,None], (1, 1, len(anchors[i]), 1))
        wh = tf.tile(wh[None, None], (*scale.astype(np.int32), 1, 1))
        grid_anchors.append(tf.concat([xy,wh], -1))
    
    return grid_anchors