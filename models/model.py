import tensorflow as tf
import numpy as np
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, HeUniform, HeNormal, Zeros
from tensorflow.keras import Model
from models.backbones import *
from models.necks import *
from models.heads import *
from utils.bbox_utils import bbox_iou
from utils.io_utils import read_model_info, write_model_info
from losses import yolov3, yolov4, yolov2

class YOLO(Model):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg['model']['name']
        self.num_classes = cfg['data']['labels']['count']
        self.nms = cfg['eval']['nms']['type']
        self.score_th = cfg['eval']['nms']['score_th']
        self.iou_th = cfg['eval']['nms']['iou_th']
        self.sigma = cfg['eval']['nms']['sigma']
        self.seed = cfg['seed']
        self.assign = cfg['train']['assign']
        self.focal = cfg['train']['focal']

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

        if 'YOLOv2' in self.model_name:
            decode = self.v2_decode
            loss = 'YOLOv2_Loss'
            if self.model_name == 'YOLOv2':
                unit = 32
                backbone = 'Darknet19_v2'
                backbone_unit = unit
                backbone_activate = 'LeakyReLU'
                neck = 'reOrg'
                neck_unit = unit
                neck_activate = 'LeakyReLU'
                head = 'Detect'
                input_size = (416, 416)
                strides = cfg['model']['strides'][2:3]
            elif self.model_name == 'YOLOv2_tiny':
                unit = 16
                backbone = 'Darknet19_v2_tiny'
                backbone_unit = unit
                backbone_activate = 'LeakyReLU'
                neck = 'Conv'
                neck_unit = unit*2**5
                neck_activate = 'LeakyReLU'
                head = 'Detect'
                input_size = (416, 416)
                strides = cfg['model']['strides'][2:3]
        elif 'YOLOv3' in self.model_name:
            decode = self.v2_decode
            loss = 'YOLOv3_Loss'
            if self.model_name == 'YOLOv3':
                unit = 32
                backbone = 'Darknet53'
                backbone_unit = unit
                backbone_activate = 'LeakyReLU'
                neck = 'FPN'
                neck_unit = unit
                neck_block_size = 2
                neck_activate = 'LeakyReLU'
                head = 'Detect'
                input_size = (512, 512)
                strides = cfg['model']['strides'][:3]
            elif self.model_name == 'YOLOv3_tiny':
                unit = 16
                backbone = 'Darknet19'
                backbone_unit = unit
                backbone_activate = 'LeakyReLU'
                neck = 'tinyFPN'
                neck_unit = unit * 2
                neck_activate = 'LeakyReLU'
                head = 'Detect'
                input_size = (416, 416)
                strides = cfg['model']['strides'][1:3]
        elif 'YOLOv4' in self.model_name:
            decode = self.v4_decode
            loss = 'YOLOv4_Loss'
            if self.model_name == 'YOLOv4':
                unit = 32
                csp = False
                backbone = 'CSPDarknet53'
                backbone_unit = unit
                backbone_activate = 'Mish'
                neck = 'PANSPP'
                neck_unit = unit
                neck_block_size = 2
                neck_activate = 'LeakyReLU'
                head = 'Detect'
                input_size = (512, 512)
                strides = cfg['model']['strides'][:3]
            elif self.model_name == 'YOLOv4_tiny':
                unit = 32
                backbone = 'CSPDarknet19'
                backbone_unit = unit
                backbone_activate = 'LeakyReLU'
                neck = 'tinyFPN'
                neck_unit = unit
                neck_activate = 'LeakyReLU'
                head = 'Detect'
                input_size = (416, 416)
                strides = cfg['model']['strides'][1:3]
            elif self.model_name == 'YOLOv4_csp':
                unit = 32
                csp = True
                backbone = 'CSPDarknet53'
                backbone_unit = unit
                backbone_activate = 'Mish'
                neck = 'CSPPANSPP'
                neck_unit = unit
                neck_block_size = 2
                neck_activate = 'Mish'
                head = 'Detect'
                input_size = (512, 512)
                strides = cfg['model']['strides'][:3]
            elif 'YOLOv4_P' in self.model_name:
                unit = 32
                backbone = 'CSPP'
                backbone_unit = unit
                backbone_activate = 'Mish'
                neck = 'CSPPANSPP'
                neck_unit = unit
                neck_block_size = 3
                neck_activate = 'Mish'
                head = 'Detect'
                size = int(self.model_name[-1])
                strides = cfg['model']['strides'][:size-2]
                if size==5:
                    input_size = (896, 896)
                elif size==6:
                    input_size = (1280, 1280)
                elif size==7:
                    input_size = (1536, 1536)

        self.input_size = np.array(input_size, np.int32)
        self.anchors = np.array(cfg['model']['anchors']).astype(np.float32) * self.input_size
        self.row_anchors, self.col_anchors = self.anchors.shape[:2]
        self.strides = np.array(strides, np.int32)
        self.anchors_grid = list(map(lambda x: tf.reshape(x, [-1,4]), get_anchors_grid(self.anchors, self.strides, self.input_size)))

        cfg['model']['input_size'] = self.input_size
        cfg['model']['anchors'] = self.anchors
        cfg['model']['strides'] = self.strides

        if loss == 'YOLOv2_Loss':
            self.loss = yolov2.loss(self.input_size, self.anchors, self.strides, self.num_classes, self.assign, self.focal)
        if loss == 'YOLOv3_Loss':
            self.loss = yolov3.loss(self.input_size, self.anchors, self.strides, self.num_classes, self.assign, self.focal)
        elif loss == 'YOLOv4_Loss':
            self.loss = yolov4.loss(self.input_size, self.anchors, self.strides, self.num_classes, self.assign, self.focal)

        if backbone == 'Darknet19_v2':
            self.backbone = Darknet19_v2(backbone_unit, activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'Darknet19_v2_tiny':
            self.backbone = Darknet19_v2_tiny(backbone_unit, activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'Darknet19':
            self.backbone = Darknet19(backbone_unit, activate=backbone_unit, kernel_initializer=self.kernel_initializer)
        elif backbone == 'Darknet53':
            self.backbone = Darknet53(backbone_unit, activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'Darknet19':
            self.backbone = Darknet19(backbone_unit, activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPDarknet53':
            self.backbone = CSPDarknet53(backbone_unit, activate=backbone_activate, csp=csp, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPDarknet19':
            self.backbone = CSPDarknet19(backbone_unit, activate=backbone_activate, kernel_initializer=self.kernel_initializer)
        elif backbone == 'CSPP':
            self.backbone = CSPP(backbone_unit, size, activate=backbone_activate, kernel_initializer=self.kernel_initializer)


        if neck == 'reOrg':
            self.neck = reOrg(neck_unit, activate=neck_activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'Conv':
            self.neck = ConvLayer(neck_unit, 3, activate=neck_activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'FPN':
            self.neck = FPN(neck_unit, self.row_anchors, neck_block_size, activate=neck_activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'PANSPP':
            self.neck = PANSPP(neck_unit, self.row_anchors, neck_block_size, activate=neck_activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'CSPFPN':
            self.neck = CSPFPN(neck_unit, self.row_anchors, neck_block_size, activate=neck_activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'CSPPANSPP':
            self.neck = CSPPANSPP(neck_unit, self.row_anchors, neck_block_size, activate=neck_activate, kernel_initializer=self.kernel_initializer)
        elif neck == 'tinyFPN':
            self.neck = tinyFPN(neck_unit, self.row_anchors, activate=neck_activate, kernel_initializer=self.kernel_initializer)
            
        if head == 'Detect':
            self.head = Detect(decode, self.row_anchors, self.col_anchors, self.num_classes, kernel_initializer=self.kernel_initializer)
            
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
    def v2_decode(self, pred):
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
    
    def NMS(self, targets):
        output = tf.zeros((0, 6), tf.float32)

        while(True):
            mask = targets[:, 4] >= self.score_th
            targets = targets[mask]

            if tf.shape(targets)[0] == 0:
                return output
            
            max_idx = tf.argmax(targets[:, 4], -1)
            max_target = targets[max_idx][None]
            output = tf.concat([output, max_target], 0)

            targets = tf.concat([targets[:max_idx], targets[max_idx+1:]], 0)
            ious = bbox_iou(max_target[:, :4], targets[:, :4], iou_type='diou')

            if self.nms == 'normal':
                new_scores = tf.where(ious >= self.iou_th, 0., targets[:, 4])
            elif self.nms == 'soft_normal':
                new_scores = tf.where(ious >= self.iou_th, targets[:, 4] * (1 - ious), targets[:, 4])
            elif self.nms == 'soft_gaussian':
                new_scores = tf.exp(-(ious)**2/self.sigma) * targets[:, 4]

            targets = tf.concat([targets[:, :4], new_scores[:, None], targets[:, 5:]], -1)
    
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