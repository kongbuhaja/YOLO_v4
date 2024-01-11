import tensorflow as tf
from losses.common import *
from utils.bbox_utils import *

class v4_loss(loss):
    def __init__(self, anchors, input_size, strides, num_classes, method='ratio'):
        super().__init__(anchors, input_size, strides, num_classes, method)
        self.conf_ratio = 1.0
        self.balance = [4.0, 1.0, 0.4]

    @tf.function
    def __call__(self, labels, preds):
        reg_loss, obj_loss, cls_loss = 0, 0, 0
    
        re = []
        # sampling
        indices, gt_box, gt_cls, anchors = self.anchor_sampling(labels, bias=0.5)
        
        # loss
        for l, pred in enumerate(preds):
            gt_obj = tf.zeros_like(pred[..., 0])
            idx = tf.concat(indices[l], -1)
            positive = tf.gather_nd(pred, idx)
                        
            # regression
            # pred_xy = tf.sigmoid(pred[..., :2])
            # pred_wh = tf.exp(pred[..., 2:4]) * anchors 
            pred_xy = tf.sigmoid(positive[..., :2]) * 2. - 0.5 # -0.5 ~ 1.5
            pred_wh = tf.square(tf.sigmoid(positive[..., 2:4])*2) * anchors[l]
            pred_box = tf.concat([pred_xy, pred_wh], -1)
            iou = bbox_iou(gt_box[l], pred_box, iou_type='ciou')
            reg_loss += tf.reduce_mean(1.0 - iou)
            
            # objectness
            gt_obj = tf.tensor_scatter_nd_update(gt_obj, idx, (1.0 - self.conf_ratio) + self.conf_ratio * tf.minimum(iou, 0.))
            pred_obj = tf.sigmoid(pred[..., 4])
            obj_loss += tf.reduce_mean(self.BCE(gt_obj, pred_obj) * self.balance[l])

            # classification
            one_hot = self.onehot_label(gt_cls[l])
            pred_cls = tf.sigmoid(positive[..., 5:])
            cls_loss += tf.reduce_mean(self.BCE(one_hot, pred_cls))

        return reg_loss, obj_loss, cls_loss
    