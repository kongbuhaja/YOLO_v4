import tensorflow as tf
from losses.base import base_loss, Focal_loss, Sampler
from utils.bbox_utils import *

class loss(base_loss):
    def __init__(self, input_size, anchors, strides, num_classes, assign, focal):
        super().__init__(num_classes)
        self.sampler = Sampler(input_size, anchors, strides, assign)
        self.conf_ratio = 1.0
        self.balance = [4.0, 1.0, 0.4, 0.06, 0.02]
        self.reg_ratio = 0.05
        self.obj_ratio = 0.7
        self.cls_ratio = 0.3
        self.obj_loss = Focal_loss(self.BCE, 'mean', focal['gamma'], focal['alpha'])
        self.cls_loss = Focal_loss(self.BCE, 'mean', focal['gamma'], focal['alpha'])

    @tf.function
    def __call__(self, labels, preds):
        reg_loss, obj_loss, cls_loss = 0., 0., 0.
        # sampling
        indices, gt_boxes, gt_cls, anchors = self.sampler(labels)
        
        # loss
        for l, pred in enumerate(preds):
            gt_obj = tf.zeros_like(pred[..., 0])
            
            if len(indices[l]) > 0:
                positive = tf.gather_nd(pred, indices[l])
                            
                # regression
                pred_xy = positive[..., :2]
                pred_wh = positive[..., 2:4] * anchors[l]
                pred_box = tf.concat([pred_xy, pred_wh], -1)
                iou = bbox_iou(gt_boxes[l], pred_box, iou_type='ciou')
                reg_loss += tf.reduce_mean(1.0 - iou)
            
                # objectness
                gt_obj = tf.tensor_scatter_nd_update(gt_obj, indices[l], (1.0 - self.conf_ratio) + self.conf_ratio * tf.maximum(iou, 0.))

                # classification
                one_hot = self.onehot_label(gt_cls[l])
                pred_cls = positive[..., 5:]
                cls_loss += self.cls_loss(one_hot, pred_cls)

            # objectness
            pred_obj = pred[..., 4]
            obj_loss += self.obj_loss(gt_obj, pred_obj) * self.balance[l]
    
        batch_size = preds[0].shape[0]
        reg_loss *= self.reg_ratio * batch_size
        obj_loss *= self.obj_ratio * batch_size
        cls_loss *= self.cls_ratio * batch_size
        total_loss = reg_loss + obj_loss + cls_loss

        return total_loss, reg_loss, obj_loss, cls_loss