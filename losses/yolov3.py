import tensorflow as tf
from losses.common import *
from utils.bbox_utils import *

class loss(loss):
    def __init__(self, input_size, anchors, strides, num_classes, assign):
        super().__init__(input_size, anchors, strides, num_classes, assign)
        self.balance = [4.0, 1.0, 0.4]
        self.coord = 5
        self.noobj = 0.5

    @tf.function
    def __call__(self, labels, preds):
        reg_loss, obj_loss, cls_loss = 0., 0., 0.
        # sampling
        indices, gt_box, gt_cls, anchors = self.anchor_sampling(labels, bias=0.5)
        
        # loss
        for l, pred in enumerate(preds):
            gt_obj = tf.zeros_like(pred[..., 0])
            
            if len(indices[l]) > 0:
                positive = tf.gather_nd(pred, indices[l])
                            
                # regression
                pred_xy = positive[..., :2]
                pred_wh = positive[..., 2:4] * anchors[l]
                pred_box = tf.concat([pred_xy, pred_wh], -1)
                reg_loss += tf.reduce_mean(tf.square(gt_box[l] - pred_box))
            
                # objectness
                gt_obj = tf.tensor_scatter_nd_update(gt_obj, indices[l], tf.ones(len(indices[l])))

                # classification
                one_hot = self.onehot_label(gt_cls[l])
                pred_cls = positive[..., 5:]
                cls_loss += tf.reduce_mean(self.BCE(one_hot, pred_cls))

            # objectness
            pred_obj = pred[..., 4]
            obj_weight = tf.where(tf.cast(gt_obj, tf.bool), 1.0, self.noobj)
            obj_loss += tf.reduce_mean(self.BCE(gt_obj, pred_obj) * obj_weight) * self.balance[l]
    
        batch_size = preds[0].shape[0]
        reg_loss *= self.coord * batch_size
        obj_loss *= batch_size
        cls_loss *= batch_size
        total_loss = reg_loss + obj_loss + cls_loss

        return total_loss, reg_loss, obj_loss, cls_loss