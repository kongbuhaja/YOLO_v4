import tensorflow as tf
from utils.bbox_utils import bbox_iou_wh

class Base_loss():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @tf.function
    def onehot_label(self, label, smooth=True, alpha=0.1):
        onehot = tf.one_hot(label, self.num_classes)
        if smooth:
            return onehot * (1. - alpha) + alpha/self.num_classes
        return onehot

    @tf.function
    def BCE(self, label, pred, eps=1e-7):
        pred = tf.minimum(tf.maximum(pred, eps), 1-eps)
        return -(label*tf.math.log(pred) + (1. - label)*tf.math.log(1. - pred))
    
class Focal_loss():
    def __init__(self, loss, reduction='none', gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        self.loss = loss
        if reduction == 'mean':
            self.reduction = tf.reduce_mean
        elif reduction == 'sum':
            self.reduction = tf.reduce_sum
        else:
            self.reduction = lambda x : x

    @tf.function
    def __call__(self, label, pred):
        loss = self.loss(label, pred)
        p_t = (label * pred) + ((1 - label) * (1 - pred))
        alpha_facthor = (label * self.alpha + (1 - label) * (1 - self.alpha)) if self.alpha else 1.0
        modulating_factor = tf.pow((1.0 - p_t), self.gamma) if self.gamma else 1.0
        
        return self.reduction(alpha_facthor * modulating_factor * loss)
    
class Sampler():
    def __init__(self, input_size, anchors, strides, assign):
        input_size = tf.constant(input_size, tf.float32)
        self.anchors = tf.cast(anchors, tf.float32)
        self.row_anchors, self.col_anchors = self.anchors.shape[:2]
        self.strides = strides
        self.scales = (input_size[None] // self.strides[:, None])
        self.assign_method, self.assign_th = (self.ratio_assign, assign['ratio_th']) if assign['method']=='ratio' else (self.iou_assign, assign['iou_th'])

    @tf.function
    def iou_assign(self, anchors, targets):
        mask = bbox_iou_wh(anchors, targets) > self.assign_th
        return mask
    
    @tf.function
    def ratio_assign(self, anchors, targets):
        ratio = targets / anchors
        mask = tf.reduce_max(tf.maximum(ratio, 1./ratio), -1) < self.assign_th
        return mask
    
    @tf.function
    def __call__(self, labels, bias=0.5):
        indices, tbox, tcls, anchs = [], [], [], []
        nt = len(labels)
        ai = tf.tile(tf.range(self.col_anchors, dtype=tf.float32)[:, None], [1, nt])
        targets = tf.concat([tf.tile(labels[None], [self.col_anchors, 1, 1]), ai[..., None]], -1)
        off = tf.constant([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], tf.float32) * bias

        for r in range(self.row_anchors):
            anchors = self.anchors[r]
            stride = tf.stack([1., self.strides[r], self.strides[r], self.strides[r], self.strides[r], 1., 1.])
            scaled_targets = targets/stride
            if nt > 0:
                # match
                match_mask = self.assign_method(anchors[:, None], targets[..., 3:5])
                scaled_targets = scaled_targets[match_mask]

                # offsets
                xy = scaled_targets[..., 1:3]
                xyi = self.scales[r] - xy
                c_xy = tf.logical_and(xy % 1. < bias, xy > 1.)
                c_xyi = tf.logical_and(xyi % 1. < bias, xyi > 1.)
                offset_mask = tf.stack([tf.ones_like(c_xy[:, 0]), c_xy[:, 0], c_xy[:, 1], c_xyi[:, 0], c_xyi[:, 1]], 0)
                scaled_targets = tf.tile(scaled_targets[None], [5, 1, 1])[offset_mask]
                offsets = (tf.zeros_like(xy)[None] + off[:, None])[offset_mask]
            else:
                scaled_targets = targets[0]
                offsets = 0.

            b = tf.cast(scaled_targets[:, 0], tf.int32)
            xy = scaled_targets[:, 1:3]
            wh = scaled_targets[:, 3:5]
            c = tf.cast(scaled_targets[:, 5], tf.int32)
            ij = tf.cast(tf.cast((xy - offsets), tf.int32), tf.float32)
            i = tf.cast(tf.clip_by_value(ij[:, 0], 0, self.scales[r, 0] - 1), tf.int32)
            j = tf.cast(tf.clip_by_value(ij[:, 1], 0, self.scales[r, 1] - 1), tf.int32)
            a = tf.cast(scaled_targets[:, 6], tf.int32)

            indices.append(tf.stack([b, j, i, a], -1))
            tbox.append(tf.concat([xy - ij, wh], -1))
            tcls.append(c)
            anchs.append(tf.gather(anchors/self.strides[r], a))
        return indices, tbox, tcls, anchs
    