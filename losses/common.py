import tensorflow as tf
from utils import bbox_utils

class loss():
    def __init__(self, input_size, anchors, strides, num_classes, method='ratio'):
        self.input_size = tf.constant(input_size, tf.float32)
        self.anchors = tf.constant(anchors, tf.float32) * self.input_size[None, None]
        self.row_anchors, self.col_anchors = self.anchors.shape[:2]
        self.strides = tf.constant(strides, tf.float32)
        self.scales = (self.input_size[None] // self.strides[:, None])
        self.assign_method, self.assign_th = (self.ratio_assign, 4.) if method=='ratio' else (self.iou_assign, 0.2)
        self.num_classes = num_classes

    @staticmethod
    @tf.function
    def focal_weight(label, pred, gamma=2):
        return tf.pow(label - pred, gamma)
    
    @staticmethod
    @tf.function
    def BCE(label, pred, eps=1e-7):
        pred = tf.minimum(tf.maximum(pred, eps), 1-eps)
        return -(label*tf.math.log(pred) + (1.-label)*tf.math.log(1.-pred))

    @tf.function
    def onehot_label(self, label, smooth=True, alpha=0.1):
        onehot = tf.one_hot(label, self.num_classes)
        if smooth:
            return onehot * (1. - alpha) + alpha/self.num_classes
        return onehot
    
    @tf.function
    def iou_assign(self, anchors, targets):
        mask = bbox_utils.bbox_iou_wh(anchors, targets) > self.assign_th
        return mask
    
    @tf.function
    def ratio_assign(self, anchors, targets):
        ratio = targets / anchors
        mask = tf.reduce_max(tf.maximum(ratio, 1./ratio), -1) < self.assign_th
        return mask
    
    @tf.function
    def anchor_sampling(self, labels, bias=0.5):
        indices, tbox, tcls, anchs = [], [], [], []
        nt = len(labels)
        ai = tf.tile(tf.range(self.col_anchors, dtype=tf.float32)[:, None], [1, nt])
        targets = tf.concat([tf.tile(labels[None], [self.row_anchors, 1, 1]), ai[..., None]], -1)
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
                xyi = self.scales[r][::-1] - xy
                c_xy = tf.logical_and(xy % 1. < bias, xy > 1.)
                c_xyi = tf.logical_and(xyi % 1. < bias, xyi > 1.)
                offset_mask = tf.stack([tf.ones_like(c_xy[:, 0]), c_xy[:, 0], c_xy[:, 1], c_xyi[:, 0], c_xyi[:, 1]], 0)
                scaled_targets = tf.tile(scaled_targets[None], [5, 1, 1])[offset_mask]
                offsets = (tf.zeros_like(xy)[None] + off[:, None])[offset_mask]
            else:
                scaled_targets = targets[0]
                offsets = 0.

            b = tf.cast(scaled_targets[:, 0:1], tf.int32)
            xy = scaled_targets[:, 1:3]
            wh = scaled_targets[:, 3:5]
            c = tf.cast(scaled_targets[:, 5], tf.int32)
            # c = self.onehot_label(scaled_targets[:, 5])
            ij = tf.floor(xy - offsets)
            i, j = tf.clip_by_value(ij[:, 0:1], 0, self.scales[r, 1]-1), tf.clip_by_value(ij[:, 1:2], 0, self.scales[r, 0]-1)
            a = tf.cast(scaled_targets[:, 6:7], tf.int32)
            xy = xy - tf.concat([i, j], -1)
            # a = tf.gather(anchors, tf.cast(scaled_targets[:, 6], tf.int32))
            # positive_samples += [tf.concat([b, j, i, xy, wh, c, a], -1)]
            indices.append((b, tf.cast(j, tf.int32), tf.cast(i, tf.int32), a))
            tbox.append(tf.concat([xy - ij, wh], -1))
            tcls.append(c)
            anchs.append(tf.gather(anchors, tf.cast(scaled_targets[:, 6], tf.int32)))
        return indices, tbox, tcls, anchs