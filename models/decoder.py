import tensorflow as tf
import numpy as np 
from utils.bbox_utils import bbox_iou, bbox_iou_np

class Decoder():
    def __init__(self, cfg):
        self.input_size = cfg['model']['input_size']
        self.strides = cfg['model']['strides']
        self.num_classes = cfg['data']['labels']['count']
        self.anchors = list(map(lambda x: tf.reshape(x, [-1,4]), get_anchors_grid(cfg['model']['anchors'], self.strides, self.input_size)))

        if cfg['model']['decode'] == 'v4':
            self.decode = self.v4_decode
        elif cfg['model']['decode'] == 'v2':
            self.decode = self.v2_decode
        else:
            self.decode = self.raw_decode

        if cfg['eval']['nms']['type'] == 'normal':
            self.nms = self.normal_nms
        elif cfg['eval']['nms']['type'] == 'soft_normal':
            self.nms = self.soft_normal_nms    
        elif cfg['eval']['nms']['type'] == 'soft_gaussian':
            self.nms = self.soft_gaussian_nms

        self.score_th = cfg['eval']['nms']['score_th']
        self.iou_th = cfg['eval']['nms']['iou_th']
        self.sigma = cfg['eval']['nms']['sigma']
    
    @tf.function
    def __call__(self, preds):
        decoded_preds = []
        for pred in preds:
            decoded_preds += [self.decode(pred)]
        
        return decoded_preds
    
    def NMS(self, targets):
        # output = tf.zeros((0, 6), tf.float32)
        output = []

        mask = targets[:, 4] >= self.score_th
        targets = targets[mask]
        while(tf.shape(targets)[0] > 0):   
            max_idx = tf.argmax(targets[:, 4], -1)
            max_target = targets[max_idx]
            # max_target = targets[max_idx][None]
            # output = tf.concat([output, max_target], 0)
            output += [max_target]

            targets = tf.concat([targets[:max_idx], targets[max_idx+1:]], 0)
            ious = bbox_iou(max_target[None, :4], targets[:, :4], iou_type='diou')
            # ious = bbox_iou(max_target[:, :4], targets[:, :4], iou_type='diou')
            scores = self.nms(ious, targets[:, 4])
            
            targets = tf.concat([targets[:, :4], scores[:, None], targets[:, 5:]], -1)

            mask = targets[:, 4] >= self.score_th
            targets = targets[mask]
        output = tf.stack(output)
        return output
    
    def NMS2(self, preds):
        output = []
        preds = preds[preds[:, 4] >= self.score_th]

        if preds.shape[0] == 0:
            return np.zeros([0, 6])
        
        while(preds.shape[0] > 0):
            max_idx = np.argmax(preds[:, 4], -1)
            max_pred = preds[max_idx]
            max_cls = max_pred[5]
            output += [max_pred]

            preds = np.concatenate([preds[:max_idx], preds[max_idx+1:]], 0)
            target_mask = preds[:, 5] == max_cls
            targets = preds[target_mask]
            ious = bbox_iou_np(max_pred[None, :4], targets[:, :4], iou_type='iou')
            preds[target_mask, 5] = np.where(ious >= self.iou_th, targets[:, 4] * (1 - ious), targets[:, 4])
            preds = preds[preds[:, 4] >= self.score_th]

        return np.stack(output)

    def NMS3(self, preds):
        output = []
        
        preds = preds[preds[:, 4] >= self.score_th]
        while(preds.shape[0] > 0):
            max_idx = tf.argmax(preds[:, 4], -1)
            max_pred = preds[max_idx]
            max_cls = max_pred[5]
            output += [max_pred]

            preds = tf.concat([preds[:max_idx], preds[max_idx+1:]], 0)
            target_idx = tf.where(preds[:, 5] == max_cls)
            targets = tf.gather(preds, target_idx[:, 0])
            ious = bbox_iou(max_pred[None, :4], targets[:, :4], iou_type='iou')
            indices = tf.concat([target_idx, tf.zeros_like(target_idx)+5], -1)
            scores = tf.where(ious >= self.iou_th, targets[:, 4] * (1 - ious), targets[:, 4])
            preds = tf.tensor_scatter_nd_update(preds, indices, scores)
            preds = preds[preds[:, 4] >= self.score_th]

        return tf.stack(output)

    @tf.function
    def bbox_decode(self, preds):
        batch_size = preds[0].shape[0]
        # bboxes = []
        # scores = []
        # classes = []
        bboxes = tf.zeros((batch_size, 0, 4))
        scores = tf.zeros((batch_size, 0, 1))
        classes = tf.zeros((batch_size, 0, 1))
        for pred, anchor, stride in zip(preds, self.anchors, self.strides):
            pred = tf.reshape(pred, [batch_size, -1, 5+self.num_classes])

            xy = pred[..., :2] + anchor[..., :2]
            wh = pred[..., 2:4] * anchor[..., 2:4]
            score = pred[..., 4:5]
            probs = pred[..., 5:]

            max_prob_id = tf.cast(tf.argmax(probs, -1)[..., None], tf.float32)
            max_prob = tf.reduce_max(probs, -1)[..., None]

            # bboxes += [tf.concat([xy, wh], -1) * stride]
            # scores += [score * max_prob]
            # classes += [max_prob_id]
            bboxes = tf.concat([bboxes, tf.concat([xy, wh], -1) * stride], 1)
            scores = tf.concat([scores, score * max_prob], 1)
            classes = tf.concat([classes, max_prob_id], 1)

        # bboxes = tf.concat(bboxes, 1)
        # scores = tf.concat(scores, 1)
        # classes = tf.concat(classes, 1)
        bboxes = tf.minimum(tf.maximum(bboxes, [0., 0., 0., 0.]), [*self.input_size, *self.input_size])

        return tf.concat([bboxes, scores, classes], -1)

    @tf.function
    def raw_decode(self, pred):
        return pred
    
    @tf.function
    def v2_decode(self, pred):
        xy = tf.sigmoid(pred[..., :2])
        wh = tf.exp(tf.minimum(pred[..., 2:4], 8.0))
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
    
    def normal_nms(self, ious, scores):
        return tf.where(ious >= self.iou_th, 0., scores)
    
    def soft_normal_nms(self, ious, scores):
        return tf.where(ious >= self.iou_th, scores * (1 - ious), scores)

    def soft_gaussian_nms(self, ious, scores):
        return tf.exp(-tf.square(ious)/self.sigma) * scores

    
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