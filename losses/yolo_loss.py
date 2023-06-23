import tensorflow as tf
from losses.common import *
from losses.loc_loss import *
from losses.conf_loss import *
from losses.prob_loss import *

@tf.function
def v4_loss(labels, preds, batch_size, anchors, strides, image_size, iou_threshold, inf, eps):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.
    for pred, label, anchor, stride in zip(preds, labels, anchors, strides):
        pred_xy = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
        pred_wh = tf.exp(pred[..., 2:4]) * anchor[..., 2:4] 
        # pred_xywh = tf.concat([pred_xy, pred_wh], -1)
        pred_xywh = tf.minimum(tf.maximum(tf.concat([pred_xy, pred_wh], -1) * stride, 0), image_size)
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_prob = tf.sigmoid(pred[..., 5:])

        label_xywh = label[..., :4]
        
        loc_loss += v4_loc_loss(pred_xywh, label_xywh, label[..., 4:5], image_size, inf)
        conf_loss += v4_conf_loss(pred_xywh, pred_conf, label_xywh, label[..., 4:5], iou_threshold, inf, eps)
        prob_loss += v4_prob_loss(pred_prob, label[..., 5:], label[..., 4:5], inf, eps)

    loc_loss = tf.reduce_sum(loc_loss) / batch_size
    conf_loss = tf.reduce_sum(conf_loss) / batch_size
    prob_loss = tf.reduce_sum(prob_loss) / batch_size
    total_loss = loc_loss + conf_loss + prob_loss

    return [loc_loss, conf_loss, prob_loss, total_loss]

@tf.function
def v3_loss(labels, preds, batch_size, anchors, strides, image_size, iou_threshold, inf, eps, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor, stride in zip(labels, preds, anchors, strides):
        pred_xy = tf.sigmoid(pred[..., :2])
        pred_wh = pred[..., 2:4]
        pred_xywh = tf.concat([pred_xy, pred_wh], -1)
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_prob = tf.sigmoid(pred[..., 5:])

        label_xy = label[..., :2] / stride - anchor[..., :2]
        label_wh = tf.math.log(tf.maximum(label[..., 2:4] / stride / anchor[..., 2:], eps))
        label_xywh = tf.concat([label_xy, label_wh], -1)

        loc_loss += v3_loc_loss(pred_xywh, label_xywh, label[..., 4:5], inf, coord)
        conf_loss += v3_conf_loss(pred_xywh, pred_conf, label_xywh, label[..., 4:5], iou_threshold, inf, eps, noobj)
        prob_loss += v3_prob_loss(pred_prob, label[..., 5:], label[..., 4:5], inf, eps)
            
    loc_loss = tf.reduce_sum(loc_loss) / batch_size
    conf_loss = tf.reduce_sum(conf_loss) / batch_size
    prob_loss = tf.reduce_sum(prob_loss) / batch_size
    total_loss = loc_loss + conf_loss + prob_loss

    return loc_loss, conf_loss, prob_loss, total_loss

# 수정필요
@tf.function
def v3_paper_loss(labels, preds, anchors, iou_threshold, inf, eps, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor in zip(labels, preds, anchors):
        loc_loss += v3_paper_loc_loss(label[..., :5], pred[..., :4], anchor, inf, eps, coord)
        conf_loss += v3_conf_loss(label[..., :5], pred[..., :5], anchor, iou_threshold, inf, eps, noobj)
        prob_loss += v3_prob_loss(label[..., 4:], pred[..., 5:], inf, eps)
        
        
    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss
    return loc_loss, conf_loss, prob_loss, total_loss

@tf.function
def v2_loss(labels, preds, anchors, strides, iou_threshold, inf, eps, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor, stride in zip(labels, preds, anchors, strides):
        loc_loss += v2_loc_loss(label[..., :5], pred[..., :4], anchor, inf, eps, coord)
        conf_loss += v2_conf_loss(label[..., :5], pred[..., :5], anchor, iou_threshold, inf, eps, noobj)
        prob_loss += v2_prob_loss(label[..., 4:], pred[..., 5:], inf, eps)
                
    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss
    return loc_loss, conf_loss, prob_loss, total_loss