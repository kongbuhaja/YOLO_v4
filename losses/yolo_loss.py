import tensorflow as tf
from losses.common import *
from losses.loc_loss import *
from losses.conf_loss import *
from losses.prob_loss import *

@tf.function
def v4_loss(labels, preds, anchors, iou_threshold, scales, inf, eps):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.
    # conf에 전체 gt하고 비교해보기 (not grid)
    for pred, label, anchor, scale in zip(preds, labels, anchors, scales):
        pred_xy = tf.sigmoid(pred[..., :2]) + anchor[..., :2]
        pred_wh = tf.exp(pred[..., 2:4]) * anchor[..., 2:]
        pred_xywh = tf.concat([pred_xy, pred_wh], -1)
        pred_conf = tf.sigmoid(pred[..., 4:5])
        pred_prob = tf.sigmoid(pred[..., 5:])
        loc_loss += yolov4_loc_loss(pred_xywh, label[..., :4], label[..., 4:5], scale, inf, eps)
        conf_loss += yolov4_conf_loss(pred_xywh, pred_conf, label[..., :4], label[..., 4:5], iou_threshold, inf, eps)
        prob_loss += yolov4_prob_loss(pred_prob, label[..., 5:], label[..., 4:5], inf, eps)

    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss

    return loc_loss, conf_loss, prob_loss, total_loss

@tf.function
def v3_loss(labels, preds, anchors, iou_threshold, inf, eps, coord=5, noobj=0.5):
    loc_loss, conf_loss, prob_loss = 0., 0., 0.

    for label, pred, anchor in zip(labels, preds, anchors):       
        loc_loss += v3_loc_loss(label[..., :5], pred[..., :4], anchor, inf, eps, coord)
        conf_loss += v3_conf_loss(label[..., :5], pred[..., :5], anchor, iou_threshold, inf, eps, noobj)
        prob_loss += v3_prob_loss(label[..., 4:], pred[..., 5:], inf, eps)
            
    loc_loss = tf.reduce_mean(loc_loss)
    conf_loss = tf.reduce_mean(conf_loss)
    prob_loss = tf.reduce_mean(prob_loss)
    total_loss = loc_loss + conf_loss + prob_loss

    return loc_loss, conf_loss, prob_loss, total_loss

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