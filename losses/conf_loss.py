import tensorflow as tf
from losses.common import *
from utils.bbox_utils import bbox_iou

@tf.function
def v4_conf_loss(pred_xywh, pred_conf, label_xywh, resp_mask, iou_threshold, inf, eps):
    conf_weight = focal_weight(resp_mask, pred_conf, gamma=2)
    ious = bbox_iou(pred_xywh, label_xywh)[..., None]
    noresp_mask = (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)

    conf_loss = conf_weight * (resp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps) +
                              noresp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps))
    conf_loss = tf.reduce_sum(tf.minimum(conf_loss, inf), [1,2,3,4])

    return conf_loss

@tf.function
def v3_conf_loss(pred_xywh, pred_conf, label_xywh, resp_mask, iou_threshold, inf, eps, noobj):   
    ious = bbox_iou(pred_xywh, label_xywh, iou_type='iou')[..., None]
    
    noresp_mask = (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
    
    obj_loss = tf.reduce_sum(tf.minimum(resp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps), inf), [1,2,3,4])
    noobj_loss = noobj * tf.reduce_sum(tf.minimum(noresp_mask * binaryCrossEntropy(resp_mask, pred_conf, eps), inf), [1,2,3,4])
    
    return obj_loss + noobj_loss

@tf.function
def v2_conf_loss(label_xywhc, pred_xywhc_raw, anchor, iou_threshold, inf, eps, noobj):
    pred_xy_iou = tf.sigmoid(pred_xywhc_raw[..., :2]) + anchor[..., :2]
    pred_wh_iou = tf.exp(pred_xywhc_raw[..., 2:4]) * anchor[..., 2:]
    pred_xywh_iou = tf.concat([pred_xy_iou, pred_wh_iou], -1)
    
    label_xywh_iou = label_xywhc[..., :4]
    
    ious = bbox_iou(pred_xywh_iou, label_xywh_iou, iou_type='iou')[..., None]
    
    pred_conf = tf.sigmoid(pred_xywhc_raw[..., 4:5])
    resp_mask = label_xywhc[..., 4:5]
    noresp_mask = (1.0 - resp_mask) * tf.cast(ious < iou_threshold, dtype=tf.float32)
    
    obj_loss = tf.reduce_sum(tf.minmum(resp_mask * tf.square(resp_mask - pred_conf), inf), [1,2,3,4])
    noobj_loss = tf.reduce_sum(tf.minimum(noobj * noresp_mask * tf.square(resp_mask - pred_conf), inf), [1,2,3,4])
    
    return obj_loss + noobj_loss