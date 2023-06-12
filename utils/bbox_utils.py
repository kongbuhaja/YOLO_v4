import tensorflow as tf
import numpy as np
from config import EPS, INF, IMAGE_SIZE

def bbox_iou(bbox1, bbox2, xywh=True, iou_type='iou', eps=EPS, inf=INF, image_size=IMAGE_SIZE):
    # if xywh:
    #     area1 = tf.reduce_prod(bbox1[..., 2:], -1)
    #     area2 = tf.reduce_prod(bbox2[..., 2:], -1)
    #     bbox1 = tf.concat([bbox1[..., :2] - bbox1[..., 2:] * 0.5, bbox1[..., :2] + bbox1[..., 2:] * 0.5], -1)
    #     bbox2 = tf.concat([bbox2[..., :2] - bbox2[..., 2:] * 0.5, bbox2[..., :2] + bbox2[..., 2:] * 0.5], -1)
    # else:
    #     area1 = tf.reduce_prod(bbox1[..., 2:] - bbox1[..., :2], -1)
    #     area2 = tf.reduce_prod(bbox2[..., 2:] - bbox2[..., :2], -1)

    if xywh:
        bbox1 = tf.concat([bbox1[..., :2] - bbox1[..., 2:] * 0.5, bbox1[..., :2] + bbox1[..., 2:] * 0.5], -1)
        bbox2 = tf.concat([bbox2[..., :2] - bbox2[..., 2:] * 0.5, bbox2[..., :2] + bbox2[..., 2:] * 0.5], -1)
    # bbox1 = tf.minimum(tf.maximum(bbox1, 0.0), image_size)
    # bbox2 = tf.minimum(tf.maximum(bbox2, 0.0), image_size)
    area1 = tf.reduce_prod(bbox1[..., 2:] - bbox1[..., :2], -1)
    area2 = tf.reduce_prod(bbox2[..., 2:] - bbox2[..., :2], -1)

    Left_Top = tf.maximum(bbox1[..., :2], bbox2[..., :2])
    Right_Bottom = tf.minimum(bbox1[..., 2:], bbox2[..., 2:])

    inter_area = tf.reduce_prod(tf.maximum(Right_Bottom - Left_Top, 0.0), -1)
    union_area = tf.maximum(area1 + area2 - inter_area, eps)

    iou = inter_area / union_area

    if iou_type in ['giou', 'diou', 'ciou']:
        c_Left_Top = tf.minimum(bbox1[..., :2], bbox2[..., :2])
        c_Right_Bottom = tf.maximum(bbox1[..., 2:], bbox2[..., 2:])
        if iou_type == 'giou':
            c_area = tf.maximum(tf.reduce_prod(c_Right_Bottom - c_Left_Top, -1), eps)
            giou = iou - (c_area - union_area)/c_area
            return giou
        
        elif iou_type in ['diou', 'ciou']:
            center_xy1 = (bbox1[..., :2] + bbox1[..., 2:]) * 0.5
            center_xy2 = (bbox2[..., :2] + bbox2[..., 2:]) * 0.5
            p_square = tf.reduce_sum(tf.minimum(tf.square(center_xy1 - center_xy2), inf), -1)
            c_square = tf.reduce_sum(tf.minimum(tf.maximum(tf.square(c_Right_Bottom - c_Left_Top), eps), inf), -1)

            if iou_type == 'diou':
                diou = iou - p_square/c_square
                return diou
            
            w1 = bbox1[..., 2] - bbox1[..., 0]
            h1 = tf.maximum(bbox1[..., 3] - bbox1[..., 1], eps)
            w2 = bbox2[..., 2] - bbox2[..., 0]
            h2 = tf.maximum(bbox2[..., 3] - bbox2[..., 1], eps)

            v = 4/tf.square(np.pi) * tf.square(tf.math.atan(w1/h1) - tf.math.atan(w2/h2))
            alpha = v/tf.maximum((1.0 - iou + v), eps)
            ciou = iou - p_square/c_square - alpha*v
            return ciou
    return iou

def xyxy_to_xywh(boxes, with_label=False):
    labels = tf.concat([(boxes[..., 0:2] + boxes[..., 2:4])*0.5, boxes[..., 2:4] - boxes[..., 0:2]],-1)
    if with_label:
        labels = tf.concat([labels, boxes[..., 4:]], -1)
    return labels

def xywh_to_xyxy(boxes, with_label=False):
    labels = tf.concat([boxes[..., :2] - boxes[..., 2:4] * 0.5 , boxes[..., :2] + boxes[..., 2:4] * 0.5], -1)
    if with_label:
        labels = tf.concat([labels, boxes[..., 4:]], -1)
    return labels

def coco_to_xyxy(boxes):
    bboxes = np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]])
    return bboxes

def normalize_bbox(w, h, bbox):
    bbox[..., [0,2]] /= w
    bbox[..., [1,3]] /= h
    return bbox

def bbox_iou_wh(wh1, wh2):
    inter_section = tf.minimum(wh1, wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = tf.maximum(wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter_area, 1e-6)
    return inter_area / union_area

def bbox_iou_wh_np(wh1, wh2):
    inter_section = np.minimum(wh1, wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = np.maximum(wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter_area, 1e-6)
    return inter_area / union_area

def extract_real_labels(labels, xywh=True):
    if xywh:
        w = labels[..., 2]
        h = labels[..., 3]
    else:
        w = labels[..., 2] - labels[..., 0]
        h = labels[..., 3] - labels[..., 1]

    return tf.gather(labels, tf.reshape(tf.where(tf.logical_and(w>0, h>0)), [-1]))