import tensorflow as tf
import numpy as np

def bbox_iou(bbox1, bbox2, xywh=True, iou_type='iou', eps=1e-7):
    if xywh:
        area1 = tf.reduce_prod(bbox1[..., 2:4], -1)
        area2 = tf.reduce_prod(bbox2[..., 2:4], -1)
        bbox1 = tf.concat([bbox1[..., :2] - bbox1[..., 2:] * 0.5, bbox1[..., :2] + bbox1[..., 2:] * 0.5], -1)
        bbox2 = tf.concat([bbox2[..., :2] - bbox2[..., 2:] * 0.5, bbox2[..., :2] + bbox2[..., 2:] * 0.5], -1)
    else:
        area1 = tf.reduce_prod(bbox1[..., 2:] - bbox1[..., :2], -1)
        area2 = tf.reduce_prod(bbox2[..., 2:] - bbox2[..., :2], -1)
    
    i_Left_Top = tf.maximum(bbox1[..., :2], bbox2[..., :2])
    i_Right_Bottom = tf.minimum(bbox1[..., 2:], bbox2[..., 2:])

    inter_area = tf.reduce_prod(tf.maximum(i_Right_Bottom - i_Left_Top, 0.0), -1)
    union_area = tf.maximum(area1 + area2 - inter_area, eps)

    iou = inter_area / union_area

    if iou_type in ['giou', 'diou', 'ciou']:
        c_Left_Top = tf.minimum(bbox1[..., :2], bbox2[..., :2])
        c_Right_Bottom = tf.maximum(bbox1[..., 2:], bbox2[..., 2:])
        if iou_type == 'giou':
            c_area = tf.maximum(tf.reduce_prod(c_Right_Bottom - c_Left_Top, -1), eps)
            return iou - (c_area - union_area)/c_area
        
        elif iou_type in ['diou', 'ciou']:
            center_xy1 = (bbox1[..., :2] + bbox1[..., 2:]) * 0.5
            center_xy2 = (bbox2[..., :2] + bbox2[..., 2:]) * 0.5
            p_square = tf.reduce_sum(tf.square(center_xy2 - center_xy1), -1)
            c_square = tf.maximum(tf.reduce_sum(tf.square(c_Right_Bottom - c_Left_Top), -1), eps)

            if iou_type == 'diou':
                return iou - p_square/c_square
            
            w1 = bbox1[..., 2] - bbox1[..., 0]
            h1 = tf.maximum(bbox1[..., 3] - bbox1[..., 1], eps)
            w2 = bbox2[..., 2] - bbox2[..., 0]
            h2 = tf.maximum(bbox2[..., 3] - bbox2[..., 1], eps)

            v = 4/tf.square(np.pi) * tf.square(tf.math.atan(w2/h2) - tf.math.atan(w1/h1))
            alpha = v/tf.maximum((1.0 - iou + v), eps)
            return iou - p_square/c_square - alpha*v
            
    return iou

def bbox_iou_np(bbox1, bbox2, xywh=True, iou_type='iou', eps=1e-7):
    if xywh:
        area1 = np.prod(bbox1[..., 2:4], -1)
        area2 = np.prod(bbox2[..., 2:4], -1)
        bbox1 = np.concatenate([bbox1[..., :2] - bbox1[..., 2:] * 0.5, bbox1[..., :2] + bbox1[..., 2:] * 0.5], -1)
        bbox2 = np.concatenate([bbox2[..., :2] - bbox2[..., 2:] * 0.5, bbox2[..., :2] + bbox2[..., 2:] * 0.5], -1)
    else:
        area1 = np.prod(bbox1[..., 2:] - bbox1[..., :2], -1)
        area2 = np.prod(bbox2[..., 2:] - bbox2[..., :2], -1)
    
    i_Left_Top = np.maximum(bbox1[..., :2], bbox2[..., :2])
    i_Right_Bottom = np.minimum(bbox1[..., 2:], bbox2[..., 2:])

    inter_area = np.prod(np.maximum(i_Right_Bottom - i_Left_Top, 0.0), -1)
    union_area = np.maximum(area1 + area2 - inter_area, eps)

    iou = inter_area / union_area
            
    if iou_type in ['giou', 'diou', 'ciou']:
        c_Left_Top = np.minimum(bbox1[..., :2], bbox2[..., :2])
        c_Right_Bottom = np.maximum(bbox1[..., 2:], bbox2[..., 2:])
        if iou_type == 'giou':
            c_area = np.maximum(np.prod(c_Right_Bottom - c_Left_Top, -1), eps)
            return iou - (c_area - union_area)/c_area
        
        elif iou_type in ['diou', 'ciou']:
            center_xy1 = (bbox1[..., :2] + bbox1[..., 2:]) * 0.5
            center_xy2 = (bbox2[..., :2] + bbox2[..., 2:]) * 0.5
            p_square = np.sum(np.square(center_xy2 - center_xy1), -1)
            c_square = np.maximum(np.sum(np.square(c_Right_Bottom - c_Left_Top), -1), eps)

            if iou_type == 'diou':
                return iou - p_square/c_square
            
            w1 = bbox1[..., 2] - bbox1[..., 0]
            h1 = np.maximum(bbox1[..., 3] - bbox1[..., 1], eps)
            w2 = bbox2[..., 2] - bbox2[..., 0]
            h2 = np.maximum(bbox2[..., 3] - bbox2[..., 1], eps)

            v = 4/np.square(np.pi) * np.square(np.arctan(w2/h2) - np.arctan(w1/h1))
            alpha = v/np.maximum((1.0 - iou + v), eps)
            return iou - p_square/c_square - alpha*v
        
    return iou

def xyxy_to_xywh(labels, start=0):
    labels = tf.concat([labels[..., :start],
                        (labels[..., start:start+2] + labels[..., start+2:start+4])*0.5,
                        labels[..., start+2:start+4] - labels[..., start:start+2],
                        labels[..., start+4:]],-1)

    return labels

def xywh_to_xyxy(labels, start=0):
    labels = tf.concat([labels[..., :start],
                        labels[..., start:start+2] - labels[..., start+2:start+4] * 0.5, 
                        labels[..., start:start+2] + labels[..., start+2:start+4] * 0.5,
                        labels[..., start+4:]], -1)
        
    return labels

def coco_to_xyxy(boxes):
    bboxes = np.concatenate([boxes[..., :2], boxes[..., :2] + boxes[..., 2:]])
    return bboxes

def bbox_iou_wh(wh1, wh2, eps=1e-7):
    inter_section = tf.minimum(wh1, wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = tf.maximum(wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter_area, eps)
    return inter_area / union_area

def bbox_iou_wh_np(wh1, wh2, eps=1e-7):
    inter_section = np.minimum(wh1, wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = np.maximum(wh1[..., 0] * wh1[..., 1] + wh2[..., 0] * wh2[..., 1] - inter_area, eps)
    return inter_area / union_area

def unresize_unpad_labels(labels, pad, ratio):
    # labels = tf.concat([(labels[..., 0:2] - pad) / ratio,
    #                     (labels[..., 2:4] - pad) / ratio,
    #                     labels[..., 4:]], -1)
    labels = tf.concat([(labels[..., 0:2] - pad) / ratio,
                        labels[..., 2:4] / ratio,
                        labels[..., 4:]], -1)

    return labels