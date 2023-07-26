import numpy as np
import tensorflow as tf
from utils.bbox_utils import bbox_iou_wh_np

def get_anchors_xywh(anchors, strides, image_size):
    anchors = tf.constant(anchors) * image_size
    grid_anchors = []
    for i in range(len(strides)):
        scale = image_size // strides[i]
        scale_range = tf.range(scale, dtype=tf.float32)
        x, y = tf.meshgrid(scale_range, scale_range)
        xy = tf.concat([x[..., None], y[..., None]], -1)

        wh = tf.constant(anchors[i], dtype=tf.float32) / strides[i]
        xy = tf.tile(xy[:,:,None], (1, 1, len(anchors[i]), 1))
        wh = tf.tile(wh[None, None], (scale, scale, 1, 1))
        grid_anchors.append(tf.concat([xy,wh], -1))
    
    return grid_anchors

def avg_iou(boxes, clusters):
    return np.mean([np.max(bbox_iou_wh_np(boxes[:, None], clusters), axis=1)])

def kmeans(boxes, k):
    num_boxes = boxes.shape[0]
    last_cluster = np.zeros((num_boxes,))
    
    clusters = boxes[np.random.choice(num_boxes, k, replace=False)]
    
    while True:
        distances = 1 - bbox_iou_wh_np(boxes[:, None], clusters)
        
        current_nearest = np.argmin(distances, axis=1)
        if(last_cluster == current_nearest).all():
            break
        for cluster in range(k):
            clusters[cluster] = np.mean(boxes[current_nearest == cluster], axis=0)
        
        last_cluster = current_nearest
    return clusters

def generate_anchors(boxes, k):
    best_avg_acc = 0.
    anchors = np.zeros((9,))
    for i in range(5):
        result = kmeans(boxes, k)
        avg_acc = avg_iou(boxes, result)
        if best_avg_acc < avg_acc:
            best_avg_acc = avg_acc
            anchors = np.array(sorted(np.array(result), key=lambda x: x[0]*x[1]), dtype=np.float16)

    return anchors, best_avg_acc
