import tensorflow as tf
from utils import bbox_utils

def prediction_to_bbox(grids, anchors, batch_size, strides, num_classes, input_size):
    bboxes = tf.zeros((batch_size, 0, 4))
    scores = tf.zeros((batch_size, 0, 1))
    classes = tf.zeros((batch_size, 0, 1))
    for grid, anchor, stride in zip(grids, anchors, strides):
        grid = tf.reshape(grid, [batch_size, -1, 5+num_classes])

        xy = tf.sigmoid(grid[..., :2]) + anchor[..., :2]
        wh = tf.exp(grid[..., 2:4]) * anchor[..., 2:4]
        score = tf.sigmoid(grid[..., 4:5])
        probs = tf.sigmoid(grid[..., 5:])

        max_prob_id = tf.cast(tf.argmax(probs, -1)[..., None], tf.float32)
        max_prob = tf.reduce_max(probs, -1)[..., None]

        bboxes = tf.concat([bboxes, tf.concat([xy, wh], -1) * stride], 1)
        scores = tf.concat([scores, score * max_prob], 1)
        classes = tf.concat([classes, max_prob_id], 1)

    bboxes = bbox_utils.xywh_to_xyxy(bboxes)
    bboxes = tf.minimum(tf.maximum(0., bboxes), input_size)

    return tf.concat([bboxes, scores, classes], -1)

def NMS(preds, score_threshold, iou_threshold, method, sigma):
    valid_mask = preds[..., 4] >= score_threshold
    
    if not tf.reduce_any(valid_mask):
        #return empty
        return tf.zeros((0, 6))

    targets = preds[valid_mask]


    for idx in range(targets.shape[0]-1):
        max_idx = tf.argmax(targets[..., 4][idx:], -1)
        max_target = targets[max_idx]

        if max_target[4] < score_threshold:
            break

        targets = tf.concat([targets[:idx], targets[max_idx:max_idx+1], targets[idx:max_idx], targets[max_idx+1:]], 0)
        # extra_boxes = tf.concat([boxes[idx:max_idx], boxes[max_idx+1:]], 0)
        ious = bbox_utils.bbox_iou(max_target[:4], targets[idx+1:, :4], xywh=False, iou_type='diou')
        if method == 'normal':
            new_scores = tf.concat([targets[:idx+1, 4], max_target[4], tf.where(ious > iou_threshold, 0., targets[idx+1:, 4])], 0)
        elif method == 'soft_normal':
            new_scores = tf.concat([targets[:idx+1, 4], max_target[4], tf.where(ious >= iou_threshold, targets[..., 4] * (1 - ious), targets[..., 4])], 0)
        elif method == 'soft_gaussian':
            new_scores = tf.concat([targets[:idx+1, 4], tf.exp(-(ious)**2/sigma) * targets[idx+1:, 4]], 0)

        targets = tf.concat([targets[:, :4], new_scores[:, None], targets[:, 5:]], -1)
    
    final_mask = targets[:, 4] >= score_threshold

    return targets[final_mask]
