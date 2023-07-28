import tensorflow as tf
from config import *
from utils import bbox_utils

def prediction_to_bbox(grids, anchors, batch_size=BATCH_SIZE, strides=STRIDES, num_classes=NUM_CLASSES, image_size=IMAGE_SIZE):
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
    bboxes = tf.minimum(tf.maximum(0., bboxes), image_size)

    return tf.concat([bboxes, scores, classes], -1)

def NMS(preds, score_threshold=SCORE_THRESHOLD, iou_threshold=IOU_THRESHOLD, sigma=SIGMA, method=NMS_TYPE):
    # NMS_preds = tf.zeros((0, 6))
    # unique_classes = list(set(preds[..., 5].numpy()))

    valid_mask = preds[..., 4] >= score_threshold
    
    if not tf.reduce_any(valid_mask):
        #return empty
        pass

    targets = preds[valid_mask]
    # boxes = preds[..., :4][valid_mask]
    # scores = preds[..., 4:5][valid_mask]
    # classes = preds[..., 5:][valid_mask]

    # sorted_idx = tf.argsort(targets[..., 4], direction='DESCENDING')
    # boxes = tf.gather(boxes, sorted_idx)
    # scores = tf.gather(scores, sorted_idx)
    # classes = tf.gather(classes, sorted_idx)
    # targets = tf.gather(targets, sorted_idx)

    for idx in range(targets.shape[0]-1):
        max_idx = tf.argmax(targets[..., 4][idx:], -1)
        max_target = targets[max_idx]
        # max_score = targets[..., 4][max_idx]
        # max_box = targets[..., :4][max_idx]
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

     


    # for unique_class in unique_classes:
    #     class_idx = tf.reshape(tf.where(tf.logical_and(preds[..., 5] == unique_class, preds[..., 4] >= score_threshold)), [-1])
    #     targets = tf.gather(preds, class_idx)
    #     while(targets.shape[0]):
    #         max_index = tf.argmax(targets[..., 4])
    #         max_target = targets[max_index][None]
    #         NMS_preds = tf.concat([NMS_preds, max_target], 0)

    #         targets = tf.concat([targets[:max_index], targets[max_index+1:]], 0)
            
    #         ious = bbox_utils.bbox_iou(max_target[..., :4], targets[..., :4], xywh=False, iou_type='diou')
    #         if method == 'normal':
    #             target_scores = tf.where(ious >= iou_threshold, 0, targets[..., 4])
    #         elif method == 'soft_normal':
    #             target_scores = tf.where(ious >= iou_threshold, targets[..., 4] * (1 - ious), targets[..., 4])
    #         elif method == 'soft_gaussian':
    #             target_scores = tf.exp(-(ious)**2/sigma) * targets[..., 4]
                
    #         filter = tf.reshape(tf.where(target_scores >= score_threshold), [-1])
    #         targets = tf.gather(targets, filter)
    #         target_scores = tf.gather(target_scores, filter)
    #         targets = tf.concat([targets[..., :4], target_scores[..., None], targets[..., 5:]], -1)
    
    # return NMS_preds