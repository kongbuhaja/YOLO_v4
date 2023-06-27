import tensorflow as tf
from config import *
import numpy as np
from utils import aug_utils, bbox_utils, anchor_utils

class DataLoader():
    def __init__(self, dtype=DTYPE, batch_size=GLOBAL_BATCH_SIZE, anchors=ANCHORS, num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES,
                 image_size=IMAGE_SIZE, strides=STRIDES, positive_iou_threshold=POSITIVE_IOU_THRESHOLD, max_bboxes=MAX_BBOXES):
                 
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.len_anchors = len(ANCHORS)
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = np.array(strides)
        self.scales = image_size//self.strides
        self.anchors = anchor_utils.get_anchors_xywh(anchors, self.strides, self.image_size)
        self.positive_iou_threshold = positive_iou_threshold
        self.max_bboxes = max_bboxes
        self.dtype = dtype
        self._length = {}

    def __call__(self, split, use_tfrecord=True, use_label=False):
        if self.dtype == 'voc':
            from datasets.voc_dataset import Dataset
        elif self.dtype == 'coco':
            from datasets.coco_dataset import Dataset
        elif self.dtype == 'custom':
            from datasets.custom_dataset import Dataset
        elif self.dtype == 'raw':
            from datasets.raw_dataset import Dataset
        dataset = Dataset(split)

        data = dataset.load(use_tfrecord)
        self._length[split] = dataset.length

        data = data.map(self.tf_preprocessing, num_parallel_calls=-1)
        data = data.cache()
        
        if split == 'train':
            data = data.shuffle(buffer_size = min(self.length(split) * 3, 200000)) # ram memory limit
            # data = data.map(aug_utils.tf_augmentation, num_parallel_calls=-1)
        
        data = data.map(self.tf_resize_padding, num_parallel_calls=-1)
        data = data.padded_batch(self.batch_size, padded_shapes=get_padded_shapes(), padding_values=get_padding_values(), drop_remainder=True)
        
        # data = data.map(lambda x, y: self.py_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
        data = data.map(lambda x, y: self.tf_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
        return data
    
    def length(self, split):
        return self._length[split]
    
    def py_labels_to_grids(self, image, labels, use_label=False):
        grids = tf.py_function(self.labels_to_grids, [labels], [tf.float32]*self.len_anchors)
        if use_label:
            return image, *grids, labels
        return image, *grids
    
    @tf.function
    def tf_preprocessing(self, image, labels, width, height):
        labels = tf.concat([labels[..., :4], tf.ones_like(labels[..., 4:5]), labels[..., 4:5]], -1)
        return tf.cast(image, tf.float32)/255., labels, width, height
    
    @tf.function
    def tf_resize_padding(self, image, labels, width, height):
        image, labels = aug_utils.tf_resize_padding(image, labels, width, height, self.image_size)
        return image, labels
    
    @tf.function
    def tf_labels_to_grids(self, image, labels, use_label):
        grids = self.labels_to_grids(labels)
        if use_label:
            return image, *grids, labels
        return image, *grids
    
    @tf.function
    def onehot_label(self, prob, smooth=True, alpha=0.1):
        onehot = tf.one_hot(tf.cast(prob, dtype=tf.int32), self.num_classes)
        if smooth:
            return onehot * (1. - alpha) + alpha/self.num_classes
        return onehot

    @tf.function
    def labels_to_grids(self, labels):
        labels = bbox_utils.xyxy_to_xywh(labels, True)
        wh = labels[..., 2:4]
        conf = labels[..., 4:5]

        onehot = conf * self.onehot_label(labels[..., 5], smooth=True)

        grids = []
        anchor_xy = [tf.reshape(anchor[..., :2], [-1,2]) for anchor in self.anchors]
        anchor_wh = [tf.reshape(anchor[..., 2:], [-1,2]) for anchor in self.anchors]

        center_anchors = tf.concat([tf.concat([anchor_xy[i] + 0.5, anchor_wh[i]], -1) * self.strides[i] for i in range(self.len_anchors)], 0)
        # base_anchor_xy = tf.concat([tf.concat([anchor_xy[i], anchor_wh[i]], -1) * self.strides[i] for i in range(self.len_anchors)], 0)
        base_anchor_xy = tf.concat([tf.concat([anchor_xy[i]], -1) * self.strides[i] for i in range(self.len_anchors)], 0)
        ious = bbox_utils.bbox_iou(center_anchors[:, None], labels[:, None, ..., :4])

        # assign maximum label
        maximum_positive_ious = tf.where(tf.greater_equal(ious, self.positive_iou_threshold), ious, 0.)

        # assign minimum label
        best_anchor_iou = tf.reduce_max(ious, -2, keepdims=True)
        minimum_positive_ious = tf.where(tf.logical_and(ious == best_anchor_iou, ious > 0), best_anchor_iou, 0.)
    
        # join minimum, maximum label
        joined_ious = tf.where(tf.cast(minimum_positive_ious, tf.bool), minimum_positive_ious, maximum_positive_ious)
        joined_positive_mask = tf.cast(tf.reduce_any(tf.cast(joined_ious, tf.bool), -1, keepdims=True), tf.float32)

        # assigned_labels = tf.concat([tf.tile(base_anchor_xy[None], [self.batch_size, 1,1]), tf.gather(tf.concat([ conf, onehot], -1), tf.argmax(joined_ious, -1), batch_dims=1) * joined_positive_mask], -1)
        # assigned_labels = tf.concat([tf.tile(base_anchor_xy[None], [self.batch_size, 1,1]), tf.gather(tf.concat([wh, conf, onehot], -1), tf.argmax(joined_ious, -1), batch_dims=1) * joined_positive_mask], -1)
        assigned_labels = tf.gather(tf.concat([labels[..., :5], onehot],-1), tf.argmax(joined_ious, -1), batch_dims=1) * joined_positive_mask

        for i in range(self.len_anchors):
            scale = self.scales[i]
            start = 0 if i==0 else tf.reduce_sum((self.image_size//self.strides[:i])**2 * self.len_anchors)
            end = start + (scale)**2 * self.len_anchors
            grids += [tf.reshape(assigned_labels[:, start:end], [self.batch_size, scale, scale, self.len_anchors, -1])]

        return grids
    
def get_padded_shapes():
    return [None, None, None], [MAX_BBOXES, None]

def get_padding_values():
    return tf.constant(0, tf.float32), tf.constant(0, tf.float32)