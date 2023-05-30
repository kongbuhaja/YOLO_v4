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
        # data = data.cache()
        
        if split == 'train':
            pass
            # data = data.shuffle(buffer_size = min(self.length(split) * 3, 50000)) # ram memory limit
            # data = data.map(aug_utils.tf_augmentation, num_parallel_calls=-1)
        
        data = data.map(self.tf_preprocessing, num_parallel_calls=-1)
        data = data.padded_batch(self.batch_size, padded_shapes=get_padded_shapes(), padding_values=get_padding_values(), drop_remainder=True)
        
        # data = data.map(lambda x, y: self.py_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
        data = data.map(lambda x, y: self.tf_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
        return data
    
    def length(self, split):
        return self._length[split]
    
    def py_labels_to_grids(self, image, labels, use_label=False):
        grids = tf.py_function(self.labels_to_grids, [labels], [tf.float32]*self.len_anchors)
        if use_label:
            labels = tf.concat([labels[..., :4], tf.ones_like(labels[..., 4:5]), labels[..., 4:5]], -1)
            return image, *grids, labels
        return image, *grids
    
    @tf.function
    def tf_preprocessing(self, image, labels, width, height):
        image, labels = aug_utils.tf_resize_padding(image, labels, width, height, self.image_size)
        labels = bbox_utils.xyxy_to_xywh(labels, True) 
        labels = tf.concat([labels[..., :4], tf.where(tf.reduce_sum(labels[..., 2:4], -1, keepdims=True)==0, 0., 1.), labels[..., 4:5]], -1)
        return tf.cast(image, tf.float32)/255., labels
    
    @tf.function
    def tf_labels_to_grids(self, image, labels, use_label):
        grids = self.labels_to_grids(labels)
        if use_label:
            return image, *grids, labels
        return image, *grids
        
    @tf.function
    def labels_to_grids(self, labels):
        conf = labels[..., 4:5]
        onehot = tf.where(tf.cast(conf, tf.bool), 0., tf.one_hot(tf.cast(labels[..., 4], dtype=tf.int32), NUM_CLASSES))
        conf_onehot = tf.concat([conf, onehot], -1)

        grids = []
        c_anchors = [tf.reshape(tf.concat([anchor[..., :2] + 0.5, anchor[..., 2:]], -1), [-1, 4]) for anchor in self.anchors]

        anchors = tf.concat([tf.reshape(c_anchors[i] * self.strides[i], [-1,4]) for i in range(self.len_anchors)], 0)

        ious = bbox_utils.bbox_iou(anchors[:, None], labels[:, None, ..., :4])

        # assign similar label
        best_label_iou = tf.reduce_max(ious, -1)
        best_label_idx = tf.argmax(ious, -1)
        positive_label_mask = tf.where(tf.greater_equal(best_label_iou, self.positive_iou_threshold), 1., 0.)

        maximum_bboxes = tf.concat([tf.tile(anchors[None, :, :2], [self.batch_size, 1, 1]),
                                   tf.gather(labels[..., 2:4], best_label_idx, batch_dims=1)], -1) * positive_label_mask[..., None]
        maximum_conf_onehot = tf.gather(conf_onehot, best_label_idx, batch_dims=1) * positive_label_mask[..., None]

        maximum_labels = tf.concat([maximum_bboxes, maximum_conf_onehot], -1)

        # assign minimum label
        best_anchor_iou = tf.reduce_max(ious, -2)
        best_anchor_mask = tf.cast(tf.logical_and(ious == best_anchor_iou[..., None, :], ious > 0.), tf.float32)

        minimum_labels_without_xy = tf.reduce_max(tf.tile(tf.concat([labels[..., 2:4], conf_onehot], -1)[:, None], [1, len(anchors), 1, 1]) * best_anchor_mask[..., None], -2)
        minimum_labels = tf.concat([tf.tile(anchors[None, :, :2], [self.batch_size, 1, 1]), minimum_labels_without_xy], -1)

        # join minimum, maximum label
        minimum_mask = tf.cast(tf.reduce_max(best_anchor_mask, -1, keepdims=True), tf.bool)
        assign_labels = tf.where(minimum_mask, minimum_labels, maximum_labels)

        for i in range(self.len_anchors):
            scale = self.scales[i]
            start = 0 if i==0 else tf.reduce_sum((self.image_size//self.strides[:i])**2 * self.len_anchors)
            end = start + (scale)**2 * self.len_anchors
            grids += [tf.reshape(assign_labels[:, start:end], [self.batch_size, scale, scale, self.len_anchors, -1])]

        return grids
    
def get_padded_shapes():
    return [None, None, None], [MAX_BBOXES, None]

def get_padding_values():
    return tf.constant(0, tf.float32), tf.constant(0, tf.float32)