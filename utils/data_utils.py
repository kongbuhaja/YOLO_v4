import tensorflow as tf
import numpy as np
from utils import aug_utils, bbox_utils, anchor_utils

class DataLoader():
    def __init__(self, dtype, labels, batch_size, anchors, num_classes, 
                 input_size, strides, positive_iou_threshold, max_bboxes, 
                 create_anchors):
        self.dtype = dtype
        self.labels = labels
        self.batch_size = batch_size
        self.col_anchors = len(anchors[0])
        self.row_anchors = len(anchors)
        self.num_classes = num_classes
        self.input_size = input_size
        self.strides = np.array(strides)
        self.scales = (self.input_size // np.array(self.strides)).tolist()
        self.anchors = anchors
        self.anchors_xywh = anchor_utils.get_anchors_xywh(anchors, self.strides, self.input_size)
        self.positive_iou_threshold = positive_iou_threshold
        self.max_bboxes = max_bboxes
        self.create_anchors = create_anchors
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
        dataset = Dataset(split, self.dtype, self.anchors, self.labels, self.input_size, self.create_anchors)

        data = dataset.load(use_tfrecord)
        self._length[split] = dataset.length

        data = data.cache()

        if split == 'train':
            data = data.shuffle(buffer_size = min(self.length(split) * 3, 200000))
            
        # if you have enough ram move this line before data.cache(), it will be faster
        data = data.map(self.tf_preprocessing, num_parallel_calls=-1) 

        if split == 'train':
            data = data.map(aug_utils.tf_augmentation, num_parallel_calls=-1)
            data = data.map(self.tf_minmax, num_parallel_calls=-1)
        
        data = data.map(lambda image, labels, width, height: aug_utils.tf_resize_padding(image, labels, width, height, self.input_size), num_parallel_calls=-1)
        data = data.padded_batch(self.batch_size, padded_shapes=self.get_padded_shapes(), padding_values=self.get_padding_values(), drop_remainder=True)
        
        # data = data.map(lambda x, y: self.py_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
        data = data.map(lambda image, labels: self.tf_encode(image, labels, use_label), num_parallel_calls=-1).prefetch(1)
        return data
    
    def length(self, split):
        return self._length[split]
    
    def py_encode(self, image, labels, use_label):
        grids = tf.py_function(self.encode, [labels], [tf.float32]*self.row_anchors)
        if use_label:
            return image, *grids, labels
        return image, *grids
    
    @tf.function
    def tf_preprocessing(self, image, labels, width, height):
        image = tf.cast(image, tf.float32)/255.
        return image, labels, width, height
    
    def tf_minmax(self, image, labels, width, height):
        return tf.maximum(tf.minimum(image, 1.), 0), labels, width, height
    
    @tf.function
    def tf_resize_padding(self, image, labels, width, height):
        image, labels = aug_utils.tf_resize_padding(image, labels, width, height, self.input_size)

        return image, labels
    
    @tf.function
    def tf_encode(self, image, labels, use_label):
        grids = self.encode(labels)
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
    def encode(self, labels):
        labels = bbox_utils.xyxy_to_xywh(labels, True)
        conf = labels[..., 4:5]

        onehot = conf * self.onehot_label(labels[..., 5], smooth=True)

        grids = []
        anchor_xy = [tf.reshape(anchor[..., :2], [-1,2]) for anchor in self.anchors_xywh]
        anchor_wh = [tf.reshape(anchor[..., 2:], [-1,2]) for anchor in self.anchors_xywh]

        center_anchors = tf.concat([tf.concat([anchor_xy[i] + 0.5, anchor_wh[i]], -1) * self.strides[i] for i in range(self.row_anchors)], 0)

        ious = bbox_utils.bbox_iou(center_anchors[:, None], labels[:, None, ..., :4])

        # assign maximum label
        maximum_positive_ious = tf.where(tf.greater_equal(ious, self.positive_iou_threshold), ious, 0.)

        # assign minimum label
        best_anchor_iou = tf.reduce_max(ious, -2, keepdims=True)
        minimum_positive_ious = tf.where(tf.logical_and(ious == best_anchor_iou, ious > 0), best_anchor_iou, 0.)
    
        # join minimum, maximum label
        joined_ious = tf.where(tf.cast(minimum_positive_ious, tf.bool), minimum_positive_ious, maximum_positive_ious)
        joined_positive_mask = tf.cast(tf.reduce_any(tf.cast(joined_ious, tf.bool), -1, keepdims=True), tf.float32)

        assigned_labels = tf.gather(tf.concat([labels[..., :5], onehot],-1), tf.argmax(joined_ious, -1), batch_dims=1) * joined_positive_mask

        for i in range(self.row_anchors):
            scale = self.scales[i]
            start = 0 if i==0 else tf.reduce_sum((self.input_size//self.strides[:i])**2 * self.col_anchors)
            end = start + (scale)**2 * self.col_anchors
            grids += [tf.reshape(assigned_labels[:, start:end], [self.batch_size, scale, scale, self.col_anchors, -1])]

        return grids

    @tf.function
    def encode2(self, labels):
        labels = bbox_utils.xyxy_to_xywh(labels, True)
        conf = labels[..., 4:5]

        onehot = conf * self.onehot_label(labels[..., 5], smooth=True)

        grids = []
        anchor_xy = [tf.reshape(anchor[..., :2], [-1,2]) for anchor in self.anchors_xywh]
        anchor_wh = [tf.reshape(anchor[..., 2:], [-1,2]) for anchor in self.anchors_xywh]

        center_anchors = tf.concat([tf.concat([anchor_xy[i] + 0.5, anchor_wh[i]], -1) * self.strides[i] for i in range(self.row_anchors)], 0)

        ious = bbox_utils.bbox_iou(center_anchors[:, None], labels[:, None, ..., :4])

        # assign maximum label
        maximum_positive_ious = tf.where(tf.greater_equal(ious, self.positive_iou_threshold), ious, 0.)     

        # assign minimum label
        best_anchor_iou = tf.reduce_max(ious, -2, keepdims=True)
        minimum_positive_ious = tf.where(tf.logical_and(ious == best_anchor_iou, ious > 0), best_anchor_iou, 0.)
    
        # join minimum, maximum label
        joined_ious = tf.where(tf.cast(minimum_positive_ious, tf.bool), minimum_positive_ious, maximum_positive_ious)
        joined_positive_mask = tf.cast(tf.reduce_any(tf.cast(joined_ious, tf.bool), -1, keepdims=True), tf.float32)

        assigned_labels = tf.gather(tf.concat([labels[..., :5], onehot],-1), tf.argmax(joined_ious, -1), batch_dims=1) * joined_positive_mask
        grid_xy = tf.concat([anchor_xy[i] for i in range(self.row_anchors)], 0)
        tiled_strides = tf.cast(tf.concat([tf.tile(stride[None, None, None], [self.batch_size, scale**2*self.col_anchors, 1]) for stride, scale in zip(self.strides, self.scales)], 1), tf.float32)
        based_xy = (grid_xy[None] - tf.minimum(tf.maximum(grid_xy[None] - assigned_labels[..., :2] / tiled_strides, -1), 0)) * joined_positive_mask * tiled_strides
        assigned_labels = tf.concat([based_xy, assigned_labels[..., 2:]], -1)

        for i in range(self.row_anchors):
            scale = self.scales[i]
            start = 0 if i==0 else tf.reduce_sum((self.input_size//self.strides[:i])**2 * self.col_anchors)
            end = start + (scale)**2 * self.col_anchors
            grids += [tf.reshape(assigned_labels[:, start:end], [self.batch_size, scale, scale, self.col_anchors, -1])]

        return grids
    
    @tf.function
    def get_padded_shapes(self):
        return [None, None, None], [self.max_bboxes, None]

    @tf.function
    def get_padding_values(self):
        return tf.constant(0, tf.float32), tf.constant(0, tf.float32)