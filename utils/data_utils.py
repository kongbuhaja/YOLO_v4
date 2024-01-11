import tensorflow as tf
import numpy as np
from utils import aug_utils, bbox_utils, anchor_utils
from utils.aug_utils import augmentation, resize_padding

class DataLoader():
    def __init__(self, dtype, labels, batch_size, anchors, input_size, 
                 strides, positive_iou_threshold, max_bboxes, create_anchors):
        self.dtype = dtype
        self.labels = labels
        self.batch_size = batch_size
        self.col_anchors = len(anchors[0])
        self.row_anchors = len(anchors)
        self.num_classes = len(labels)
        self.input_size = input_size.astype(np.float32)
        self.strides = np.array(strides)
        self.scales = (self.input_size[None] // self.strides[:, None])
        self.anchors = tf.constant(anchors, tf.float32) * self.input_size[None, None]
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
        dataset = Dataset(split, self.dtype, self.anchors, self.labels, self.create_anchors)

        data = dataset.load(use_tfrecord)
        self._length[split] = dataset.length

        # data = data.cache()

        # if split == 'train':
            # tf.random.set_seed(42)
            # data = data.shuffle(buffer_size = min(self.length(split) * 3, 200000), seed=42)
            
        # # if you have enough ram move this line before data.cache(), it will be faster

        data = data.map(self.normalization, num_parallel_calls=-1) 
        data = data.map(lambda image, labels, width, height: self.tf_augmentation(image, labels, width, height, self.input_size, split), num_parallel_calls=-1)
        data = data.map(self.tf_minmax, num_parallel_calls=-1)

        # data = data.map(self.expand_confidence, num_parallel_calls=-1)
        data = data.padded_batch(self.batch_size, padded_shapes=self.get_padded_shapes(), padding_values=self.get_padding_values(), drop_remainder=True)

        # # data = data.map(lambda x, y: self.py_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
        data = data.map(lambda image, labels: self.encode(image, labels), num_parallel_calls=-1).prefetch(1)
        # data = data.map(lambda image, labels: self.sampling(image, labels, use_label), num_parallel_calls=-1).prefetch(1)
        return data
    
    def length(self, split):
        return self._length[split]
    
    def py_encode(self, image, labels, use_label):
        grids = tf.py_function(self.encode, [labels], [tf.float32]*self.row_anchors)
        if use_label:
            return image, *grids, labels
        return image, *grids

    @tf.function
    def tf_augmentation(self, image, labels, width, height, input_size, split):
        if split == 'train':
            image, labels, width, height = augmentation(image, labels, width, height, input_size)
            image, labels = resize_padding(image, labels, width, height, input_size, random=True)
        else:
            image, labels = resize_padding(image, labels, width, height, input_size)
        return image, labels

    @tf.function
    def normalization(self, image, labels, width, height):
        image = tf.cast(image, tf.float32)/255.
        return image, labels, width, height
    
    @tf.function
    def tf_minmax(self, image, labels):
        return tf.maximum(tf.minimum(image, 1.), 0), labels
    
    @tf.function
    def sampling(self, image, labels, use_label):
        positive_samples = self.positive_samping(labels)
        if use_label:
            return image, *positive_samples, labels
        return image, *positive_samples
    
    @tf.function
    def onehot_label(self, label, smooth=True, alpha=0.1):
        onehot = tf.one_hot(tf.cast(label, dtype=tf.int32), self.num_classes)
        if smooth:
            return onehot * (1. - alpha) + alpha/self.num_classes
        return onehot

    @tf.function
    def encode(self, image, labels):
        # Squeeze GT
        idx = tf.tile(tf.range(self.batch_size, dtype=tf.float32)[:, None, None], [1, self.max_bboxes, 1])
        mask = tf.logical_and(labels[..., 2]!=0, labels[..., 3]!=0)
        labels = bbox_utils.xyxy_to_xywh(labels[mask], True)
        idx = idx[mask]
        encoded_labels = tf.concat([idx, labels], -1)
        return image, encoded_labels

    @tf.function
    def get_padded_shapes(self):
        return [None, None, None], [self.max_bboxes, None]

    @tf.function
    def get_padding_values(self):
        return tf.constant(0, tf.float32), tf.constant(0, tf.float32)