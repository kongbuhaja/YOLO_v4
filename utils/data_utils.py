import tensorflow as tf
from config import *
import numpy as np
from utils import aug_utils, bbox_utils, anchor_utils

class DataLoader():
    def __init__(self, dtype=DTYPE, batch_size=BATCH_SIZE, anchors=ANCHORS, num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES,
                 image_size=IMAGE_SIZE, strides=STRIDES, iou_threshold=IOU_THRESHOLD, max_bboxes=MAX_BBOXES):
                 
        self.batch_size = batch_size
        self.num_anchors = num_anchors
        self.len_anchors = len(ANCHORS)
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = np.array(strides)
        self.anchors = anchor_utils.get_anchors_xywh(anchors, self.strides, self.image_size)
        self.iou_threshold = iou_threshold
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
            data = data.shuffle(buffer_size = min(self.length(split) * 3, 10000)) # ram memory limit
            data = data.map(aug_utils.tf_augmentation, num_parallel_calls=-1)
        
        data = data.map(self.tf_preprocessing, num_parallel_calls=-1)
        data = data.padded_batch(self.batch_size, padded_shapes=get_padded_shapes(), padding_values=get_padding_values(), drop_remainder=True)
        
        data = data.map(lambda x, y: self.py_labels_to_grids(x, y, use_label), num_parallel_calls=-1).prefetch(1)
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
        return tf.cast(image, tf.float32)/255., labels
    
    @tf.function
    def tf_labels_to_grids(self, image, labels): # not working
        grids = self.labels_to_grids2(labels)
        return image, *grids
        
    def labels_to_grids(self, labels):
        grids = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 5+self.num_classes)) for stride in self.strides]
        ious = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 100), dtype=tf.float32) for stride in self.strides]
        best_ious = tf.zeros((self.batch_size, self.max_bboxes))
    
        no_obj = (tf.reduce_sum(labels[..., 2:4], -1) == 0)[..., None]
        conf = tf.cast(tf.where(no_obj, tf.zeros_like(no_obj), tf.ones_like(no_obj)), tf.float32)
        onehot = tf.where(no_obj, tf.zeros_like(conf), tf.one_hot(tf.cast(labels[..., 4], dtype=tf.int32), NUM_CLASSES))
        conf_onehot = tf.concat([conf, onehot], -1)
        
        labels = bbox_utils.xyxy_to_xywh(labels, True)
        for i in range(self.len_anchors):
            anchor = tf.concat([self.anchors[i][..., :2] + 0.5, self.anchors[i][..., 2:]],-1)
            scaled_bboxes = labels[..., :4] / self.strides[i]
            
            ious[i] = bbox_utils.bbox_iou(anchor[..., None,:], scaled_bboxes[:,None,None,None], iou_type='iou')
            
            new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
            
            max_ious_id = tf.argmax(ious[i], -1)
     
            max_ious_mask = tf.cast(tf.reduce_max(ious[i], -1)[..., None] >= self.iou_threshold, tf.float32)
            grids[i] = tf.gather(new_labels, max_ious_id, batch_dims=1) * max_ious_mask
            
            best_ious = tf.maximum(tf.reduce_max(ious[i], [1,2,3]), best_ious)

        if tf.reduce_any(best_ious < self.iou_threshold):
            for i in range(self.len_anchors):
                anchor = tf.concat([self.anchors[i][..., :2] + 0.5, self.anchors[i][..., 2:]],-1)
                
                scaled_bboxes = labels[..., :4] / self.strides[i]
                
                non_zero_ious_mask = tf.cast(tf.where(best_ious!=0, tf.ones_like(best_ious), tf.zeros_like(best_ious)), tf.bool)[:,None,None,None]

                best_mask = tf.reduce_any(tf.math.logical_and((ious[i] == best_ious[:,None,None,None]), non_zero_ious_mask), -1)[..., None]
                
                if tf.reduce_any(best_mask):
                    new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
                    
                    max_ious_id = tf.argmax(ious[i], -1)
                    best_masked_grid = tf.gather(new_labels, max_ious_id, batch_dims=1) * tf.cast(best_mask, tf.float32)
                    
                    grids[i] = tf.where(best_mask, best_masked_grid, grids[i])
        return grids
    
    @tf.function
    def labels_to_grids2(self, labels):
        # grids = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 5+self.num_classes)) for stride in self.strides]
        grids = [tf.TensorArray(tf.float32, 1) for stride in self.strides]
        best_ious = tf.zeros((self.batch_size, self.max_bboxes))
        ious = [tf.zeros((self.batch_size, self.image_size//stride, self.image_size//stride , self.num_anchors, 100), dtype=tf.float32) for stride in self.strides]
    
        no_obj = tf.reduce_sum(labels[..., 2:4], -1) == 0
        conf = tf.cast(tf.where(no_obj, tf.zeros_like(no_obj), tf.ones_like(no_obj)), tf.float32)[..., None]
        onehot = tf.where(no_obj[..., None], tf.zeros_like(conf), tf.one_hot(tf.cast(labels[..., 4], dtype=tf.int32), NUM_CLASSES))
        conf_onehot = tf.concat([conf, onehot], -1)
        
        for i in range(self.len_anchors):
            anchor = tf.concat([self.anchors[i][..., :2] + 0.5, self.anchors[i][..., 2:]],-1)
            scaled_bboxes = labels[..., :4] / self.strides[i]
            
            ious[i] = bbox_utils.bbox_iou(anchor[..., None,:], scaled_bboxes[:,None,None,None], iou_type='iou')

            new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
            
            max_ious_id = tf.argmax(ious[i], -1)
            max_ious_mask = tf.cast(tf.reduce_max(ious[i], -1)[..., None] >= self.iou_threshold, tf.float32)
            # grids[i] = tf.gather(new_labels, max_ious_id, batch_dims=1) * max_ious_mask
            grids[i].write(0, tf.gather(new_labels, max_ious_id, batch_dims=1) * max_ious_mask)
            best_ious = tf.maximum(tf.reduce_max(ious[i], [1,2,3]), best_ious)
        
        if tf.reduce_any(best_ious < self.iou_threshold):
            for i in range(self.len_anchors):
                anchor = tf.concat([self.anchors[i][..., :2] + 0.5, self.anchors[i][..., 2:]],-1)
                
                scaled_bboxes = labels[..., :4] / self.strides[i]
                
                non_zero_ious_mask = tf.cast(tf.where(best_ious!=0, tf.ones_like(best_ious), tf.zeros_like(best_ious)), tf.bool)[:,None,None,None]
                best_mask = tf.reduce_any(tf.math.logical_and((ious[i] == best_ious[:,None,None,None]), non_zero_ious_mask), -1)[..., None]
      
                if tf.reduce_any(best_mask):
                    new_labels =  tf.concat([scaled_bboxes, conf_onehot], -1)
                    max_ious_id = tf.argmax(ious[i], -1)
                    best_masked_grid = tf.gather(new_labels, max_ious_id, batch_dims=1) * tf.cast(best_mask, tf.float32)

                    # grids[i] = tf.where(best_mask, best_masked_grid, grids[i])
                    grids[i].write(0, tf.where(best_mask, best_masked_grid, grids[i].read(0)))
        return grids[0].read(0), grids[1].read(0), grids[2].read(0)
    
def get_padded_shapes():
    return [None, None, None], [MAX_BBOXES, None]

def get_padding_values():
    return tf.constant(0, tf.float32), tf.constant(0, tf.float32)