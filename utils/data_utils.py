import tensorflow as tf
import numpy as np
from utils.bbox_utils import xyxy_to_xywh
from datasets.voc_dataset import Dataset as voc_dataset
from datasets.coco_dataset import Dataset as coco_dataset
from datasets.custom_dataset import Dataset as custom_dataset
from utils.augmentation import resize_padding, random_augmentation

class DataLoader():
    def __init__(self, cfg):
        self.labels = cfg['data']['labels']['name']
        self.batch_size = cfg['batch_size']
        self.row_anchors , self.col_anchors = cfg['model']['anchors'].shape[:2]
        self.num_classes = cfg['data']['labels']['count']
        self.input_size = cfg['model']['input_size'].astype(np.float32)
        self.anchors = cfg['model']['anchors']
        self.max_bboxes = cfg['data']['max_bboxes']
        self.seed = cfg['seed']
        self.length = {}

        if cfg['data']['name'] == 'coco':
            self.Dataset = coco_dataset
        elif cfg['data']['name'] == 'voc':
            self.Dataset = voc_dataset
        elif cfg['data']['name'] == 'custom':
            self.Dataset = custom_dataset
    
    def __call__(self, split, shuffle=True, augmentation=False, resize=True, cache=True):
        dataset = self.Dataset(split, self.anchors, self.labels)

        data, self.length[split] = dataset.load()
        data = data.cache() if cache else data
        data = data.shuffle(buffer_size = self.length[split], seed=self.seed, reshuffle_each_iteration=True) if shuffle else data

        data = data.map(self.read_image, num_parallel_calls=-1)
        data = data.map(self.normalization, num_parallel_calls=-1) 
        data = data.map(lambda image, labels: random_augmentation(image, labels, self.input_size, seed=self.seed), num_parallel_calls=-1) if augmentation else data
        data = data.map(lambda image, labels: resize_padding(image, labels, self.input_size, augmentation, seed=self.seed), num_parallel_calls=-1) if resize else data

        data = data.padded_batch(self.batch_size, padded_shapes=self.get_padded_shapes(), padding_values=self.get_padding_values(), drop_remainder=True)

        data = data.map(lambda image, labels: self.squeeze(image, labels), num_parallel_calls=-1).prefetch(1)

        return data
    
    @tf.function
    def read_image(self, file, labels):
        image_raw = tf.io.read_file(file)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        return tf.cast(image, tf.float32), labels

    @tf.function
    def normalization(self, image, labels):
        image = tf.cast(image, tf.float32)/255.
        return image, labels

    @tf.function
    def squeeze(self, image, labels):
        idx = tf.tile(tf.range(self.batch_size, dtype=tf.float32)[:, None, None], [1, self.max_bboxes, 1])
        mask = tf.logical_and(labels[..., 2]!=0, labels[..., 3]!=0)
        labels = xyxy_to_xywh(labels[mask], True)
        idx = idx[mask]
        squeezed_labels = tf.concat([idx, labels], -1)
        return image, squeezed_labels

    @tf.function
    def get_padded_shapes(self):
        return [None, None, None], [self.max_bboxes, None]

    @tf.function
    def get_padding_values(self):
        return tf.constant(0, tf.float32), tf.constant(0, tf.float32)