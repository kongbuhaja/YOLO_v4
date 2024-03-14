import tensorflow as tf
import numpy as np
from utils.bbox_utils import xyxy_to_xywh
from datasets.voc_dataset import Dataset as voc_dataset
from datasets.coco_dataset import Dataset as coco_dataset
from datasets.custom_dataset import Dataset as custom_dataset
from utils.augmentation import resize, resize_padding, random_augmentation

class DataLoader():
    def __init__(self, cfg):
        self.labels = cfg['data']['labels']['name']
        self.row_anchors , self.col_anchors = cfg['model']['anchors'].shape[:2]
        self.num_classes = cfg['data']['labels']['count']
        self.input_size = cfg['model']['input_size'].astype(np.float32)
        self.max_bboxes = cfg['data']['max_bboxes']
        self.seed = cfg['seed']
        self.length = {}

        if cfg['data']['name'] == 'coco':
            self.Dataset = coco_dataset
        elif cfg['data']['name'] == 'voc':
            self.Dataset = voc_dataset
        elif cfg['data']['name'] == 'custom':
            self.Dataset = custom_dataset
    
    def __call__(self, split, batch_size=1, aug=None, cache=True):
        dataset = self.Dataset(split, self.labels)

        data, self.length[split] = dataset.load()

        data = data.map(self.read_image, num_parallel_calls=-1)
        data = data.map(self.normalization, num_parallel_calls=-1) 
        data = data.cache() if cache else data
        data = data.shuffle(buffer_size = self.length[split], seed=self.seed, reshuffle_each_iteration=True) if aug else data

        data = data.map(lambda image, labels: random_augmentation(image, labels, self.input_size, seed=self.seed), num_parallel_calls=-1) if aug else data

        dataset = self.method(split, data, batch_size, aug)
        
        return dataset

    def method(self, split, data, batch_size, aug):
        if aug:
            method = self.mosaic
        else:
            random_pad = True if split=='train' else False
            data = data.map(lambda image, labels: resize_padding(image, labels, self.input_size, random_pad, seed=self.seed), num_parallel_calls=-1)
            method = self.batch

        dataset = tf.data.Dataset.from_generator(
            lambda: method(data, batch_size),
            output_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32))
        )
        return dataset
    
    def batch(self, data, batch_size):
        batch_images = []
        batch_labels = []
        idx = 0
        
        for image, labels in data:
            batch_images += [image]
            batch_labels += [tf.concat([tf.zeros(labels.shape[:-1], dtype=tf.float32)[..., None]+idx, labels], -1)]
            idx += 1
            if len(batch_images) == batch_size:
                yield tf.stack(batch_images, 0), tf.concat(batch_labels, 0)
                batch_images = []
                batch_labels = []
                idx = 0

    def mosaic(self, data, batch_size, size=4):
        batch_images = []
        batch_labels = []
        batch_mosaic_images = []
        batch_mosaic_labels = []
        s = np.sqrt(size).astype(np.int32)
        
        for image, labels in data:
            batch_images += [image]
            batch_labels += [labels]

            if len(batch_images) == batch_size:
                for idx in range(batch_size//size):
                    mosaic_image = np.zeros([*(self.input_size.astype(np.int32)), 3])
                    mosaic_labels = []
                    xc, yc = tf.unstack(tf.cast(tf.random.uniform([2], minval=self.input_size//3, maxval=self.input_size//3*2, seed=self.seed), dtype=tf.int32))
                    # indices = tf.random.uniform([size], minval=0, maxval=batch_size, dtype=tf.int32, seed=self.seed)
                    indices = tf.range(idx*4, (idx+1)*4)

                    for i, index in enumerate(indices):
                        image = batch_images[index]
                        labels = batch_labels[index]
                        w, h = self.input_size[0]*(i%s) - xc*(i%s*s + 1 - s), self.input_size[1]*(i//s) - yc*(i//s*s + 1 - s)
                        x, y = xc*(i%2), yc*(i//2)

                        image, labels = resize_padding(image, labels, tf.cast([w, h], tf.float32), True, seed=self.seed)
                        mosaic_image[y:y+h, x:x+w] = image
                        mosaic_labels += [labels + [x, y, x, y, 0]]

                    batch_mosaic_images += [mosaic_image]
                    mosaic_labels = tf.concat(mosaic_labels, 0)
                    batch_mosaic_labels += [tf.concat([tf.zeros(mosaic_labels.shape[:-1], dtype=tf.float32)[..., None]+idx, mosaic_labels], -1)]
                
                yield tf.stack(batch_mosaic_images, 0), tf.concat(batch_mosaic_labels, 0)
                batch_images = []
                batch_labels = []
                batch_mosaic_images = []
                batch_mosaic_labels = []

    def read_image(self, file, labels):
        image_raw = tf.io.read_file(file)
        image = tf.image.decode_jpeg(image_raw, channels=3)
        return tf.cast(image, tf.float32), labels

    @tf.function
    def normalization(self, image, labels):
        image = tf.cast(image, tf.float32)/255.
        return image, labels
