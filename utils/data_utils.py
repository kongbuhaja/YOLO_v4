import tensorflow as tf
import numpy as np
from utils.bbox_utils import xyxy_to_xywh
from datasets.voc_dataset import Dataset as voc_dataset
from datasets.coco_dataset import Dataset as coco_dataset
from datasets.custom_dataset import Dataset as custom_dataset
from utils.aug_utils import resize, resize_padding, mosaic_augmentation, batch_augmentation, crop

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
        data = data.shuffle(buffer_size = self.length[split], seed=self.seed, reshuffle_each_iteration=True) if aug else data
        data = data.cache() if cache else data

        data = data.map(self.preprocess, num_parallel_calls=-1)
        data = self.augmentation(data, aug, seed=self.seed)
        dataset = self.method(split, data, batch_size, aug, seed=self.seed)
        dataset = dataset.prefetch(-1)

        return dataset
    
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
    def preprocess(self, file, labels):
        image, labels = self.read_image(file, labels)
        image, labels = self.normalization(image, labels)
        image, labels = resize(image, labels, self.input_size)
        return image, labels
    
    def augmentation(self, data, aug, seed=42):
        if aug:
            augmentation = mosaic_augmentation if aug['mosaic'] else batch_augmentation
            return data.map(lambda image, labels: augmentation(image, labels, aug, seed=seed), num_parallel_calls=-1)
        return data

    def method(self, split, data, batch_size, aug, seed=42):
        if aug:
            if aug['mosaic']:
                method = self.mosaic
            else:
                random_pad = True if split=='train' else False
                data = data.map(lambda image, labels: resize_padding(image, labels, self.input_size, random_pad, seed=seed), num_parallel_calls=-1)
                method = self.batch
        else:
            random_pad = True if split=='train' else False
            data = data.map(lambda image, labels: resize_padding(image, labels, self.input_size, random_pad, seed=seed), num_parallel_calls=-1)
            method = self.batch

        dataset = tf.data.Dataset.from_generator(
            lambda: method(data, batch_size, seed=seed),
            output_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None], dtype=tf.float32))
        )
        return dataset
    
    def batch(self, data, batch_size, seed=42):
        batch_images = []
        batch_labels = []

        for image, labels in data:
            for idx in range(batch_size):
                batch_images += [image]
                batch_labels += [tf.concat([tf.zeros(labels.shape[:-1], dtype=tf.float32)[..., None]+idx, labels], -1)]
            yield tf.stack(batch_images, 0), tf.concat(batch_labels, 0)
            
            batch_images = []
            batch_labels = []

    def mosaic(self, data, batch_size, size=4, seed=42):
        batch_images = []
        batch_labels = []
        batch_mosaic_images = []
        batch_mosaic_labels = []
        s = np.sqrt(size).astype(np.int32)
        mosaic_size = self.input_size * 2
        ix1, iy1, ix2, iy2 = *(mosaic_size//size), *(mosaic_size//size*(size-1))
        
        for image, labels in data:
            batch_images += [image]
            batch_labels += [labels]

            if len(batch_images) == batch_size:
                for idx in range(batch_size//size):
                    mosaic_image = np.zeros([*(mosaic_size.astype(np.int32)), 3])
                    mosaic_labels = []
                    xc, yc = tf.unstack(tf.cast(tf.random.uniform([2], 
                                                                  minval=mosaic_size//(2**size)*(2**(size-1)-1), 
                                                                  maxval=mosaic_size//(2**size)*(2**(size-1)+1), 
                                                                  seed=seed), dtype=tf.int32))
                    xcf, ycf = tf.cast(xc, tf.float32), tf.cast(yc, tf.float32)
                    # indices = tf.random.uniform([size], minval=0, maxval=batch_size, dtype=tf.int32, seed=seed)
                    indices = tf.range(idx*size, (idx+1)*size)

                    for i, index in enumerate(indices):
                        image = batch_images[index]
                        labels = batch_labels[index]
                        width, height = mosaic_size[0] - xcf if i%2 else xcf, mosaic_size[1] - ycf if i//2 else ycf
                        image, labels = resize(image, labels, tf.stack([width, height]))

                        h, w = tf.unstack(image.shape[:2])
                        x1, y1 = xc if i%s else xc - w, yc if i//s else yc - h
                        x2, y2 = x1+w, y1+h
 
                        mosaic_image[y1:y2, x1:x2] = image
                        mosaic_labels += [labels + [x1, y1, x1, y1, 0]]

                    mosaic_labels = tf.concat(mosaic_labels, 0)
                    crop_image, crop_labels = crop(mosaic_image, mosaic_labels, tf.stack([ix1, iy1, ix2, iy2]))
                    batch_mosaic_images += [crop_image]
                    batch_mosaic_labels += [tf.concat([tf.zeros(crop_labels.shape[:-1], dtype=tf.float32)[..., None]+idx, crop_labels], -1)]
                    
                yield tf.stack(batch_mosaic_images, 0), tf.concat(batch_mosaic_labels, 0)
                batch_images = []
                batch_labels = []
                batch_mosaic_images = []
                batch_mosaic_labels = []


