import tensorflow as tf
import numpy as np
from datasets.voc_dataset import Dataset as voc_dataset
from datasets.coco_dataset import Dataset as coco_dataset
from datasets.custom_dataset import Dataset as custom_dataset
from utils.aug_utils import resize, resize_padding, mosaic_augmentation, batch_augmentation, crop
from utils.bbox_utils import xyxy_to_xywh

class DataLoader():
    def __init__(self, cfg):
        self.labels = cfg['data']['labels']['name']
        self.row_anchors , self.col_anchors = cfg['model']['anchors'].shape[:2]
        self.num_classes = cfg['data']['labels']['count']
        self.input_size = cfg['model']['input_size'].astype(np.float32)
        self.seed = cfg['seed']
        self.length = {}

        if cfg['data']['name'] == 'coco':
            self.Dataset = coco_dataset
        elif cfg['data']['name'] == 'voc':
            self.Dataset = voc_dataset
        elif cfg['data']['name'] == 'custom':
            self.Dataset = custom_dataset
    
    def __call__(self, split, batch_size=1, aug=None, resize=True, cache=True):
        dataset = self.Dataset(split, self.labels)

        data, self.length[split] = dataset.load()
        data = data.cache() if cache else data
        data = data.shuffle(buffer_size = self.length[split], seed=self.seed, reshuffle_each_iteration=True) if aug else data
        
        data = data.map(lambda image, labels: self.preprocess(image, labels, split), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = self.augmentation(data, aug, seed=self.seed)
        data = self.method(split, data, batch_size, aug, resize, seed=self.seed)
        data = data.map(self.xyxy_to_xywh, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        data = data.prefetch(tf.data.experimental.AUTOTUNE)

        return data
    
    @tf.function
    def read_image(self, file, labels):
        image_raw = tf.io.read_file(file)
        image = tf.cast(tf.image.decode_jpeg(image_raw, channels=3), tf.float32)
        return image, labels

    @tf.function
    def normalization(self, image, labels):
        image = image/255.
        return image, labels

    @tf.function
    def preprocess(self, file, labels, split):
        image, labels = self.read_image(file, labels)
        image, labels = self.normalization(image, labels)
        # image, labels = resize(image, labels, self.input_size) if split=='train' else (image, labels)
        return image, labels
    
    @tf.function
    def xyxy_to_xywh(self, image, labels):
        labels = xyxy_to_xywh(labels, start=1)
        return image, labels
    
    def augmentation(self, data, aug, seed=42):
        if aug:
            augmentation = mosaic_augmentation if aug['mosaic'] else batch_augmentation
            return data.map(lambda image, labels: augmentation(image, labels, aug, seed=seed), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return data

    def method(self, split, data, batch_size, aug, resize, seed=42):
        if aug:
            if aug['mosaic']:
                self.mosaic_size = tf.round(self.input_size * 1.5)
                ix1, iy1 = tf.unstack(tf.round((self.mosaic_size - self.input_size)/2))
                ix2, iy2 = ix1+self.input_size[0], iy1+self.input_size[1]
                self.crop_xyxy = tf.stack([ix1, iy1, ix2, iy2])
                method = self.mosaic
            else:
                random_pad = True if split=='train' else False
                data = data.map(lambda image, labels: resize_padding(image, labels, self.input_size, random_pad, seed=seed), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                method = self.batch
        else:
            random_pad = True if split=='train' else False
            data = data.map(lambda image, labels: resize_padding(image, labels, self.input_size, random_pad, seed=seed), num_parallel_calls=tf.data.experimental.AUTOTUNE) if resize else data
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

    def mosaic(self, data, batch_size, size=4, seed=42):
        batch_images = []
        batch_labels = []
        batch_mosaic_images = tf.zeros([0]+tf.unstack(self.input_size)+[3])
        batch_mosaic_labels = tf.zeros([0, 6], tf.float32)
        s = tf.cast(tf.sqrt(tf.cast(size, tf.float32)), tf.int32)
        
        for image, labels in data:
            batch_images += [image]
            batch_labels += [labels]

            if len(batch_images) == batch_size:
                for idx in range(batch_size):
                    mosaic_image = tf.Variable(tf.zeros(tf.unstack(self.mosaic_size)+[3]), trainable=False)
                    mosaic_labels = tf.zeros([0, 5], tf.float32)
                    xc, yc = tf.unstack(tf.cast(tf.random.uniform([2], 
                                                                  minval=self.mosaic_size//(2**size)*(2**(size-1)-1), 
                                                                  maxval=self.mosaic_size//(2**size)*(2**(size-1)+1), 
                                                                  seed=seed), dtype=tf.int32))
                    xcf, ycf = tf.cast(xc, tf.float32), tf.cast(yc, tf.float32)
                    indices = tf.random.uniform([size], minval=0, maxval=batch_size, dtype=tf.int32, seed=seed)
                    # indices = tf.range(idx*size, (idx+1)*size)

                    for i, index in enumerate(indices):
                        image = batch_images[index]
                        labels = batch_labels[index]
                        width, height = self.mosaic_size[0] - xcf if i%2 else xcf, self.mosaic_size[1] - ycf if i//2 else ycf
                        image, labels = resize(image, labels, tf.stack([width, height]))

                        h, w = tf.unstack(image.shape[:2])
                        x1, y1 = xc if i%s else xc - w, yc if i//s else yc - h
                        x2, y2 = x1+w, y1+h
 
                        mosaic_image[y1:y2, x1:x2].assign(image)
                        mosaic_labels = tf.concat([mosaic_labels, labels + [x1, y1, x1, y1, 0]], 0)

                    crop_image, crop_labels = crop(mosaic_image.value(), mosaic_labels, self.crop_xyxy)
                    batch_mosaic_images = tf.concat([batch_mosaic_images, crop_image[None]], 0)
                    batch_mosaic_labels = tf.concat([batch_mosaic_labels, tf.concat([tf.zeros(crop_labels.shape[:-1], dtype=tf.float32)[..., None]+idx, crop_labels], -1)], 0)
                    
                yield batch_mosaic_images, tf.concat(batch_mosaic_labels, 0)
                batch_images = []
                batch_labels = []
                batch_mosaic_images = tf.zeros([0]+tf.unstack(self.input_size)+[3])
                batch_mosaic_labels = tf.zeros([0, 6], tf.float32)


    def mosaic2(self, data, batch_size, size=4, seed=42):
        batch_images = []
        batch_labels = []
        batch_mosaic_images = []
        batch_mosaic_labels = []
        s = tf.cast(tf.sqrt(tf.cast(size, tf.float32)), tf.int32)

        
        for image, labels in data:
            batch_images += [image]
            batch_labels += [labels]

            if len(batch_images) == batch_size:
                for idx in range(batch_size):
                    mosaic_image = tf.Variable(tf.zeros(tf.unstack(self.mosaic_size)+[3]), trainable=False)
                    mosaic_labels = []
                    xc, yc = tf.unstack(tf.cast(tf.random.uniform([2], 
                                                                  minval=self.mosaic_size//(2**size)*(2**(size-1)-1), 
                                                                  maxval=self.mosaic_size//(2**size)*(2**(size-1)+1), 
                                                                  seed=seed), dtype=tf.int32))
                    xcf, ycf = tf.cast(xc, tf.float32), tf.cast(yc, tf.float32)
                    indices = tf.random.uniform([size], minval=0, maxval=batch_size, dtype=tf.int32, seed=seed)
                    # indices = tf.range(idx*size, (idx+1)*size)

                    for i, index in enumerate(indices):
                        image = batch_images[index]
                        labels = batch_labels[index]
                        width, height = self.mosaic_size[0] - xcf if i%2 else xcf, self.mosaic_size[1] - ycf if i//2 else ycf
                        image, labels = resize(image, labels, tf.stack([width, height]))

                        h, w = tf.unstack(image.shape[:2])
                        x1, y1 = xc if i%s else xc - w, yc if i//s else yc - h
                        x2, y2 = x1+w, y1+h
 
                        mosaic_image[y1:y2, x1:x2].assign(image)
                        mosaic_labels += [labels + [x1, y1, x1, y1, 0]]

                    mosaic_labels = tf.concat(mosaic_labels, 0)
                    crop_image, crop_labels = crop(mosaic_image.value(), mosaic_labels, self.crop_xyxy)
                    crop_labels = tf.concat([tf.fill([crop_labels.shape[0], 1], float(idx)), crop_labels], -1)
                    batch_mosaic_images += [crop_image]
                    batch_mosaic_labels += [crop_labels]
                batch_mosaic_images = tf.stack(batch_mosaic_images, 0)
                batch_mosaic_labels = tf.concat(batch_mosaic_labels, 0)
                yield batch_mosaic_images, tf.concat(batch_mosaic_labels, 0)
                batch_images = []
                batch_labels = []
                batch_mosaic_images = []
                batch_mosaic_labels = []


    def mosaic3(self, data, batch_size, size=4, seed=42):
        batch_images = []
        batch_labels = []
        batch_mosaic_images = []
        batch_mosaic_labels = []
        s = tf.cast(tf.sqrt(tf.cast(size, tf.float32)), tf.int32)

        
        for image, labels in data:
            batch_images += [image]
            batch_labels += [labels]

            if len(batch_images) == batch_size:
                for idx in range(batch_size):
                    mosaic_image = []
                    mosaic_labels = []
                    xc, yc = tf.unstack(tf.cast(tf.random.uniform([2], 
                                                                  minval=self.mosaic_size//(2**size)*(2**(size-1)-1), 
                                                                  maxval=self.mosaic_size//(2**size)*(2**(size-1)+1), 
                                                                  seed=seed), dtype=tf.int32))
                    xcf, ycf = tf.cast(xc, tf.float32), tf.cast(yc, tf.float32)
                    indices = tf.random.uniform([size], minval=0, maxval=batch_size, dtype=tf.int32, seed=seed)
                    # indices = tf.range(idx*size, (idx+1)*size)

                    for i, index in enumerate(indices):
                        image = batch_images[index]
                        labels = batch_labels[index]
                        width, height = self.mosaic_size[0] - xcf if i%2 else xcf, self.mosaic_size[1] - ycf if i//2 else ycf
                        image, labels = resize_padding(image, labels, tf.stack([width, height]))

                        h, w = tf.unstack(image.shape[:2])
                        x1, y1 = xc if i%s else xc - w, yc if i//s else yc - h
                        x2, y2 = x1+w, y1+h

                        mosaic_image += [image]
                        mosaic_labels += [labels + [x1, y1, x1, y1, 0]]

                    mosaic_image = tf.concat([tf.concat(mosaic_image[0:2], 1), tf.concat(mosaic_image[2:4], 1)], 0)
                    mosaic_labels = tf.concat(mosaic_labels, 0)
                    crop_image, crop_labels = crop(mosaic_image, mosaic_labels, self.crop_xyxy)
                    crop_labels = tf.concat([tf.fill([crop_labels.shape[0], 1], float(idx)), crop_labels], -1)
                    batch_mosaic_images += [crop_image]
                    batch_mosaic_labels += [crop_labels]
                batch_mosaic_images = tf.stack(batch_mosaic_images, 0)
                batch_mosaic_labels = tf.concat(batch_mosaic_labels, 0)
                yield batch_mosaic_images, tf.concat(batch_mosaic_labels, 0)
                batch_images = []
                batch_labels = []
                batch_mosaic_images = []
                batch_mosaic_labels = []