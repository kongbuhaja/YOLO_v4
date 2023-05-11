from config import *
from utils import anchor_utils, io_utils
import numpy as np
import sys, os, cv2, gdown, zipfile, shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

class Base_Dataset():
    def __init__(self, split, dtype, anchors, labels, image_size, create_anchors):
        self.split = split
        self.dtype = dtype
        self.anchors = np.array(anchors)
        self.labels = labels
        self.image_size = image_size
        self.create_anchors = create_anchors
        self.data = []
        self.length = 0
        print(f'Dataset: {self.dtype} {self.split}')
    
    def load(self, use_tfrecord):
        assert self.split in ['train', 'val', 'test'], 'Check your dataset type and split.'
        self.download_dataset()
        if self.create_anchors:
            normalized_wh = self.read_files()
            self.make_new_anchors(normalized_wh)
            print('Anchors are changed. You need to restart file!')
            print('Please restart train.py')
            sys.exit()
        if use_tfrecord:
            filepath = f'./data/{self.dtype}/{self.split}.tfrecord'
            infopath = f'./data/{self.dtype}/{self.split}.txt'
            if os.path.exists(filepath):
                print(f'{filepath} is exist')
            else:
                normalized_wh = self.read_files()
                self.make_tfrecord(filepath, infopath)                
            data = self.read_tfrecord(filepath)
        else:
            normalized_wh = self.read_files()
            data = tf.data.Dataset.from_generator(self.generator, 
                                                  output_types=(tf.uint8, tf.float32, tf.float32, tf.float32),
                                                  output_shapes=((None, None, 3), (None, 5), (), ()))
        self.length = self.len(use_tfrecord)
        return data

    def read_tfrecord(self, filepath):
        dataset =  tf.data.TFRecordDataset(filepath, num_parallel_reads=-1) \
                        .map(parse_tfrecord_fn)
        return dataset
    
    def generator(self):
        for image_file, labels in self.data:
            image = self.read_image(image_file)
            labels = tf.constant([[0, 0, 0, 0, 0]], tf.float32) if len(labels)==0 else labels
            height, width = image.shape[:2]
            yield image, labels, float(width), float(height)
    
    def make_tfrecord(self, filepath, infopath):    
        with open(infopath, 'w') as f:
            f.write(str(len(self.data)))

        print(f'Start make {filepath}......      ', end='', flush=True)
        with tf.io.TFRecordWriter(filepath) as writer:
            for image_file, labels in tqdm.tqdm(self.data):
                image = self.read_image(image_file)
                height, width = image.shape[:2]
                writer.write(_data_features(image, labels, float(width), float(height)))
        print('Done!')

    def download_dataset(self):
        if self.dtype in ['voc', 'coco']:
            out_dir = './data/' + self.dtype
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                if self.dtype == 'voc':
                    year = '/2012'
                elif self.dtype == 'coco':
                    year = '/2017'
                tfds.load(self.dtype + year, data_dir=out_dir)
                if os.path.exists(out_dir + '/' + self.dtype):
                    shutil.rmtree(out_dir + '/' + self.dtype)
                for file in os.listdir(out_dir + '/downloads/'):
                    if file.endswith('.tar') or file.endswith('zip') or file.endswith('.INFO'):
                        os.remove(out_dir + '/downloads/' + file)
        
        elif self.dtype == 'custom':
            path = 'https://drive.google.com/uc?id='
            file = '15G2fgzBd8uXPr8yLgcJFhOYfpZlzd294'
            out_dir = './data/' + self.dtype + '/'
            
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            
                gdown.download(path + file, out_dir + self.dtype + '.zip')
            
                with zipfile.ZipFile(out_dir + self.dtype + '.zip', 'r') as zip:
                    zip.extractall(out_dir)
                
                os.remove(out_dir + self.dtype + '.zip')

    def make_new_anchors(self, normalized_wh):
        print(f'Start calulate anchors......      ', end='', flush=True)
        anchors = np.array(ANCHORS)
        new_anchors = anchor_utils.generate_anchors(normalized_wh, np.prod(anchors.shape[:-1]), IMAGE_SIZE, IMAGE_SIZE)
        io_utils.edit_config(str(ANCHORS), str(new_anchors.reshape(anchors.shape).tolist()))
        print('Done!')
        
    def read_image(self, image_file):
        image = cv2.imread(image_file)
        return image[..., ::-1]
    
    def len(self, use_tfrecord):
        if use_tfrecord:
            infopath = f'./data/{self.dtype}/{self.split}.txt'
            with open(infopath, 'r') as f:
                lines = f.readlines()
            return int(lines[0]) 
        return len(self.data)

def parse_tfrecord_fn(example):
    feature_description={
        'image': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.float32),
        'width': tf.io.FixedLenFeature([], tf.float32),
        'height': tf.io.FixedLenFeature([], tf.float32)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example['image'] = tf.io.decode_jpeg(example['image'], channels=3)
    example['labels'] = tf.reshape(tf.sparse.to_dense(example['labels']), (-1, 5))

    return example['image'], example['labels'], example['width'], example['height']
        
def _image_feature(value):
    return _bytes_feature(tf.io.encode_jpeg(value).numpy())
def _array_feature(value):
    if 'float' in value.dtype.name:
        return _float_feature(np.reshape(value, (-1)))
    elif 'int' in value.dtype.name:
        return _int64_feature(np.reshape(value, (-1)))
    raise Exception(f'Wrong array dtype: {value.dtype}')
def _string_feature(value):
    return _bytes_feature(value.encode('utf-8'))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    if type(value) == float:
        value=[value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _int64_feature(value):
    if type(value) == int:
        value=[value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _data_features(image, labels, width, height):
    image_feature = _image_feature(image)
    labels_feature = _array_feature(np.array(labels))
    width_feature = _float_feature(width)
    height_feature = _float_feature(height)
    
    objects_features = {
        'image': image_feature,
        'labels': labels_feature,
        'width': width_feature,
        'height': height_feature
    }      
    example=tf.train.Example(features=tf.train.Features(feature=objects_features))
    return example.SerializeToString()