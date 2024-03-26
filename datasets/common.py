import numpy as np
import os, cv2, gdown, zipfile, shutil, tqdm, urllib.request, tarfile
import tensorflow as tf
import tensorflow_datasets as tfds

class Base_Dataset():
    def __init__(self, split, dtype, labels):
        self.split = split
        self.dtype = dtype
        # self.anchors = np.array(anchors)
        self.labels = labels
        self.data = []
        self.length = 0
        print(f'Dataset: {self.dtype} {self.split}')
    
    def load(self):
        assert self.split in ['train', 'val'], 'Check your dataset type and split.'
        
        filepath = f'./data/{self.dtype}/{self.split}.tfrecord'
        infopath = f'./data/{self.dtype}/{self.split}.info'

        if os.path.exists(filepath):
            print(f'{filepath} is exist')
        else:
            self.download_dataset()
            self.read_files()
            self.make_tfrecord(filepath, infopath)      
            
        del self.data

        return self.read_tfrecord(filepath, infopath)

    def read_tfrecord(self, filepath, infopath):
        dataset =  tf.data.TFRecordDataset(filepath, num_parallel_reads=-1) \
                        .map(parse_tfrecord_fn)
        
        with open(infopath, 'r') as f:
            lines = f.readlines()

        return dataset, int(lines[0])
    
    def make_tfrecord(self, filepath, infopath):    
        with open(infopath, 'w') as f:
            f.write(str(len(self.data)))

        print(f'Start make {filepath}......      ', end='', flush=True)
        with tf.io.TFRecordWriter(filepath) as writer:
            for image_file, labels in tqdm.tqdm(self.data):
                image = self.read_image(image_file)
                writer.write(_data_features(image, labels))
                # writer.write(_data_features(image_file, labels))
        print('Done!')

    def download_from_server(self):
        print('Download dataset from server')
        download_from_server(self.dtype, path='data')
        extract(self.dtype, path='data')
        os.remove(f'data/{self.dtype}.tar.gz')
        
    def read_image(self, image_file):
        image = cv2.imread(image_file)
        return image[..., ::-1]
    
    # Overwrite in subclass
    def download_dataset(self):
        pass
    def read_files(self):
        pass
    def load_directory(self, split):
        pass
    def parse_annotation(self, anno_path):
        pass

def download_from_server(data_name, path, ip_address='166.104.144.76', port=8000):
    url = f'http://{ip_address}:{port}/Object_Detection/{data_name}.tar.gz'
    urllib.request.urlretrieve(url, f'{path}/{data_name}.tar.gz')

def extract(data_name, path):
    with tarfile.open(f'{path}/{data_name}.tar.gz', 'r:gz') as tar:
        tar.extractall(f'data/')

def parse_tfrecord_fn(example):
    feature_description={
        'image': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    # example['image'] = tf.io.decode_jpeg(example['image'], channels=3)
    example['labels'] = tf.reshape(tf.sparse.to_dense(example['labels']), [-1, 5])

    return example['image'], example['labels']
        
def _image_feature(value):
    return _bytes_feature(tf.io.encode_jpeg(value).numpy())
def _array_feature(value):
    if 'float' in value.dtype.name:
        return _float_feature(np.reshape(value, [-1]))
    elif 'int' in value.dtype.name:
        return _int64_feature(np.reshape(value, [-1]))
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
def _data_features(image, labels):
    image_feature = _image_feature(image)
    # image_feature = _string_feature(image)
    labels_feature = _array_feature(np.array(labels))
    
    objects_features = {
        'image': image_feature,
        'labels': labels_feature,
    }      
    example=tf.train.Example(features=tf.train.Features(feature=objects_features))
    return example.SerializeToString()