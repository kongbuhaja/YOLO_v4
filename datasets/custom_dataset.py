import numpy as np
import os
import xml.etree.ElementTree as ET
from datasets.common import Base_Dataset

class Dataset(Base_Dataset):
    def __init__(self, split, dtype, anchors, labels, input_size, create_anchors):
        super().__init__(split, dtype, anchors, labels, input_size, create_anchors)

    def load(self, use_tfrecord=True):
        return super().load(use_tfrecord)

    def read_files(self):
        normalized_wh = np.zeros((0,2))
        print('Reading local_files...  ', end='', flush=True)
        
        anno_dir, image_dir = self.load_directory(self.split)
        
        anno_files = os.listdir(anno_dir)

        for anno_file in anno_files:
            image_file, labels, width, height = self.parse_annotation(anno_dir + anno_file)
            self.data += [[image_dir + image_file, labels, width, height]]
            
        np.random.shuffle(self.data)
        print('Done!')
        
        return normalized_wh
        
    def load_directory(self, split):
        extracted_dir = './data/' + self.dtype + '/'
        for dir in os.listdir(extracted_dir):
            if os.path.isdir(extracted_dir + dir):
                if split == 'train' and 'tra' in dir:
                    break
                elif split == 'val' and 'val' in dir:
                    break
        anno_dir = extracted_dir + dir + '/Annotations/'
        image_dir = extracted_dir + dir + '/JPEGImages/'
        return anno_dir, image_dir

    def parse_annotation(self, anno_path):
        tree = ET.parse(anno_path)
        labels = []
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
            elif 'width' in elem.tag:
                width = float(elem.text)
            elif 'height' in elem.tag:
                height = float(elem.text)
            elif 'object' in elem.tag:
                for attr in list(elem):
                    if 'name' in attr.tag:
                        label = float(self.labels.index(attr.text))
                    elif 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = float(dim.text)
                            elif 'ymin' in dim.tag:
                                ymin = float(dim.text)
                            elif 'xmax' in dim.tag:
                                xmax = float(dim.text)
                            elif 'ymax' in dim.tag:
                                ymax = float(dim.text)
                        labels.append([xmin, ymin, xmax, ymax, 1., label])
            
        return filename, labels, width, height
