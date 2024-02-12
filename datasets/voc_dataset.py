import numpy as np
import os, shutil
import xml.etree.ElementTree as ET
from datasets.common import Base_Dataset
import tensorflow_datasets as tfds

class Dataset(Base_Dataset):
    def __init__(self, split, anchors, labels):
        super().__init__(split, 'voc', anchors, labels)

    def download_dataset(self):
        out_dir = './data/voc'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            try:
                tfds.load('voc2012', data_dir=out_dir)
                if os.path.exists(f'{out_dir}/voc'):
                    shutil.rmtree(f'{out_dir}/voc')
                for file in os.listdir(f'{out_dir}/downloads/'):
                    if file.endswith('.tar') or file.endswith('zip') or file.endswith('.INFO'):
                        os.remove(f'{out_dir}/downloads/{file}')
            except:
                self.download_from_server()

    def read_files(self):
        print('Reading local_files...  ', end='', flush=True)
        
        anno_dir, image_dir = self.load_directory(self.split)
        
        anno_files = os.listdir(anno_dir)
        if self.split == 'train':
            anno_files = anno_files[:-5000]
        elif self.split == 'val':
            anno_files = anno_files[-5000:]
        
        if self.split != 'test':
            for anno_file in anno_files:
                image_file, labels = self.parse_annotation(anno_dir + anno_file)
                if self.split != 'test':
                    self.data += [[image_dir + image_file, labels]]
        else:
            image_files = os.listdir(image_dir)
            for image_file in image_files:
                self.data += [[image_dir + image_file, []]]
                
        np.random.shuffle(self.data)
        print('Done!')
        
    def load_directory(self, split):
        extracted_dir = './data/voc/downloads/extracted/'
        for dir in os.listdir(extracted_dir):
            if split == 'test' and split in dir:
                break
            elif split != 'test' and 'tra' in dir:
                break
            
        anno_dir = extracted_dir + dir + '/VOCdevkit/VOC2012/Annotations/'
        image_dir = extracted_dir + dir + '/VOCdevkit/VOC2012/JPEGImages/'
        return anno_dir, image_dir

    def parse_annotation(self, anno_path):
        tree = ET.parse(anno_path)
        labels = []
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                filename = elem.text
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
                        labels.append([xmin, ymin, xmax, ymax, label])
            
        return filename, labels