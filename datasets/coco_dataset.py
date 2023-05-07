import numpy as np
import os, json
from config import *
from utils import bbox_utils
from datasets.common import Base_Dataset

class Dataset(Base_Dataset):
    def __init__(self, split, dtype=DTYPE, anchors=ANCHORS, labels=LABELS, image_size=IMAGE_SIZE,
                 create_anchors=CREATE_ANCHORS):
        super().__init__(split, dtype, anchors, labels, image_size, create_anchors)

    def load(self, use_tfrecord=True):
        return super().load(use_tfrecord)
    
    def read_files(self):
        print('Reading local_files...  ', end='', flush=True)
        
        anno_dir, image_dir = self.load_directory(self.split)
        
        anno_files = os.listdir(anno_dir)
        for anno_file in anno_files:
            if self.split == 'test' and 'dev' not in anno_file:
                break
            elif self.split in anno_file and 'instances' in anno_file:
                break        
        
        parsed_data, normalized_wh = self.parse_annotation(anno_dir + anno_file)
        
        for value in parsed_data.values():
            file_name = value['file_name']
            
            if self.split == 'test':
                labels = []
            elif len(value['labels'])==0:
                labels = []
            else:
                labels = value['labels']
            
            self.data += [[image_dir + file_name, labels]]
        np.random.shuffle(self.data)
        print('Done!')
               
        return normalized_wh

    def load_directory(self, split):
        extracted_dir = './data/' + self.dtype + '/downloads/extracted/'
        for dir in os.listdir(extracted_dir):
            if 'anno' in dir:
                if split=='train' and 'train' in dir:
                        anno_dir = dir
                elif split=='val' and 'va' in dir:
                        anno_dir = dir
                elif split=='test' and 'test' in dir:
                        anno_dir = dir
            else:
                if split in dir:
                    image_dir = dir
            
        anno_dir = extracted_dir + anno_dir + '/annotations/'
        image_dir = extracted_dir + image_dir + '/' + split + '2017/'
    
        return anno_dir, image_dir

    def parse_annotation(self, anno_path):
        data = {}
        categories = {}
        normalized_wh = np.zeros((0,2))

        with open(anno_path) as f:
            json_data = json.load(f)
            
        for category in json_data['categories']:
            categories[category['id']] = category['name']
            
        for image in json_data['images']:
            data[image['id']] = {'file_name': image['file_name'],
                                 'labels': []}
            if self.create_anchors:
                data[image['id']]['length'] = np.max([image['height'], image['width']])

        for anno in json_data['annotations']:
            bbox = bbox_utils.coco_to_xyxy(np.array(anno['bbox']))
            data[anno['image_id']]['labels'] += [[*bbox, float(LABELS.index(categories[anno['category_id']]))]]
                
        if self.create_anchors:
            for value in data.values():
                length = value['length']
                labels_wh = np.array(value['labels'])[..., 2:4]/length
                if(len(labels_wh)!=0):
                    normalized_wh = np.concatenate([normalized_wh, labels_wh], 0)

        return data, normalized_wh