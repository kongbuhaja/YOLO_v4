import numpy as np
import os, json, shutil
from utils import bbox_utils
from datasets.common import Base_Dataset
import tensorflow_datasets as tfds


class Dataset(Base_Dataset):
    def __init__(self, split, anchors, labels):
        super().__init__(split, 'coco', anchors, labels)

    def download_dataset(self):
        out_dir = './data/coco'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            try:
                tfds.load('coco2017', data_dir=out_dir)
                if os.path.exists(f'{out_dir}/coco'):
                    shutil.rmtree(f'{out_dir}/coco')
                for file in os.listdir(f'{out_dir}/downloads/'):
                    if file.endswith('.tar') or file.endswith('zip') or file.endswith('.INFO'):
                        os.remove(f'{out_dir}/downloads/{file}')
            except:
                self.download_from_server()
    
    def read_files(self):
        print('Reading local_files...  ', end='', flush=True)
        
        anno_dir, image_dir = self.load_directory(self.split)
        
        anno_files = os.listdir(anno_dir)
        for anno_file in anno_files:
            if self.split == 'test' and 'dev' not in anno_file:
                break
            elif self.split in anno_file and 'instances' in anno_file:
                break        

        parsed_data = self.parse_annotation(anno_dir + anno_file)
        
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

    def load_directory(self, split):
        extracted_dir = './data/coco/downloads/extracted/'
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
    
        with open(anno_path) as f:
            json_data = json.load(f)
            
        for category in json_data['categories']:
            categories[category['id']] = category['name']
            
        for image in json_data['images']:
            data[image['id']] = {'file_name': image['file_name'],
                                 'labels': []}        

        for anno in json_data['annotations']:
            bbox = bbox_utils.coco_to_xyxy(np.array(anno['bbox']))
            data[anno['image_id']]['labels'] += [[*bbox ,float(self.labels.index(categories[anno['category_id']]))]]

        return data