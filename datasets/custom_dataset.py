import numpy as np
import os, gdown, zipfile
import xml.etree.ElementTree as ET
from datasets.common import Base_Dataset

class Dataset(Base_Dataset):
    def __init__(self, split, anchors, labels):
        super().__init__(split, 'custom', anchors, labels)

    def download_dataset(self):
        path = 'https://drive.google.com/uc?id='
        file = '15G2fgzBd8uXPr8yLgcJFhOYfpZlzd294'
        out_dir = './data/' + self.dtype + '/'
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
            try:
                gdown.download(path + file, out_dir + self.dtype + '.zip')
                with zipfile.ZipFile(out_dir + self.dtype + '.zip', 'r') as zip:
                    zip.extractall(out_dir)
                
                os.remove(out_dir + self.dtype + '.zip')
            except:
                self.download_from_server()

    def read_files(self):
        print('Reading local_files...  ', end='', flush=True)
        
        anno_dir, image_dir = self.load_directory(self.split)
        
        anno_files = os.listdir(anno_dir)

        for anno_file in anno_files:
            image_file, labels = self.parse_annotation(anno_dir + anno_file)
            self.data += [[image_dir + image_file, labels]]
            
        np.random.shuffle(self.data)
        print('Done!')
                
    def load_directory(self, split):
        extracted_dir = './data/custom/'
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
