import os
from config import *
from datasets.common import Base_Dataset

class Dataset(Base_Dataset):
    def __init__(self, split, dtype=DTYPE, anchors=ANCHORS, labels=LABELS, image_size=IMAGE_SIZE,
                 create_anchors=CREATE_ANCHORS):
        super().__init__(split, dtype, anchors, labels, image_size, create_anchors)

    def load(self, use_tfrecord=False):
        return super().load(use_tfrecord=False)

    def read_files(self):
        print('Reading local_files...  ', end='', flush=True)
        
        image_dir = self.load_directory(self.split)
        for image_file in os.listdir(image_dir):
            self.data += [[image_dir + image_file, []]]
            
        print('Done!')
        
        return []
    
    def load_directory(self, split):
        image_dir = './data/' + self.dtype + '/'
        return image_dir