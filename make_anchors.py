import tqdm
import tensorflow as tf
from models.model import *
from utils.data_utils import DataLoader
from utils.io_utils import read_cfg, write_cfg
from utils.env_utils import env_set
from utils.bbox_utils import bbox_iou_wh

def main():
    cfg = read_cfg()
    cfg['batch_size'] = 1
    cfg['mosaic'] = 0
    cfg['model']['anchors'] = np.zeros([0,0])
    cfg['model']['input_size'] = np.zeros([2,])

    anchors = Anchors(cfg['seed'])
    temp = []
    
    dataloader = DataLoader(cfg)

    train_dataset = dataloader('train', cfg['batch_size'], aug=cfg['aug'])
    train_dataset_length = dataloader.length['train']

    anchor_tqdm = tqdm.tqdm(train_dataset, total=train_dataset_length, ncols=160, desc=f'Reading train dataset', ascii=' =', colour='red')
    for image, label in anchor_tqdm:
        temp += [label[..., 3:5]/tf.cast(tf.reduce_max(image.shape[1:3]), tf.float32)]

    cfg['data']['anchors'] = anchors.generate_anchors(tf.concat(temp, 0))
    write_cfg({'data': cfg['data']})

class Anchors:
    def __init__(self, seed):
        self.data = None
        self.seed = seed
        self.shape = ['1x5', '2x3', '3x3', '3x4', '4x4', '5x4']
        self.anchors = dict()
        
    def kmeans(self, boxes, k):
        boxes = tf.random.shuffle(boxes, seed=self.seed)
        num_boxes = boxes.shape[0]
        last_cluster = tf.zeros((num_boxes,), dtype=tf.int64)
        clusters = boxes[:k].numpy()

        while True:
            distances = 1 - bbox_iou_wh(boxes[:, None], clusters)

            nearest = tf.argmin(distances, axis=1)
            if tf.reduce_all(last_cluster == nearest):
                break
            for cluster in range(k):
                clusters[cluster] = tf.reduce_mean(boxes[nearest == cluster], axis=0).numpy()

            last_cluster = nearest
        return clusters
    
    def generate_anchors(self, boxes, times=5):
        for shape in self.shape:
            best_acc = 0.
            size = [int(s) for s in shape.split('x')]
            k = tf.reduce_prod(size)
            for i in range(times):
                anchors = self.kmeans(boxes, k)
                acc = tf.reduce_mean([tf.reduce_max(bbox_iou_wh(boxes[:, None], anchors), 1)])
                if best_acc < acc:
                    best_acc = acc.numpy()
                    self.anchors[shape] = tf.gather(anchors, tf.argsort(tf.reduce_prod(anchors, -1))).numpy().reshape([*size, 2]).tolist()
            print(f'{shape} anchors acc: {best_acc*100:.4}%')
        return self.anchors

if __name__ == '__main__':
    main()

