import tensorflow as tf
from tensorflow.keras import Model
from models.common import *
from losses.common import get_loss
from models.decoder import Decoder
from utils.io_utils import read_model_info, write_model_info

class YOLO(Model):
    def __init__(self, cfg):
        super().__init__()
        cfg['model']['kernel_init'] = get_kernel_initializer(cfg)    
            
        self.backbone = get_backbone(cfg)
        self.neck = get_neck(cfg)
        self.head = get_head(cfg)
        self.decoder = Decoder(cfg)
        self.loss = get_loss(cfg)

        print(f"Model: {cfg['model']['name']}")
        print(f"Backbone: {cfg['model']['backbone']['name']}")
        print(f"Neck: {cfg['model']['neck']['name']}")
        print(f"Head: {cfg['model']['head']['name']}")
        print(f"Input size: {cfg['model']['input_size'][0]}x{cfg['model']['input_size'][1]}")
        print(f"Loss Function: {cfg['model']['loss']} loss")

    @tf.function
    def call(self, x, training=False):
        backbone = self.backbone(x, training)
        neck = self.neck(backbone, training)
        head = self.head(neck, training)
        preds = self.decoder(head)

        return preds
    
def load_model(cfg):
    model = YOLO(cfg)
    if cfg['model']['load']:
        try:
            model.load_weights(cfg['model']['checkpoint'])
            saved = read_model_info(cfg['model']['checkpoint'])
            print(f"succeed to load model| epoch:{saved['epoch']} mAP50:{saved['mAP50']} mAP:{saved['mAP']} total_loss:{saved['total_loss']}")
            return model, saved['epoch'], saved['mAP']
        except:
            print(f'{cfg["model"]["checkpoint"]} is not exist.')
    print('make new model')
    return model, 1, -1

def save_model(model, epoch, mAP50, mAP, loss, checkpoint):
    model.save_weights(checkpoint)
    write_model_info(checkpoint, epoch, mAP50, mAP, loss)
    if 'best' in checkpoint:
        print(f'\033[32mbest_model is saved with {mAP:.4f} mAP in {epoch} epoch\033[0m')