import tensorflow as tf
import numpy as np
import argparse, yaml
from datetime import datetime
from utils.env_utils import env_set

def args_parse():
    parser = argparse.ArgumentParser(description='YOLOv4')
    parser.add_argument('--gpus', dest='gpus', type=str, default='', help='which device do you want to use')
    parser.add_argument('--model', dest='model', type=str, default='', 
                        choices=['YOLOv3', 'YOLOv3_tiny', 'YOLOv4', 'YOLOv4_tiny', 'YOLOv4_csp', 'YOLOv4_P5', 'YOLOv4_P6', 'YOLOv4_P7'], 
                        help='model to train')
    parser.add_argument('--data', dest='data', type=str, default='', help='dataset for training')

    args = parser.parse_args()

    return args

def read_cfg():
    args=args_parse()
    
    with open('yaml/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    cfg['model']['name'] = args.model if args.model else cfg['model']['name']
    cfg['data'] = args.data if args.data else cfg['data']
    cfg['gpus'] = args.gpus if args.gpus else str(cfg['gpus'])
    
    with open(f"yaml/data/{cfg['data']}.yaml") as f:
        cfg.update(yaml.load(f, Loader=yaml.FullLoader))

    if cfg['model']['name'] in ['YOLOv3_tiny', 'YOLOv4_tiny']:
        cfg['model']['anchors'] = cfg['data']['anchors']['2x3']
    elif cfg['model']['name'] in ['YOLOv3', 'YOLOv4', 'YOLOv4_csp']:
        cfg['model']['anchors'] = cfg['data']['anchors']['3x3']
    elif cfg['model']['name'] in ['YOLOv4_P5']:
        cfg['model']['anchors'] = cfg['data']['anchors']['3x4']
    elif cfg['model']['name'] in ['YOLOv4_P6']:
        cfg['model']['anchors'] = cfg['data']['anchors']['4x4']
    elif cfg['model']['name'] in ['YOLOv4_P7']:
        cfg['model']['anchors'] = cfg['data']['anchors']['5x4']
    cfg['model']['anchors'] = np.array(cfg['model']['anchors'])
    cfg['model']['dir'] = f"{cfg['model']['dir']}/{cfg['data']['name']}/{cfg['model']['name']}"
    cfg['model']['train_checkpoint'] = f"{cfg['model']['dir']}/train_loss/{cfg['model']['name']}"
    cfg['model']['loss_checkpoint'] = f"{cfg['model']['dir']}/val_loss/{cfg['model']['name']}"
    cfg['model']['map_checkpoint'] = f"{cfg['model']['dir']}/map_loss/{cfg['model']['name']}"
    cfg['model']['checkpoint'] = cfg['model'][cfg['model']['checkpoint']]

    cfg['eval']['dir'] = f"{cfg['eval']['dir']}/{cfg['data']['name']}/{cfg['model']['name']}"
    cfg['eval']['image_dir'] = f"{cfg['eval']['dir']}/image"
    cfg['eval']['video_dir'] = f"{cfg['eval']['dir']}/video"
    if 'soft' in cfg['eval']['nms']:
        cfg['eval']['score_th'] = 0.01
    else:
        cfg['eval']['score_th'] = 0.4    
        
    cfg['log']['dir'] = f"{cfg['log']['dir']}/{cfg['model']['name']}_{cfg['data']['name']}_{datetime.now().strftime('%Y-%m-%d|%H:%M:%S')}"

    env_set(cfg)

    return cfg

def write_model_info(checkpoints, epoch, mAP50, mAP, loss):
    with open(checkpoints + '.info', 'w') as f:
        text = f'epoch:{epoch}\n' +\
               f'mAP50:{mAP50}\n' +\
               f'mAP:{mAP}\n' +\
               f'total_loss:{loss[0]}\n' +\
               f'reg_loss:{loss[1]}\n' +\
               f'conf_loss:{loss[2]}\n' +\
               f'prob_loss:{loss[3]}\n'
        f.write(text)
        
def read_model_info(checkpoints):
    saved_parameter = {}
    with open(checkpoints + '.info', 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line[:-1].split(':')
            if key == 'epoch':
                saved_parameter[key] = int(value)
            else:
                saved_parameter[key] = float(value)
    return saved_parameter

class Logger():
    def __init__(self, cfg):
        self.logdir = cfg['log']['dir']
        self.train_logger = tf.summary.create_file_writer(self.logdir)
        self.eval_logger = tf.summary.create_file_writer(self.logdir)
    
    def write_train_summary(self, step, lr, loss):
        with self.train_logger.as_default():
            tf.summary.scalar('lr', lr, step=step)
            tf.summary.scalar('Train/total_loss', loss[0], step=step)
            tf.summary.scalar('Train/reg_loss', loss[1], step=step)
            tf.summary.scalar('Train/obj_loss', loss[2], step=step)
            tf.summary.scalar('Train/cls_loss', loss[3], step=step)

        self.train_logger.flush()

    def write_eval_summary(self, step, mAP50, mAP, loss):
        with self.eval_logger.as_default():
            tf.summary.scalar("Eval/mAP50", mAP50, step=step)
            tf.summary.scalar("Eval/mAP", mAP, step=step)
            tf.summary.scalar("Eval/total_loss", loss[0], step=step)
            tf.summary.scalar("Eval/reg_loss", loss[1], step=step)
            tf.summary.scalar("Eval/obj_loss", loss[2], step=step)
            tf.summary.scalar("Eval/cls_loss", loss[3], step=step)

        self.eval_logger.flush()

def write_cfg(cfg):
    text = cfg_to_str(cfg)
    with open(f"yaml/data/{cfg['data']['name']}.yaml", 'w') as f:
        f.write(text)
    
def cfg_to_str(cfg, tab='  ', indent=0):
    text = ''
    for key, value in sorted(cfg.items()):
        if isinstance(value, dict) or isinstance(value, list):
            text += f'{tab*indent}{key}:\n'
            if isinstance(value, dict):
                text += cfg_to_str(value, indent=indent+1)
            elif isinstance(value, list):
                text += array_to_str(np.array(value), tab=tab*(indent+1))
        else:
            text += f'{tab*indent}{key}: {value}\n'
        text += '\n' if indent==1 else ''
    return text

def array_to_str(data, tab='', open='[', close=']', indent=1, first=True, last=False):
    text = ''
    if len(data.shape) == 1:
        text += f'{tab}{open}'
        for i, value in enumerate(data, 1):
            text += f'{value}, '
            if i%5 == 0:
                text += f'\n{tab} ' if i!=len(data) else ''

        text = f'{text[:-2]}{close}\n'

    elif len(data.shape) == 2:
        text += f'{tab}{open*indent}' if first else f'{tab}{" "*(indent-1)}{open*(indent-1)}'

        indent -= 1
        for i, (w, h) in enumerate(data):
            text += f'{open}{w:.6f}, {h:.6f}{close}, '
        
        text = text[:-2]
        text += f'{close*(indent+1)}\n' if last else f'{close*indent},\n'

    else:
        for i, sub_data in enumerate(data):
            text += array_to_str(sub_data, tab=tab, indent=indent+1, first=not(i), last=i==(len(data)-1))
    return text
