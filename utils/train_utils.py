from config import *
from utils import io_utils
import math

class warmup_lr_scheduler():
    def __init__(self, warmup_max_step, init_lr):
        self.warmup_max_step = warmup_max_step
        self.init_lr = init_lr

    def __call__(self, warmup_step):
        lr = self.init_lr / self.warmup_max_step * (warmup_step + 1)
        return lr

class step_lr_scheduler():
    def __init__(self, init_lr, step_per_epoch):
        self.init_lr = init_lr
        self.step_per_epoch = step_per_epoch
    
    def __call__(self, step):
        epoch = (step + 1) / self.step_per_epoch
        if epoch < 200:
            lr = self.init_lr 
        elif epoch < 300:
            lr = self.init_lr * 0.5
        elif epoch < 400:
            lr = self.init_lr * 0.1
        elif epoch < 500:
            lr = self.init_lr * 0.05
        else:
            lr = self.init_lr * 0.01
        return lr
    
class poly_lr_scheduler():
    def __init__(self, init_lr, max_step, power):
        self.init_lr = init_lr
        self.max_step = max_step
        self.power = power
    
    def __call__(self, step):
        lr = self.init_lr * (1 - (step/(self.max_step)))**self.power
        return lr
    
class cosine_annealing_lr_scheduler():
    def __init__(self, init_lr, T_step, T_mult, min_lr):
        self.init_lr = init_lr
        self.T_step = T_step
        self.T_max = T_step
        self.T_mult = T_mult
        self.csum = 0
        self.min_lr = min_lr

    def __call__(self, step):
        T_cur = step - self.csum
        while(T_cur >= self.T_max):
            T_cur -= self.T_max
            self.csum += self.T_max
            self.T_max *= self.T_mult
            
        lr = self.min_lr + 0.5 * (self.init_lr - self.min_lr) * (1 + math.cos(T_cur / self.T_max * math.pi))
        return lr
    
class LR_scheduler():
    def __init__(self, step_per_epoch, max_step, warmup_max_step, lr_type, init_lr):
        self.step_per_epoch = step_per_epoch
        self.max_step = max_step
        self.warmup_max_step = warmup_max_step
        self.lr_type = lr_type
        self.init_lr = init_lr
        self.warmup_lr_scheduler = warmup_lr_scheduler(self.warmup_max_step, self.init_lr)
        if self.lr_type == 'cosine_annealing':
            self.lr_scheduler = cosine_annealing_lr_scheduler(self.init_lr, T_STEP, T_MULT, MIN_LR)
        elif self.lr_type == 'poly':
            self.lr_scheduler = poly_lr_scheduler(self.init_lr, self.max_step, POWER)
        elif self.lr_type == 'step':
            self.lr_scheduler = step_lr_scheduler(self.init_lr, self.step_per_epoch)

    def __call__(self, step, warmup_step):
        lr = self.lr_scheduler(step - self.warmup_max_step)
        if warmup_step < self.warmup_max_step:
            lr = self.warmup_lr_scheduler(warmup_step)
        return lr

def load_model(model, checkpoints):
    try:
        model.load_weights(checkpoints)
        saved = io_utils.read_model_info()
        return model, saved['epoch'], saved['mAP'], saved['total_loss']
    except:
        print('checkpoints is not exist. \nmake new model')
        return model, 1, -1., INF

def get_model(load_checkpoints=LOAD_CHECKPOINTS):
    if MODEL_TYPE == 'YOLOv4':
        from models.yolov4 import YOLO
    elif MODEL_TYPE == 'YOLOv3':
        from models.yolov3 import YOLO
    elif MODEL_TYPE == 'YOLOv3_tiny':
        from models.yolov3_tiny import YOLO
    print(f'Model: {MODEL_TYPE}')
    print(f'Loss Metric: {LOSS_METRIC}')
    
    if load_checkpoints:
        return load_model(YOLO(), CHECKPOINTS)
    print('make new model')
    return YOLO(), 1, -1., INF

def save_model(model, epoch, mAP, loss, dir_path):
    checkpoints = dir_path + MODEL_TYPE

    model.save_weights(checkpoints)
    io_utils.write_model_info(checkpoints, epoch, mAP, loss)
    if 'train' not in dir_path:
        print(f'{dir_path} epoch:{epoch}, mAP:{mAP:.7f} best_model is saved')
