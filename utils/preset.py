import os
import tensorflow as tf
from config import *

def os_preset():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def tf_preset():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0],True)
        except RuntimeError as error:
            print(error)

def checkpoint_preset():
    if not os.path.exists(TRAIN_CHECKPOINTS_DIR):
        os.makedirs(TRAIN_CHECKPOINTS_DIR)
    if not os.path.exists(LOSS_CHECKPOINTS_DIR):
        os.makedirs(LOSS_CHECKPOINTS_DIR)
    if not os.path.exists(MAP_CHECKPOINTS_DIR):
        os.makedirs(MAP_CHECKPOINTS_DIR)
        
def log_preset():
    if not os.path.exists(LOGDIR):
        # shutil.rmtree(TRAIN_LOGDIR)
        os.makedirs(LOGDIR)

def output_preset():
    if not os.path.exists(OUTPUT_DIR + 'image/'):
        os.makedirs(OUTPUT_DIR + 'image/')
    if not os.path.exists(OUTPUT_DIR + 'video/'):
        os.makedirs(OUTPUT_DIR + 'video/')

def preset():
    os_preset()
    tf_preset()
    checkpoint_preset()
    log_preset()
    output_preset()