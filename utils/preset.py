import os
import tensorflow as tf
from config import *

def os_preset(gpus):
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # if GPUS==1:
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUS)
    #     # os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # else:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''.join([str(i)+', ' for i in range(gpus)])
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def tf_preset():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
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

def preset(gpus=GPUS):
    os_preset(gpus)
    tf_preset()
    checkpoint_preset()
    log_preset()
    output_preset()
