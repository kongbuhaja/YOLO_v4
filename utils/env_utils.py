import os, random
import tensorflow as tf
import numpy as np

def os_set(gpus):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def hw_set():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as error:
            print(error)

def random_seed_set(seed):
    os.environ['PYTHONASHSEED']=str(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def dir_check(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

def output_set(output_dir):
    if not os.path.exists(output_dir + 'image/'):
        os.makedirs(output_dir + 'image/')
    if not os.path.exists(output_dir + 'video/'):
        os.makedirs(output_dir + 'video/')

def env_set(cfg):
    random_seed_set(cfg['seed'])
    os_set(cfg['gpus'])
    hw_set()
    dir_check(cfg['eval']['image_dir'])
    dir_check(cfg['eval']['video_dir'])
