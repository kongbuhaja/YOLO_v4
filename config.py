# hardware config
GPUS = 1

# data config
DTYPE = 'custom'
IMAGE_SIZE = 416
MAX_BBOXES = 100
CREATE_ANCHORS = False
POSITIVE_IOU_THRESHOLD = 0.5

# train config
EPOCHS = 2000
BATCH_SIZE = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE * GPUS
LOSS_METRIC = 'YOLOv4Loss'
LR = 1e-3
LR_SCHEDULER = 'cosine_annealing'
IOU_THRESHOLD = 0.5
EPS = 1e-5
INF = 1e+30
EVAL_PER_EPOCHS = 1
WARMUP_EPOCHS = 5

if LR_SCHEDULER == 'poly':
    POWER = 0.9
elif LR_SCHEDULER == 'cosine_annealing':
    T_STEP = 10
    T_MULT = 2
    MIN_LR = 1e-6

# model config
# YOLOv3, YOLOv3_tiny, YOLOv4, YOLOv4_tiny, YOLOv4_csp
MODEL_TYPE = 'YOLOv4'
BASED_DTYPE = 'custom'

LOAD_CHECKPOINTS = False
CHECKPOINTS_DIR = 'checkpoints/' + BASED_DTYPE + '/'
TRAIN_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/train_loss/'
LOSS_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/val_loss/'
MAP_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/val_mAP/'
CHECKPOINTS = MAP_CHECKPOINTS_DIR + MODEL_TYPE

if 'tiny' in MODEL_TYPE:
    STRIDES = [16, 32]
else:
    STRIDES = [8, 16, 32]

if 'YOLOv3' in MODEL_TYPE:
    COORD = 5
    NOOBJ = 0.5

# log config
LOGDIR = 'logs/' + MODEL_TYPE + '_' + DTYPE + '_log'

# inference config
NMS_TYPE = 'soft_gaussian'
if 'soft' in NMS_TYPE:
    SCORE_THRESHOLD = 0.001
else:
    SCORE_THRESHOLD = 0.5
SIGMA = 0.3
OUTPUT_DIR = 'outputs/' + DTYPE + '/' + BASED_DTYPE + '_' + MODEL_TYPE + '/'

# cam config
VIDEO_PATH = 0

# labels and anchors config
if BASED_DTYPE =='voc':
    LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    if 'tiny' in MODEL_TYPE:
        ANCHORS = [[[0.1673583984375, 0.465576171875], [0.6298828125, 0.257080078125], [0.39990234375, 0.61962890625]], [[0.76513671875, 0.47216796875], [0.62841796875, 0.81787109375], [0.9189453125, 0.689453125]]]
    else:
        ANCHORS = [[[0.111572265625, 0.445068359375], [0.61767578125, 0.1873779296875], [0.255859375, 0.552734375]], [[0.5068359375, 0.427490234375], [0.42626953125, 0.73291015625], [0.86572265625, 0.42333984375]], [[0.66650390625, 0.64208984375], [0.65380859375, 0.95166015625], [0.9345703125, 0.69140625]]]

elif BASED_DTYPE == 'coco':
    LABELS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
              'bus', 'train', 'truck', 'boat', 'traffic light', 
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
              'cat', 'dog', 'horse', 'sheep', 'cow', 
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
              'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
              'wine glass', 'cup', 'fork', 'knife', 'spoon', 
              'bowl', 'banana', 'apple', 'sandwich', 'orange', 
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 
               'cake', 'chair', 'couch', 'potted plant', 'bed', 
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    if 'tiny' in MODEL_TYPE:
        ANCHORS = [[[0.1314697265625, 0.420166015625], [0.56884765625, 0.1724853515625], [0.3408203125, 0.52392578125]], [[0.71728515625, 0.37353515625], [0.57275390625, 0.7294921875], [0.8935546875, 0.62255859375]]]
    else:
        ANCHORS = [[[0.100341796875, 0.42431640625], [0.5556640625, 0.1165771484375], [0.3046875, 0.300048828125]], [[0.263916015625, 0.6298828125], [0.74951171875, 0.26611328125], [0.493408203125, 0.448974609375]], [[0.82177734375, 0.463134765625], [0.54638671875, 0.751953125], [0.8857421875, 0.7041015625]]]

elif BASED_DTYPE == 'custom':
    LABELS = ['Nam Joo-hyuk', 'Kim Da-mi', 'Kim Seong-cheol', 'Yoo Jae-suk', 
              'Kim Tae-ri', 'Choi Woo-shik']
    if 'tiny' in MODEL_TYPE:
        ANCHORS = [[[0.36083984375, 0.21044921875], [0.428466796875, 0.326904296875], [0.66064453125, 0.268310546875]], [[0.474853515625, 0.479736328125], [0.697265625, 0.40625], [0.6162109375, 0.68212890625]]]
    else:
        ANCHORS = [[[0.3447265625, 0.2054443359375], [0.39599609375, 0.3251953125], [0.53173828125, 0.279296875]], [[0.31640625, 0.6259765625], [0.474365234375, 0.41796875], [0.79833984375, 0.2493896484375]], [[0.70263671875, 0.38916015625], [0.57421875, 0.5478515625], [0.6689453125, 0.720703125]]]


NUM_CLASSES = len(LABELS)

# draw config
DRAW = True
