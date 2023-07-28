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
    SCORE_THRESHOLD = 0.01
else:
    SCORE_THRESHOLD = 0.5
SIGMA = 0.5
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
        ANCHORS = [[[0.068359375, 0.09033203125], [0.174560546875, 0.2257080078125], [0.251708984375, 0.46142578125]], [[0.59423828125, 0.306884765625], [0.444091796875, 0.6396484375], [0.7998046875, 0.66162109375]]]
    else:
        ANCHORS = [[[0.054229736328125, 0.0770263671875], [0.134765625, 0.154541015625], [0.155517578125, 0.337158203125]], [[0.344482421875, 0.241943359375], [0.265380859375, 0.5166015625], [0.7080078125, 0.351318359375]], [[0.436767578125, 0.59423828125], [0.62939453125, 0.7744140625], [0.88623046875, 0.625]]]

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
        ANCHORS = [[[0.0343017578125, 0.041412353515625], [0.10675048828125, 0.11260986328125], [0.1656494140625, 0.283203125]], [[0.428466796875, 0.189208984375], [0.34814453125, 0.498291015625], [0.759765625, 0.5908203125]]]
    else:
        ANCHORS = [[[0.0285797119140625, 0.034210205078125], [0.07244873046875, 0.08563232421875], [0.11053466796875, 0.209228515625]], [[0.2216796875, 0.1068115234375], [0.314453125, 0.23974609375], [0.1898193359375, 0.41064453125]], [[0.6982421875, 0.2978515625], [0.397216796875, 0.56103515625], [0.79931640625, 0.67236328125]]]

elif BASED_DTYPE == 'custom':
    LABELS = ['Nam Joo-hyuk', 'Kim Da-mi', 'Kim Seong-cheol', 'Yoo Jae-suk', 
              'Kim Tae-ri', 'Choi Woo-shik']
    if 'tiny' in MODEL_TYPE:
        ANCHORS = [[[0.08148193359375, 0.08953857421875], [0.126220703125, 0.1395263671875], [0.17041015625, 0.1866455078125]], [[0.22265625, 0.24462890625], [0.2958984375, 0.325927734375], [0.426513671875, 0.485595703125]]]
    else:
        ANCHORS = [[[0.07672119140625, 0.08563232421875], [0.11700439453125, 0.1260986328125], [0.150146484375, 0.1695556640625]], [[0.1883544921875, 0.20263671875], [0.222412109375, 0.24462890625], [0.275390625, 0.302490234375]], [[0.345947265625, 0.384033203125], [0.4228515625, 0.480712890625], [0.5546875, 0.65625]]]


NUM_CLASSES = len(LABELS)

# draw config
DRAW = True
