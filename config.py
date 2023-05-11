# data config
DTYPE = 'voc'
IMAGE_SIZE = 416
BATCH_SIZE = 16
MAX_BBOXES = 100
CREATE_ANCHORS = False

# train config
EPOCHS = 200
LR = 1e-2
LR_SCHEDULER = 'cosine_annealing'
IOU_THRESHOLD = 0.5
EPS = 1e-7
INF = 1e+30
EVAL_PER_EPOCHS = 1
WARMUP_EPOCHS = 5

if LR_SCHEDULER == 'poly':
    POWER = 0.9
elif LR_SCHEDULER == 'cosine_annealing':
    T_STEP = 10
    T_MULT = 2
    MIN_LR = 1e-7

# model config
MODEL_TYPE = 'YOLOv4'
BASED_DTYPE = 'voc'
LOAD_CHECKPOINTS = True
CHECKPOINTS_DIR = 'checkpoints/' + BASED_DTYPE + '/'
TRAIN_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/train_loss/'
LOSS_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/val_loss/'
MAP_CHECKPOINTS_DIR = CHECKPOINTS_DIR + MODEL_TYPE + '/val_mAP/'
CHECKPOINTS = MAP_CHECKPOINTS_DIR + MODEL_TYPE

if 'tiny' in MODEL_TYPE:
    STRIDES = [16, 32]
else:
    STRIDES = [8, 16, 32]

if 'YOLOv3' in MODEL_TYPE or True:
    COORD = 5
    NOOBJ = 0.5

# log config
LOGDIR = 'logs/' + MODEL_TYPE + '_' + DTYPE + '_log'

# inference config
NMS_TYPE = 'soft_gaussian'
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
    if MODEL_TYPE in ['YOLOv3', 'YOLOv4']:
        ANCHORS = [[[22, 31], [55, 64], [69, 144]], [[141, 89], [118, 227], [188, 166]], [[335, 146], [220, 290], [356, 283]]]
    elif MODEL_TYPE in ['YOLOv3_tiny', 'YOLOv4_tiny']:
        ANCHORS = [[[28, 37], [72, 91], [102, 187]], [[249, 127], [180, 262], [330, 278]]]

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
    if MODEL_TYPE in ['YOLOv3', 'YOLOv4']:
        ANCHORS = [[[41, 176], [231, 48], [126, 124]], [[109, 261], [311, 110], [205, 186]], [[341, 192], [227, 312], [368, 292]]]
    elif MODEL_TYPE in ['YOLOv3_tiny', 'YOLOv4_tiny']:
        ANCHORS = [[[9, 37], [71, 89], [100, 183]], [[244, 125], [179, 259], [331, 273]]]
           
elif BASED_DTYPE == 'custom':
    LABELS = ['Nam Joo-hyuk', 'Kim Da-mi', 'Kim Seong-cheol', 'Yoo Jae-suk', 
              'Kim Tae-ri', 'Choi Woo-shik']
    if MODEL_TYPE in ['YOLOv3', 'YOLOv4']:
        ANCHORS = [[[31, 35], [48, 52], [60, 68]], [[71, 78], [83, 90], [96, 106]], [[118, 129], [152, 171], [200, 228]]]
    elif MODEL_TYPE in ['YOLOv3_tiny', 'YOLOv4_tiny']:
        ANCHORS = [[[36, 39], [60, 67], [85, 93]], [[113, 125], [151, 170], [200, 228]]]

NUM_CLASSES = len(LABELS)
NUM_ANCHORS = len(ANCHORS)

# draw config
DRAW = False