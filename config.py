# hardware config
GPUS = 4

# model config
# YOLOv3, YOLOv3_tiny, YOLOv4, YOLOv4_tiny, YOLOv4_csp, YOLOv4_P5-7
MODEL_TYPE = 'YOLOv4_csp'
STRIDES = [8, 16, 32, 64, 128]
KERNEL_INITIALIZER = 'glorot'

# data config
DTYPE = 'coco'
MAX_BBOXES = 100
CREATE_ANCHORS = False
POSITIVE_IOU_THRESHOLD = 0.5

# train config
EPOCHS = 2000
BATCH_SIZE = 24
GLOBAL_BATCH_SIZE = BATCH_SIZE * GPUS
# cosine_annealing, poly, step
LR_SCHEDULER = 'cosine_annealing'
LR = 1e-3
IOU_THRESHOLD = 0.5
EPS = 1e-6
INF = 1e+30
EVAL_PER_EPOCHS = 5
WARMUP_EPOCHS = 10

LOAD_CHECKPOINTS = True
CHECKPOINTS_DIR = 'checkpoints/' + DTYPE + '/' + MODEL_TYPE + '/'
TRAIN_CHECKPOINTS_DIR = CHECKPOINTS_DIR + 'train_loss/'
LOSS_CHECKPOINTS_DIR = CHECKPOINTS_DIR + 'val_loss/'
MAP_CHECKPOINTS_DIR = CHECKPOINTS_DIR + 'val_mAP/'
CHECKPOINTS = MAP_CHECKPOINTS_DIR + MODEL_TYPE

# log config
LOGDIR = 'logs/' + MODEL_TYPE + '_' + DTYPE + '_log'

# inference config
NMS_TYPE = 'soft_gaussian'
DEFAULT_SCORE_THRESHOLD = 0.25
if 'soft' in NMS_TYPE:
    MINIMUM_SCORE_THRESHOLD = 0.001
else:
    MINIMUM_SCORE_THRESHOLD = DEFAULT_SCORE_THRESHOLD
SIGMA = 0.5
OUTPUT_DIR = 'outputs/' + DTYPE + '/' + MODEL_TYPE + '/'

# cam config
VIDEO_PATH = 0

# labels and anchors config
if DTYPE =='voc':
    LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person',
              'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    if 'tiny' in MODEL_TYPE:
        ANCHORS = [[[0.068359375, 0.09033203125], [0.174560546875, 0.2257080078125], [0.251708984375, 0.46142578125]], [[0.59423828125, 0.306884765625], [0.444091796875, 0.6396484375], [0.7998046875, 0.66162109375]]]
    elif MODEL_TYPE in ['YOLOv3', 'YOLOv4', 'YOLOv4_csp']:
        ANCHORS = [[[0.054229736328125, 0.0770263671875], [0.134765625, 0.154541015625], [0.155517578125, 0.337158203125]], [[0.344482421875, 0.241943359375], [0.265380859375, 0.5166015625], [0.7080078125, 0.351318359375]], [[0.436767578125, 0.59423828125], [0.62939453125, 0.7744140625], [0.88623046875, 0.625]]]
    elif MODEL_TYPE == 'YOLOv4_P5':
        ANCHORS = [[[0.051513671875, 0.06231689453125], [0.0826416015625, 0.1600341796875], [0.190673828125, 0.11907958984375], [0.1383056640625, 0.318115234375]], [[0.255126953125, 0.243408203125], [0.244384765625, 0.495361328125], [0.42431640625, 0.313720703125], [0.378173828125, 0.6484375]], [[0.81396484375, 0.301513671875], [0.5869140625, 0.509765625], [0.62109375, 0.822265625], [0.88916015625, 0.6337890625]]]
    elif MODEL_TYPE == 'YOLOv4_P6':
        ANCHORS = [[[0.043701171875, 0.058380126953125], [0.06890869140625, 0.1575927734375], [0.1307373046875, 0.08709716796875], [0.13720703125, 0.2169189453125]], [[0.26806640625, 0.1502685546875], [0.136962890625, 0.3828125], [0.253662109375, 0.299560546875], [0.21826171875, 0.5341796875]], [[0.60400390625, 0.2103271484375], [0.44873046875, 0.354248046875], [0.33642578125, 0.50048828125], [0.380126953125, 0.74560546875]], [[0.8251953125, 0.383056640625], [0.56982421875, 0.56494140625], [0.64306640625, 0.84765625], [0.89013671875, 0.654296875]]]
    elif MODEL_TYPE == 'YOLOv4_P7':
        ANCHORS = [[[0.04095458984375, 0.047637939453125], [0.056915283203125, 0.11407470703125], [0.1317138671875, 0.078369140625], [0.0863037109375, 0.1973876953125]], [[0.170654296875, 0.1600341796875], [0.1287841796875, 0.3056640625], [0.34814453125, 0.1619873046875], [0.2353515625, 0.27197265625]], [[0.174560546875, 0.475341796875], [0.275390625, 0.418212890625], [0.434814453125, 0.32421875], [0.7001953125, 0.2086181640625]], [[0.269287109375, 0.66650390625], [0.378662109375, 0.5263671875], [0.8173828125, 0.344482421875], [0.5654296875, 0.53466796875]], [[0.439453125, 0.7685546875], [0.837890625, 0.52294921875], [0.6611328125, 0.81787109375], [0.91259765625, 0.69970703125]]]

elif DTYPE == 'coco':
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
    elif MODEL_TYPE in ['YOLOv3', 'YOLOv4', 'YOLOv4_csp']:
        ANCHORS = [[[0.0285797119140625, 0.034210205078125], [0.07244873046875, 0.08563232421875], [0.11053466796875, 0.209228515625]], [[0.2216796875, 0.1068115234375], [0.314453125, 0.23974609375], [0.1898193359375, 0.41064453125]], [[0.6982421875, 0.2978515625], [0.397216796875, 0.56103515625], [0.79931640625, 0.67236328125]]]
    elif MODEL_TYPE == 'YOLOv4_P5':
        ANCHORS = [[[0.024200439453125, 0.028533935546875], [0.04327392578125, 0.08599853515625], [0.096923828125, 0.051055908203125], [0.0953369140625, 0.1529541015625]], [[0.246337890625, 0.094970703125], [0.12744140625, 0.33154296875], [0.211181640625, 0.205810546875], [0.3916015625, 0.2666015625]], [[0.251708984375, 0.483642578125], [0.76220703125, 0.306640625], [0.471435546875, 0.60205078125], [0.8447265625, 0.68017578125]]]
    elif MODEL_TYPE == 'YOLOv4_P6':
        ANCHORS = [[[0.0223846435546875, 0.024261474609375], [0.0372314453125, 0.06292724609375], [0.102783203125, 0.047576904296875], [0.06402587890625, 0.12310791015625]], [[0.1558837890625, 0.1048583984375], [0.0989990234375, 0.234619140625], [0.200439453125, 0.191162109375], [0.42236328125, 0.11956787109375]], [[0.168701171875, 0.39599609375], [0.308349609375, 0.276123046875], [0.283935546875, 0.5693359375], [0.71142578125, 0.2371826171875]], [[0.451171875, 0.40087890625], [0.505859375, 0.6962890625], [0.7958984375, 0.456298828125], [0.876953125, 0.734375]]]
    elif MODEL_TYPE == 'YOLOv4_P7':
        ANCHORS = [[[0.0190887451171875, 0.0233001708984375], [0.03302001953125, 0.062347412109375], [0.072998046875, 0.033599853515625], [0.05401611328125, 0.12548828125]], [[0.09417724609375, 0.0753173828125], [0.2225341796875, 0.0745849609375], [0.1339111328125, 0.1376953125], [0.08441162109375, 0.23388671875]], [[0.2301025390625, 0.1851806640625], [0.150146484375, 0.293701171875], [0.509765625, 0.12213134765625], [0.1785888671875, 0.50390625]], [[0.379638671875, 0.247802734375], [0.2705078125, 0.3564453125], [0.77783203125, 0.25732421875], [0.47705078125, 0.42041015625]], [[0.3251953125, 0.62109375], [0.806640625, 0.47412109375], [0.5537109375, 0.7265625], [0.90869140625, 0.7265625]]]


elif DTYPE == 'custom':
    LABELS = ['Nam Joo-hyuk', 'Kim Da-mi', 'Kim Seong-cheol', 'Yoo Jae-suk', 
              'Kim Tae-ri', 'Choi Woo-shik']
    if 'tiny' in MODEL_TYPE:
        ANCHORS = [[[0.08148193359375, 0.08953857421875], [0.126220703125, 0.1395263671875], [0.17041015625, 0.1866455078125]], [[0.22265625, 0.24462890625], [0.2958984375, 0.325927734375], [0.426513671875, 0.485595703125]]]
    elif MODEL_TYPE in ['YOLOv3', 'YOLOv4', 'YOLOv4_csp']:
        ANCHORS = [[[0.07672119140625, 0.08563232421875], [0.11700439453125, 0.1260986328125], [0.150146484375, 0.1695556640625]], [[0.1883544921875, 0.20263671875], [0.222412109375, 0.24462890625], [0.275390625, 0.302490234375]], [[0.345947265625, 0.384033203125], [0.4228515625, 0.480712890625], [0.5546875, 0.65625]]]
    elif MODEL_TYPE == 'YOLOv4_P5':
        ANCHORS = [[[0.0721435546875, 0.080078125], [0.10333251953125, 0.11175537109375], [0.1312255859375, 0.146728515625], [0.158447265625, 0.17578125]], [[0.1905517578125, 0.2060546875], [0.2183837890625, 0.2403564453125], [0.254150390625, 0.28173828125], [0.28857421875, 0.313720703125]], [[0.330810546875, 0.365478515625], [0.37451171875, 0.425537109375], [0.440673828125, 0.501953125], [0.5771484375, 0.6650390625]]]
    elif MODEL_TYPE == 'YOLOv4_P6':
        ANCHORS = [[[0.0654296875, 0.0726318359375], [0.09197998046875, 0.1002197265625], [0.12115478515625, 0.1282958984375], [0.1256103515625, 0.158935546875]], [[0.1483154296875, 0.1600341796875], [0.15625, 0.1783447265625], [0.1776123046875, 0.1883544921875], [0.1944580078125, 0.2099609375]], [[0.2122802734375, 0.237548828125], [0.2371826171875, 0.255859375], [0.260986328125, 0.284912109375], [0.286865234375, 0.31640625]], [[0.330810546875, 0.365478515625], [0.37451171875, 0.425537109375], [0.440673828125, 0.501953125], [0.5771484375, 0.6650390625]]]
    elif MODEL_TYPE == 'YOLOv4_P7':
        ANCHORS = [[[0.036956787109375, 0.044647216796875], [0.0692138671875, 0.07635498046875], [0.08587646484375, 0.09613037109375], [0.1044921875, 0.109130859375]], [[0.1173095703125, 0.1273193359375], [0.1339111328125, 0.1405029296875], [0.1199951171875, 0.1661376953125], [0.148681640625, 0.1593017578125]], [[0.1490478515625, 0.1787109375], [0.1632080078125, 0.1754150390625], [0.1778564453125, 0.1920166015625], [0.1962890625, 0.2100830078125]], [[0.2122802734375, 0.237548828125], [0.2371826171875, 0.255859375], [0.260986328125, 0.288330078125], [0.2900390625, 0.31689453125]], [[0.330810546875, 0.365478515625], [0.37451171875, 0.425537109375], [0.440673828125, 0.501953125], [0.5771484375, 0.6650390625]]]


NUM_CLASSES = len(LABELS)

# draw config
DRAW = True
SAVE = True
