from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, HeUniform, HeNormal, Zeros
from models.backbones import *
from models.necks import *
from models.heads import *

def get_kernel_initializer(cfg):
    if cfg['model']['kernel_init'] == 'glorot_uniform':
        return GlorotUniform(seed=cfg['seed'])
    elif cfg['model']['kernel_init'] == 'glorot_normal':
        return GlorotNormal(seed=cfg['seed'])
    elif cfg['model']['kernel_init'] == 'he_uniform':
        return HeUniform(seed=cfg['seed'])
    elif cfg['model']['kernel_init'] == 'he_normal':
        return HeNormal(seed=cfg['seed'])
    else:
        return Zeros()
    
def get_backbone(cfg):
    if cfg['model']['backbone']['name'] == 'CSPP':
        return CSPP(cfg['model']['backbone']['unit'], len(cfg['model']['anchors'])+2, activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['backbone']['name'] == 'CSPDarknet53':
        return CSPDarknet53(cfg['model']['backbone']['unit'], cfg['model']['backbone']['csp'], activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['backbone']['name'] == 'Darknet53':
        return Darknet53(cfg['model']['backbone']['unit'], activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['backbone']['name'] == 'CSPDarknet19':
        return CSPDarknet19(cfg['model']['backbone']['unit'], activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['backbone']['name'] == 'Darknet19':
        return Darknet19(cfg['model']['backbone']['unit'], activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['backbone']['name'] == 'Darknet19_v2':
        return Darknet19_v2(cfg['model']['backbone']['unit'], activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['backbone']['name'] == 'Darknet19_v2_tiny':
        return Darknet19_v2_tiny(cfg['model']['backbone']['unit'], activate=cfg['model']['backbone']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    else:
        return None
    
def get_neck(cfg):
    if cfg['model']['neck']['name'] == 'CSPPANSPP':
        return CSPPANSPP(cfg['model']['neck']['unit'], len(cfg['model']['anchors']), cfg['model']['neck']['block_size'], activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['neck']['name'] == 'PANSPP':
        return PANSPP(cfg['model']['neck']['unit'], len(cfg['model']['anchors']), cfg['model']['neck']['block_size'], activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['neck']['name'] == 'CSPFPN':
        return CSPFPN(cfg['model']['neck']['unit'], len(cfg['model']['anchors']), cfg['model']['neck']['block_size'], activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['neck']['name'] == 'FPN':
        return FPN(cfg['model']['neck']['unit'], len(cfg['model']['anchors']), cfg['model']['neck']['block_size'], activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['neck']['name'] == 'tinyFPN':
        return tinyFPN(cfg['model']['neck']['unit'], len(cfg['model']['anchors']), activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['neck']['name'] == 'reOrg':
        return reOrg(cfg['model']['neck']['unit'], activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    elif cfg['model']['neck']['name'] == 'Conv':
        return ConvNeck(cfg['model']['neck']['unit'], activate=cfg['model']['neck']['activate'], kernel_initializer=cfg['model']['kernel_init'])
    else:
        return None
    
def get_head(cfg):
    if cfg['model']['head']['name'] == 'Detect':
        return Detect(len(cfg['model']['anchors']), len(cfg['model']['anchors'][0]), cfg['data']['labels']['count'], kernel_initializer=cfg['model']['kernel_init'])
        