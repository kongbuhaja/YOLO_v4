from losses import yolov2, yolov3, yolov4

def get_loss(cfg):
        if cfg['model']['loss'] == 'v4':
            return yolov4.loss(cfg['model']['input_size'], cfg['model']['anchors'], cfg['model']['strides'], 
                               cfg['data']['labels']['count'], cfg['train']['assign'], cfg['train']['focal'])
        elif cfg['model']['loss'] == 'v3':
            return yolov3.loss(cfg['model']['input_size'], cfg['model']['anchors'], cfg['model']['strides'], 
                               cfg['data']['labels']['count'], cfg['train']['assign'], cfg['train']['focal'])
        elif cfg['model']['loss'] == 'v2':
            return yolov2.loss(cfg['model']['input_size'], cfg['model']['anchors'], cfg['model']['strides'], 
                               cfg['data']['labels']['count'], cfg['train']['assign'], cfg['train']['focal'])
