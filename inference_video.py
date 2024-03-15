import cv2, time
import tensorflow as tf
import numpy as np
from models.model import load_model
from utils.io_utils import read_cfg
from utils.aug_utils import resize_padding_without_labels
from utils.bbox_utils import unresize_unpad_labels
from utils.draw_utils import Player

def main():
    cfg = read_cfg()
    cfg['model']['load'] = True

    show = bool(cfg['eval']['draw'])

    model, _, _, _, _ = load_model(cfg)
    player = Player(cfg)

    out_size = tf.cast(model.input_size, tf.float32)
    ratio = model.input_size / max(player.width, player.height)

    @tf.function
    def inference_step(frame):
        resized_frame, _, pad = resize_padding_without_labels(frame, out_size)
        preds = model(resized_frame[None])
        decoded_preds = model.output(preds)
        NMS_preds = model.NMS(decoded_preds[0])
        preds = unresize_unpad_labels(NMS_preds, pad, ratio)
        return preds
    
    def draw_step(frame, preds):
        player(frame, preds)
        if show:
            player.show()

    player.init_time()
    while(cv2.waitKey(1) != 27):        
        ret, origin_frame = player.read()
        if ret:
            preds = inference_step(origin_frame[..., ::-1]).numpy()
            draw_step(origin_frame, preds)
       
    player.release()

if __name__ == '__main__':
    main()