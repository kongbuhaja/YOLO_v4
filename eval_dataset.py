import tqdm
import tensorflow as tf
import numpy as np
from models.model import load_model
from utils.data_utils import DataLoader
from utils.eval_utils import Eval
from utils.io_utils import read_cfg
from utils.draw_utils import Painter
from utils.aug_utils import resize_padding_without_labels
from utils.bbox_utils import unresize_unpad_labels, xywh_to_xyxy

def main():
    cfg = read_cfg()
    cfg['model']['load'] = True

    draw = cfg['eval']['draw']
    
    model, _, _ = load_model(cfg)
    dataloader = DataLoader(cfg)
    eval = Eval(cfg)
    painter = Painter(cfg)
    
    valid_dataset = dataloader('val', cfg['eval']['batch_size'], resize=False)
    valid_dataset_length = dataloader.length['val'] // cfg['eval']['batch_size']
    out_size = tf.cast(cfg['model']['input_size'], tf.float32)

    def inference_step(image):
        processd_image, pad, ratio = resize_padding_without_labels(image, out_size)
        preds = model(processd_image[None])
        preds = model.decoder.bbox_decode(preds)
        preds = model.decoder.NMS(preds[0])
        preds = unresize_unpad_labels(preds, pad, ratio)

        return preds.numpy()

    def update_eval_step(labels, preds):
        eval.update(labels, preds)
    
    def draw_step(image, labels, preds):
        if draw//10:
            labels = xywh_to_xyxy(labels).numpy()
            preds = xywh_to_xyxy(preds).numpy()
            image = (image*255).numpy().astype(np.uint8)
            painter.draw_image(image, labels, preds)
    
    eval_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_length, ncols=160, desc=f'Evaluate', ascii=' =', colour='blue')
    for image, labels in eval_tqdm:
        image, labels = image[0], labels[:, 1:].numpy()
        preds = inference_step(image)
        update_eval_step(labels, preds)
        draw_step(image, labels, preds)

    eval.compute_mAP()
    evaluation = eval.get_result()
    eval.write_eval(evaluation)
    print(evaluation)
    
if __name__ == '__main__':
    main()