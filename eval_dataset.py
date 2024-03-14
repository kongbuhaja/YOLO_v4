import tqdm
import tensorflow as tf
import numpy as np
from models.model import load_model
from utils.data_utils import DataLoader
from utils.eval_utils import Eval
from utils.io_utils import read_cfg
from utils.draw_utils import Painter
from utils.augmentation import eval_resize_padding
from utils.bbox_utils import unresize_unpad_labels

def main():
    cfg = read_cfg()
    cfg['model']['load'] = True

    draw = cfg['eval']['draw']
    
    model, _, _, _, _ = load_model(cfg)
    dataloader = DataLoader(cfg)
    eval = Eval(cfg)
    painter = Painter(cfg)
    
    valid_dataset = dataloader('val')
    valid_dataset_legnth = dataloader.length['val']
    out_size = tf.cast(model.input_size, tf.float32)

    def inference_step(image):
        processd_image, pad = eval_resize_padding(image, out_size)
        ratio = out_size / tf.cast(tf.reduce_max(tf.shape(image)[:2]), tf.float32)
        input = processd_image[None]
        preds = model(input)
        preds = model.output(preds)
        NMS_preds = model.NMS(preds[0])
        preds = unresize_unpad_labels(NMS_preds, pad, ratio)
        return preds

    def update_eval_step(preds, labels):
        eval.update_stats(preds, labels)
    
    def draw_step(image, labels, preds):
        if draw//10:
            image = (image*255).numpy().astype(np.uint8)
            painter.draw_image(image, labels, preds)
    
    eval_tqdm = tqdm.tqdm(valid_dataset, total=valid_dataset_legnth, ncols=160, desc=f'Evaluate', ascii=' =', colour='red')
    for image, labels in eval_tqdm:
        image, labels = image[0], labels.numpy()
        preds = inference_step(image).numpy()
        update_eval_step(preds, labels)
        draw_step(image, labels, preds)

    eval.calculate_mAP()
    evaluation = eval.get_result()
    eval.write_eval(evaluation)
    print(evaluation)
    
if __name__ == '__main__':
    main()