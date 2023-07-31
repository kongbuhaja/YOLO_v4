from utils import data_utils, train_utils, anchor_utils, post_processing, draw_utils, bbox_utils, eval_utils, io_utils
from utils.preset import preset
from config import *
import tensorflow as tf
import tqdm
import numpy as np


def main():
    anchors = list(map(lambda x: tf.reshape(x, [-1,4]), anchor_utils.get_anchors_xywh(ANCHORS, STRIDES, IMAGE_SIZE)))
    dataloader = data_utils.DataLoader(batch_size=BATCH_SIZE)
    test_dataset = dataloader('val', use_label=True)
    test_dataset_legnth = int(dataloader.length('val')//BATCH_SIZE)
    
    model, _, _, _, _ = train_utils.get_model(load_checkpoints=True)
    
    eval = eval_utils.Eval()
    
    all_images = []
    all_grids = []
    all_labels = []
    
    test_tqdm = tqdm.tqdm(test_dataset, total=test_dataset_legnth, desc=f'inference data')
    for batch_data in test_tqdm:
        batch_images = batch_data[0]
        batch_labels = batch_data[-1]
        
        all_images.append((batch_images.numpy()[...,::-1]*255.).astype(np.uint8))
        all_grids.append([grid.numpy() for grid in model(batch_images)])
        all_labels.append(batch_labels.numpy())        
        # break
        
    inference_tqdm = tqdm.tqdm(range(len(all_images)), desc=f'draw and calculate')
    for i in inference_tqdm:
        batch_images = all_images[i]
        batch_grids = all_grids[i]
        batch_labels = all_labels[i]
        batch_processed_preds = post_processing.prediction_to_bbox(batch_grids, anchors)
        for image, processed_preds, labels in zip(batch_images, batch_processed_preds, batch_labels):
            NMS_preds = post_processing.NMS(processed_preds).numpy()
            labels = bbox_utils.extract_real_labels(labels).numpy()
            if DRAW:
                pred = draw_utils.draw_labels(image.copy(), NMS_preds, xywh=False)
                origin = draw_utils.draw_labels(image.copy(), labels, xywh=False)
                output = np.concatenate([origin, pred], 1)
                draw_utils.show_and_save_image(output, just_save=True)

            eval.update_stats(NMS_preds, labels)
    eval.calculate_mAP()
    evaluation = eval.get_result()
    print(evaluation)
    io_utils.write_eval(evaluation)   
    
    
if __name__ == '__main__':
    preset()
    main()