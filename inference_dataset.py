from utils import data_utils, train_utils, anchor_utils, post_processing, draw_utils, bbox_utils, eval_utils, io_utils
from utils.preset import preset
from config import *
import tensorflow as tf
import tqdm
import numpy as np


def main():
    model, _, _, _, _ = train_utils.load_model(MODEL_TYPE, ANCHORS, NUM_CLASSES, STRIDES, IOU_THRESHOLD,
                                               EPS, INF, KERNEL_INITIALIZER, True, CHECKPOINTS)
    dataloader = data_utils.DataLoader(DTYPE, LABELS, BATCH_SIZE, ANCHORS, NUM_CLASSES, 
                                       model.input_size, model.strides, POSITIVE_IOU_THRESHOLD, MAX_BBOXES, 
                                       CREATE_ANCHORS)
    
    test_dataset = dataloader('val', use_label=True)
    test_dataset_legnth = dataloader.length('val') // BATCH_SIZE
        
    anchors_xywh = list(map(lambda x: tf.reshape(x, [-1,4]), model.anchors_xywh))

    eval = eval_utils.Eval(LABELS, EPS)
    
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
        
    inference_tqdm = tqdm.tqdm(range(len(all_images)), desc=f'draw and calculate')
    for i in inference_tqdm:
        batch_images = all_images[i]
        batch_grids = all_grids[i]
        batch_labels = all_labels[i]
        batch_processed_preds = post_processing.prediction_to_bbox(batch_grids, anchors_xywh, BATCH_SIZE, model.strides, NUM_CLASSES, model.input_size)
        for image, processed_preds, labels in zip(batch_images, batch_processed_preds, batch_labels):
            NMS_preds = post_processing.NMS(processed_preds, SCORE_THRESHOLD, IOU_THRESHOLD, NMS_TYPE, SIGMA).numpy()
            labels = bbox_utils.extract_real_labels(labels).numpy()
            if DRAW:
                pred = draw_utils.draw_labels(image.copy(), NMS_preds, LABELS, xywh=False)
                origin = draw_utils.draw_labels(image.copy(), labels, LABELS, xywh=False)
                output = np.concatenate([origin, pred], 1)
                draw_utils.draw_image(output, model.input_size, OUTPUT_DIR, save=SAVE)

            eval.update_stats(NMS_preds, labels)
    eval.calculate_mAP()
    evaluation = eval.get_result()
    print(evaluation)
    io_utils.write_eval(evaluation, OUTPUT_DIR)   
    
    
if __name__ == '__main__':
    preset()
    main()