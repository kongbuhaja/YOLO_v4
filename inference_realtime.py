import cv2, glob, os, time
from utils.preset import preset
import numpy as np
import tensorflow as tf
from config import *
from utils import bbox_utils, draw_utils, io_utils, train_utils, aug_utils, post_processing, anchor_utils

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ratio = min(IMAGE_SIZE/width, IMAGE_SIZE/height)
    pad_w = (IMAGE_SIZE - width * ratio) // 2
    pad_h = (IMAGE_SIZE - height * ratio) // 2
    print(f'video path:{VIDEO_PATH}')
    print(f'video size:{width} x {height}')
    print(f'video fps:{fps}')
        
    filepath = OUTPUT_DIR + 'video/' + 'inference_'
    filepath += str(len(glob.glob(filepath+'*.mp4')))
    writer = cv2.VideoWriter(filepath + 'avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    
    model, _, _ = train_utils.get_model()
    anchors = list(map(lambda x: tf.reshape(x,[-1,4]), anchor_utils.get_anchors_xywh(ANCHORS, STRIDES, IMAGE_SIZE)))

    
    total_frames = 0 
    start_time = time.time()
    prev_time = 0
    while(cv2.waitKey(1) != 27):
        cur_time = time.time()
        
        ret, origin_frame = cap.read()
        if ret:
            total_frames += 1
            resized_frame, pad = aug_utils.tf_resize_padding(cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB), 
                                                             tf.zeros((1,5)), width, height, IMAGE_SIZE)
            grids = model(tf.cast(resized_frame[None], tf.float32)/255.)
            
            processed_preds = post_processing.prediction_to_bbox(grids, anchors)
            NMS_preds = post_processing.NMS(processed_preds)

            NMS_bboxes = (NMS_preds[..., :4] - pad[..., :4])/ratio
            
            NMS_preds = tf.concat([NMS_bboxes, NMS_preds[4:]])
            pred = draw_utils.draw_labels(origin_frame, NMS_preds)
                        
            sec = cur_time - prev_time
            fps = 1 / sec
            prev_time = cur_time
            text = f'Inference Time: {sec:.3f}, FPS: {fps:.1f}'
            
            cv2.putText(pred, text, (10,10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255,255,255), 1)
            cv2.imshow('Inference', pred)
            
            writer.write(pred)
       
    end_time = time.time()
    sec = end_time - start_time
    avg_fps = total_frames / sec
    print(f'Average Inferrence Time:{1/avg_fps:.3f}, Average FPS:{avg_fps:.3f}')
    writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    preset()
    main()