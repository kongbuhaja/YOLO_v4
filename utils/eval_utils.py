import tensorflow as tf
import numpy as np
from utils import bbox_utils
from config import *
#### add ious, ids
class stats:
    def __init__(self, labels=LABELS, iou_threshold=IOU_THRESHOLD):
        self.stats = {}
        for i, label in enumerate(labels):
            self.stats[i] = {
                'label': label,
                'total': 0,
                'tp': [],
                'fp': [],
                'scores':[],
            }
        self.iou_threshold = iou_threshold
        self.mAP = 0.0

    def update_stats(self, preds, gt_labels):
        if len(preds)==0 or len(gt_labels)==0:
            return
        pred_bboxes = preds[..., :4]
        pred_scores = preds[..., 4]
        pred_classes = preds[..., 5]
        
        gt_bboxes = gt_labels[..., :4]
        gt_classes = gt_labels[..., 5]
                
        ious = bbox_utils.bbox_iou(pred_bboxes[:,None], gt_bboxes[None], xywh=False, iou_type='iou')
        pred_ious = tf.reduce_max(ious, -1)
        pred_ids = tf.argmax(ious, -1, output_type=tf.int32)
        sorted_pred_ids = tf.argsort(pred_ious, direction="DESCENDING")
        u_classes, u_idx, u_count = tf.unique_with_counts(tf.reshape(gt_classes, (-1)))
        for i, u_class, in enumerate(u_classes):
            self.stats[int(u_class)]["total"] += u_count[i]

        past_ids = []
        for sorted_pred_id in sorted_pred_ids:             
            pred_class = int(pred_classes[sorted_pred_id])
            iou = pred_ious[sorted_pred_id]
            pred_id = pred_ids[sorted_pred_id]
            score = pred_scores[sorted_pred_id]
            
            gt_class = int(gt_classes[pred_id])
            
            self.stats[pred_class]['scores'].append(score)
            
            if iou >= self.iou_threshold and pred_class == gt_class and pred_id not in past_ids:
                self.stats[pred_class]['tp'] += [1]
                self.stats[pred_class]['fp'] += [0]
                past_ids += [pred_id]
            else:
                self.stats[pred_class]['tp'] += [0]
                self.stats[pred_class]['fp'] += [1]
    
    def calculate_mAP(self):
        aps = []
        for label in self.stats.keys():
            label_stats = self.stats[label]
            ids = np.argsort(-np.array(label_stats['scores']))
            total = label_stats['total']
            
            cumsum_tp = np.cumsum(np.array(label_stats['tp'])[ids])
            cumsum_fp = np.cumsum(np.array(label_stats['fp'])[ids])
            
            if total == 0:
                recall = np.zeros_like(cumsum_tp) + EPS
            else:
                recall = cumsum_tp / total
            precision = cumsum_tp / (cumsum_tp + cumsum_fp)
            ap = self.calculate_AP(recall, precision)
        
            self.stats[label]['recall'] = recall
            self.stats[label]['precision'] = precision
            self.stats[label]['ap'] = ap
            aps.append(ap)
        self.mAP = np.mean(aps)
        return self.mAP
        
    def calculate_AP(self, recall, precision):
        ap = 0 
        for r in np.arange(0, 1.1, 0.1):
            prec_rec = precision[recall >= r]
            if len(prec_rec) > 0:
                ap += np.max(prec_rec)
        ap /= 11
        return ap
    
    def get_result(self):
        text = ''
        class_max_length = np.max(list(map(lambda x: len(x), LABELS)))
        block_max_length = 51 - class_max_length
        for label in self.stats.keys():
            ap = self.stats[label]['ap']
            class_name = self.stats[label]['label']
            block = '■' * int(ap * block_max_length)
            text += f'{class_name:>{class_max_length}}|'
            text += f'{block:<{block_max_length}}|{ap:.2f}\n'
        text += f'mAP{int(self.iou_threshold*100)}: {self.mAP:.8f}'
        return text
        