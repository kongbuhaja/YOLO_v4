import numpy as np
from utils import bbox_utils
from config import *

class stats:
    def __init__(self, labels=LABELS):
        self.stats = {}
        self.iou_threshold = [0.5+ i*0.05 for i in range(10)]
        self.mAP = 0.0
        self.mAP50 = 0.0
        for i, label in enumerate(labels):
            self.stats[i] = {
                'label': label,
                'total': 0,
                'tp': [[] for _ in range(len(self.iou_threshold))],
                'fp': [[] for _ in range(len(self.iou_threshold))],
                'scores':[],
            }
        

    def update_stats(self, preds, gt_labels):      
        pred_bboxes = preds[..., :4]
        pred_scores = preds[..., 4]
        pred_classes = preds[..., 5]

        gt_bboxes = gt_labels[..., :4]
        gt_classes = gt_labels[..., 5]

        if not gt_labels.shape[0]:
            for pred_class in pred_classes:
                for i in range(len(self.iou_threshold)):
                    self.stats[pred_class]['tp'][i] += [0]
                    self.stats[pred_class]['fp'][i] += [1]
            return

        u_classes, u_count = np.unique(gt_classes, return_counts=True)
        for i, u_class, in enumerate(u_classes):
            self.stats[int(u_class)]["total"] += u_count[i]

        if not preds.shape[0]:
            return
                
        ious = bbox_utils.bbox_iou(pred_bboxes[:,None], gt_bboxes[None], xywh=False, iou_type='iou').numpy()
        pred_ious = np.max(ious, -1)
        pred_ids = np.argmax(ious, -1)
        sorted_iou_ids = np.argsort(-pred_ious)

        past_ids = [[[]] for _ in range(len(self.iou_threshold))]
        for iou_id in sorted_iou_ids:             
            pred_class = int(pred_classes[iou_id])
            iou = pred_ious[iou_id]
            pred_id = pred_ids[iou_id]
            score = pred_scores[iou_id]
            
            gt_class = int(gt_classes[pred_id])
            
            self.stats[pred_class]['scores'].append(score)
            
            for i, iou_threshold in enumerate(self.iou_threshold):
                if pred_class == gt_class and pred_id not in past_ids[i] and iou >= iou_threshold:
                    self.stats[pred_class]['tp'][i] += [1]
                    self.stats[pred_class]['fp'][i] += [0]
                    past_ids[i] += [pred_id]
                else:
                    self.stats[pred_class]['tp'][i] += [0]
                    self.stats[pred_class]['fp'][i] += [1]
    
    def calculate_mAP(self):
        aps = []
        ap50s = []

        for label in self.stats.keys():
            label_stats = self.stats[label]
            ids = np.argsort(-np.array(label_stats['scores']))
            total = label_stats['total']
            
            cumsum_tps = np.cumsum(np.array(label_stats['tp'])[:, ids], -1)
            cumsum_fps = np.cumsum(np.array(label_stats['fp'])[:, ids], -1)
            
            if total == 0:
                recalls = np.zeros_like(cumsum_tps) + EPS
            else:
                recalls = cumsum_tps / total
            precisions = cumsum_tps / (cumsum_tps + cumsum_fps)
            ap, ap50 = self.calculate_AP(recalls, precisions)
        
            self.stats[label]['ap'] = ap
            self.stats[label]['ap50'] = ap50
            aps.append(ap)
            ap50s.append(ap50)

        self.mAP = np.mean(aps)
        self.mAP50 = np.mean(ap50s)
        return self.mAP
        
    def calculate_AP(self, recalls, precisions):
        aps = []
        for recall, precision in zip(recalls, precisions):
            ap = 0
            for r in np.arange(0, 1.01, 0.01):
                prec_rec = precision[recall >= r]
                if len(prec_rec) > 0:
                    ap += np.max(prec_rec)

            ap /= 101
            aps += [ap]
        return np.mean(aps), aps[0]
    
    def get_result(self):
        text = ''
        class_max_length = np.max(list(map(lambda x: len(x), LABELS)))
        block_max_length = 51 - class_max_length
        for label in self.stats.keys():
            ap = self.stats[label]['ap']
            ap50 = self.stats[label]['ap50']
            class_name = self.stats[label]['label']
            block = '■' * int(ap * block_max_length)
            text += f'{class_name:>{class_max_length}}|'
            text += f'{block:<{block_max_length}}|AP50:{ap50:.2f}, AP:{ap:.2f}\n'

        text += f'mAP50: {self.mAP50:.4f}, mAP: {self.mAP:.4f}'
        return text
        