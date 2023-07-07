import numpy as np
from utils import bbox_utils
from config import *

class stats:
    def __init__(self, labels=LABELS):
        self.stats = {}
        self.iou_threshold = np.array([0.5+ i*0.05 for i in range(10)])
        self.mAP = 0.0
        self.mAP50 = 0.0
        for i, label in enumerate(labels):
            self.stats[i] = {
                'label': label,
                'total': 0,
                'tp': [],
                'fp': [],
                'scores':[]
            }
    
    def init_stat(self):
        for i in range(len(self.stats)):
            self.stats[i]['total'] = 0
            self.stats[i]['tp'] = []
            self.stats[i]['fp'] = []
            self.stats[i]['scores'] = []


    def update_stats(self, preds, gt_labels):      
        pred_bboxes = preds[..., :4].numpy()
        pred_scores = preds[..., 4].numpy()
        pred_classes = preds[..., 5].numpy()

        gt_bboxes = gt_labels[..., :4].numpy()
        gt_classes = gt_labels[..., 5].numpy()

        if not gt_labels.shape[0]:
            for pred_class in pred_classes.astype(np.int32):
                self.stats[pred_class]['tp'] += [np.array([0] * len(self.iou_threshold))]
                self.stats[pred_class]['fp'] += [np.array([1] * len(self.iou_threshold))]
            return

        u_classes, u_count = np.unique(gt_classes, return_counts=True)
        for i, u_class, in enumerate(u_classes.astype(np.int32)):
            self.stats[u_class]["total"] += u_count[i]

        if not preds.shape[0]:
            return
                
        for u_class in np.unique(pred_classes):
            cls_gt_bboxes = gt_bboxes[gt_classes==u_class]
            if cls_gt_bboxes.shape[0] == 0:
                self.stats[u_class]['tp'] += [np.array([0] * len(self.iou_threshold))]
                self.stats[u_class]['fp'] += [np.array([1] * len(self.iou_threshold))]
            else:
                cls_pred_bboxes = pred_bboxes[pred_classes==u_class]
                cls_pred_scores = pred_scores[pred_classes==u_class]
                cls_ious = bbox_utils.bbox_iou(cls_pred_bboxes[:, None], cls_gt_bboxes[None], xywh=False).numpy()
            
                cls_max_ious = np.max(cls_ious, -1)
                # sorted_pred_ids = np.argsort(-cls_max_ious) # past_ids 뺄거면 정렬 지우자

                # for pred_id in sorted_pred_ids:
                for pred_id in range(len(cls_pred_scores)):
                    iou = cls_max_ious[pred_id]
                    score = cls_pred_scores[pred_id]
                            
                    self.stats[u_class]['scores'].append(score)
                
                    tp = (iou >= self.iou_threshold).astype(int)
                    self.stats[u_class]['tp'] += [tp]
                    self.stats[u_class]['fp'] += [1 - tp]

    def calculate_mAP(self):
        aps = []
        ap50s = []

        for label in self.stats.keys():
            label_stats = self.stats[label]
            ids = np.argsort(-np.array(label_stats['scores']))
            total = label_stats['total']
            
            cumsum_tps = np.cumsum(np.array(label_stats['tp'])[ids], 0)
            cumsum_fps = np.cumsum(np.array(label_stats['fp'])[ids], 0)
            
            if cumsum_tps.shape[0] == 0:
                ap = 0
                ap50 = 0
            else:
                recalls = cumsum_tps / total
                precisions = cumsum_tps / (cumsum_tps + cumsum_fps)
                ap, ap50 = self.calculate_AP(recalls, precisions)
                
            self.stats[label]['ap'] = ap
            self.stats[label]['ap50'] = ap50
            aps += [ap]
            ap50s += [ap50]

        self.mAP = np.mean(aps)
        self.mAP50 = np.mean(ap50s)
        return self.mAP50, self.mAP
        
    def calculate_AP(self, recalls, precisions):
        aps = np.array([0.] * len(self.iou_threshold))
        for r in np.arange(0, 1.01, 0.01):
            prec_rec = np.where(recalls >= r, precisions, 0)
            aps += np.amax(prec_rec, 0)
        aps /= 101
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
        