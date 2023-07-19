import numpy as np
from utils.bbox_utils import bbox_iou
from config import *

class stats:
    def __init__(self, labels=LABELS):
        self.stats = {}
        self.num_of_ious = 10
        self.iou_thresholds = np.array([0.5 + i * 0.05 for i in range(self.num_of_ious)])

        for i, label in enumerate(labels):
            self.stats[i]={'label': label,
                           'total': 0,
                           'score': [],
                           'tp': [],
                           'ap': np.array([0.] * self.num_of_ious)}

    def init_stat(self):
        for i in range(len(self.stats)):
            self.stats[i]['total'] = 0
            self.stats[i]['score'] = []
            self.stats[i]['tp'] = []
            self.stats[i]['ap'] = np.array([0.] * self.num_of_ious)

    def update_stats(self, pred, gt):
        if gt.shape[0] == 0 and pred.shape[0] == 0:
            return

        pred_bboxes = pred[..., :4]
        pred_scores = pred[..., 4]
        pred_classes = pred[..., 5].astype(np.int32)
        

        gt_bboxes = gt[..., :4]
        gt_classes = gt[..., 5].astype(np.int32)
        gt_unique_classes, counts = np.unique(gt_classes, return_counts=True)

        for gt_unique_class, count in zip(gt_unique_classes, counts):
            self.stats[gt_unique_class]['total'] += count  

        if pred.shape[0] == 0:
            if gt.shape[0] != 0:
                for gt_unique_classes, count in zip(gt_unique_classes, counts):
                    self.stats[gt_unique_class]['score'] += [0]
                    self.stats[gt_unique_class]['tp'] += [np.array([0] * self.num_of_ious, bool)] * count
            return

        elif gt.shape[0] == 0:
            pred_unique_classes, counts = np.unique(pred_classes, return_counts=True)
            for pred_unique_class, count in zip(pred_unique_classes, counts):
                scores = pred_scores[pred_classes==pred_unique_class]
                self.stats[pred_unique_class]['score'] += scores.tolist()
                self.stats[pred_unique_class]['tp'] += [np.array([0] * self.num_of_ious, bool)] * count

        else:
            for gt_unique_class in gt_unique_classes:
                matched_pred_bboxes = pred_bboxes[pred_classes==gt_unique_class]
                matched_pred_scores = pred_scores[pred_classes==gt_unique_class]

                matched_gt_bboxes = gt_bboxes[gt_classes==gt_unique_class]

                if matched_gt_bboxes.shape[0] == 0:
                    for score in matched_pred_scores:
                        self.stats[gt_unique_class]['score'] += [score]
                        self.stats[gt_unique_class]['tp'] += [np.array([0] * self.num_of_ious, bool)] * count
                else:
                    matched_ious = bbox_iou(matched_pred_bboxes[:, None], matched_gt_bboxes[None], xywh=False).numpy()
                    matched_max_ious = np.max(matched_ious, -1)
                    matched_pred_ids = np.argmax(matched_ious, -1)

                    past_ids = [[] for i in self.iou_thresholds]
                    for matched_pred_score, matched_max_iou, matched_pred_id in zip(matched_pred_scores, matched_max_ious, matched_pred_ids):
                        result = matched_max_iou >= self.iou_thresholds
                        for i in range(self.num_of_ious):
                            if matched_pred_id in past_ids[i]:
                                result[i] = False
                            else:
                                past_ids[i] += [matched_pred_id]
                        self.stats[gt_unique_class]['score'] += [matched_pred_score]
                        self.stats[gt_unique_class]['tp'] += [result]

        for gt_unique_class, count in zip(gt_unique_classes, counts):
            self.stats[gt_unique_class]['total'] += count  

    def calculate_mAP(self):
        for label_id in self.stats.keys():
            ids = np.argsort(-np.array(self.stats[label_id]['score']))
            total = self.stats[label_id]['total']
            if len(self.stats[label_id]['tp'])==0:
                ap = np.array([0.] * self.num_of_ious)
            else:
                cumsum_tps = np.cumsum(np.array(self.stats[label_id]['tp'])[ids], 0)
                cumsum_fps = np.cumsum(1 - np.array(self.stats[label_id]['tp'])[ids], 0)
                recalls = cumsum_tps / (total + 1e-10)
                precisions = cumsum_tps / (cumsum_tps + cumsum_fps + 1e-10)
                ap = self.calculate_AP(recalls, precisions)

            self.stats[label_id]['ap'] = ap
 
        return self.get_AP(label='all')

    def calculate_AP(self, recalls, precisions):
        ap = []
        mrec = np.concatenate([np.array([[0.] * self.num_of_ious]), recalls, np.array([[1.] * self.num_of_ious])], 0)
        mpre = np.concatenate([np.array([[1.] * self.num_of_ious]), precisions, np.array([[0.] * self.num_of_ious])], 0)
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        x = np.linspace(0, 1, 101)

        for i in range(self.num_of_ious):
            ap += [np.trapz(np.interp(x, mrec[..., i], mpre[..., i]), x)]
        return ap

    def get_AP(self, label='all'):
        aps = []
        if label=='all':
            for label_id in self.stats.keys():
                if len(self.stats[label_id]['ap']) != 0:
                    aps += [self.stats[label_id]['ap']]
                else:
                    aps += [np.array([0]*10)]
            ap = np.mean(aps, 0)
        else:
            if type(label) == str:
                for label_id in self.stats.keys():
                    if self.stats[label_id]['label']==label:
                        ap = self.stats[label_id]['ap']
            else:
                ap = self.stats[label]['ap']
            if len(ap) == 0:
                ap = np.array([0.]*10)
        return ap[0], np.mean(ap)

    def get_result(self):
        text = ''
        class_max_length = np.max(list(map(lambda x: len(x), LABELS)))
        block_max_length = 51 - class_max_length
        for label_id in self.stats.keys():
            ap50, ap = self.get_AP(label_id)
            class_name = self.stats[label_id]['label']
            block = '■' * int(ap * block_max_length)
            text += f'{class_name:>{class_max_length}}|'
            text += f'{block:<{block_max_length}}|AP50:{ap50:.2f}, AP:{ap:.2f}\n'

        mAP50, mAP = self.get_AP()
        text += f'mAP50: {mAP50:.4f}, mAP: {mAP:.4f}'
        return text