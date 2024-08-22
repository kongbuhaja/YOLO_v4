import numpy as np
from utils.bbox_utils import bbox_iou_np

class Eval:
    def __init__(self, cfg, eps=1e-10):
        self.labels=cfg['data']['labels']['name']
        self.eval_per_epoch = cfg['train']['eval_per_epoch']
        self.result_dir = cfg['eval']['dir']
        self.warmup_epochs = cfg['train']['lr_scheduler']['warmup_epochs']
        self.eps = eps
        self.num_of_threshold = 10
        self.iou_thresholds = np.linspace(0.5, 0.95, self.num_of_threshold)
        self.stats = []
        self.ap = np.zeros((len(self.labels), self.num_of_threshold))

    def check(self, epoch):
        return epoch >= self.warmup_epochs and epoch % self.eval_per_epoch == 0

    def init_stat(self):
        self.stats = dict(tp=[], conf=[], pred_cls=[], gt_cls=[])

    def update(self, gt, pred):
        tp = np.zeros((pred.shape[0], self.iou_thresholds.shape[0])).astype(bool)
        pred_cls = pred[:, -1]
        pred_conf = pred[:, -2]
        gt_cls = gt[:, -1]
        cls_mask = pred_cls[:, None] == gt_cls
        ious = bbox_iou_np(pred[:, None, :4], gt[:, :4])
        ious = ious * cls_mask
        
        for i, iou_threshold in enumerate(self.iou_thresholds):
            matches = np.nonzero(ious >= iou_threshold)
            matches = np.array(matches).T
            if matches.shape[0]:
                if matches.shape[0] > 1:
                    matches = matches[ious[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                    matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                try:
                    tp[matches[:, 0].astype(int), i] = True
                except:
                    
                    print(matches)
                    print()
                    print(tp.shape)
                    print()
                    print(pred, gt)
                    tp[matches[:, 0].astype(int), i] = True

        self.stats['conf'] += [pred_conf]
        self.stats['pred_cls'] += [pred_cls]
        self.stats['gt_cls'] += [gt_cls]
        self.stats['tp'] += [tp]

    def compute_mAP(self):
        tp, conf, pred_cls, gt_cls = self.extract_stat()
        idx = np.argsort(-conf)
        tp, conf, pred_cls = tp[idx], conf[idx], pred_cls[idx]

        for c, n_g in zip(*np.unique(gt_cls.astype(np.int32), return_counts=True)):
            i = pred_cls == c
            n_p = i.sum()

            if n_p == 0 or n_g == 0:
                continue
            
            tpc = tp[i].cumsum(0)
            fpc = (1 - tp[i]).cumsum(0)

            recall = tpc / (n_g + self.eps)

            precision = tpc / (tpc + fpc)
 
            for j in range(self.num_of_threshold):
                self.ap[c, j] = self.compute_ap(recall[:, j], precision[:, j])
        
        return np.mean(self.ap[:, 0]), np.mean(self.ap)

    def compute_ap(self, recall, precision):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([1.], precision, [0.]))

        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)

        return ap
    
    def extract_stat(self):
            stat = {key: np.concatenate(value, 0) for key, value in self.stats.items()}
            def extract(tp, conf, gt_cls, pred_cls):
                return tp, conf, pred_cls, gt_cls
            return extract(**stat)

    def get_result(self):
        text = ''
        class_max_length = np.max(list(map(lambda x: len(x), self.labels)))
        block_max_length = 51 - class_max_length

        for label_id, label in enumerate(self.labels):
            ap50, ap = self.ap[label_id, 0], np.mean(self.ap[label_id])
            block = '■' * int(ap * block_max_length)
            text += f'{label:>{class_max_length}}|'
            text += f'{block:<{block_max_length}}|AP50:{ap50:.2f}, AP:{ap:.2f}\n'

        mAP50, mAP = np.mean(self.ap[:, 0]), np.mean(self.ap)
        text += f'mAP50: {mAP50:.4f}, mAP: {mAP:.4f}'
        return text
    
    def write_eval(self, text):
        path = f'{self.result_dir}/evaluation.txt'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)