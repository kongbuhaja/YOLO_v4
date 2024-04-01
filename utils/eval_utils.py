import numpy as np
from utils.bbox_utils import bbox_iou_np

class Eval:
    def __init__(self, cfg, eps=1e-6):
        self.labels=cfg['data']['labels']['name']
        self.eval_per_epoch = cfg['train']['eval_per_epoch']
        self.result_dir = cfg['eval']['dir']
        self.warmup_epochs = cfg['train']['lr_scheduler']['warmup_epochs']
        self.eps = eps
        self.num_of_ious = 10
        self.iou_thresholds = 0.5 + (0.5) / self.num_of_ious * np.arange(self.num_of_ious)
        self.stats = []
        self.ap = np.zeros((len(self.labels), self.num_of_ious))

    def check(self, epoch):
        return epoch >= self.warmup_epochs and epoch % self.eval_per_epoch == 0

    def init_stat(self):
        del self.stats
        self.stats = []

    def update(self, labels, preds):
        correct = np.zeros([preds.shape[0], self.num_of_ious])    

        if preds.shape[0] == 0:
            if labels.shape[0]:
                self.stats += [[correct, np.zeros(0), np.zeros(0), labels[:, 4]]]

        elif labels.shape[0]:
            detect = []
            for c in np.unique(labels[:, 4]):
                ti = (c == labels[:, 4]).nonzero()[0]
                pi = (c == preds[:, 5]).nonzero()[0]
                
                if pi.shape[0]:
                    ious = bbox_iou_np(preds[pi, :4][:, None], labels[ti, :4][None])
                    best_id = np.argmax(ious, -1)
                    best_iou = np.max(ious, -1)

                    detect_set = set()
                    for j in (ious > self.iou_thresholds[0]).nonzero()[0]:
                        det = ti[best_id[j]]
                        if det not in detect_set:
                            detect_set.add(det)
                            detect += [det]
                            correct[pi[j]] = best_iou[j] > self.iou_thresholds
                            if len(detect) == labels.shape[0]:
                                break
            self.stats += [[correct, preds[:, 4], preds[:, 5], labels[:, 4]]]

    def compute_mAP(self):
        if len(self.stats) == 0:
            return 0., 0.

        tp, conf, pred_cls, gt_cls = [np.concatenate(x, 0) for x in zip(*self.stats)]
        idx = np.argsort(-conf)
        tp, conf, pred_cls = tp[idx], conf[idx], pred_cls[idx]

        for c in np.unique(gt_cls.astype(np.int32)):
            i = pred_cls == c
            n_g = (gt_cls == c).sum()
            n_p = i.sum()

            if n_p == 0 or n_g == 0:
                continue
            
            tpc = tp[i].cumsum(0)
            fpc = (1 - tp[i]).cumsum(0)

            recall = tpc / (n_g + self.eps)

            precision = tpc / (tpc + fpc)
 
            for j in range(self.num_of_ious):
                self.ap[c, j] = self.compute_ap(recall[:, j], precision[:, j])
        
        return np.mean(self.ap[:, 0]), np.mean(self.ap)

    def compute_ap(self, recall, precision):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([1.], precision, [0.]))

        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)

        return ap

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