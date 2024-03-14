import numpy as np
from utils.bbox_utils import bbox_iou

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

    def update_stats(self, pred, gt):
        if pred.shape[0] == 0:
            if gt.shape[0] == 0:
                self.stats += [[np.zeros((0, self.num_of_ious), dtype=bool), [], [], gt[..., 4]]]
            return

        correct = np.zeros((pred.shape[0], self.num_of_ious), dtype=bool)
        if gt.shape[0]:
            detected = []

            for gt_unique_class in np.unique(gt[..., 4]):
                ti = (gt_unique_class == gt[..., 4]).nonzero()[0]
                pi = (gt_unique_class == pred[..., 5]).nonzero()[0]
                if pi.shape[0]:
                    ious = bbox_iou(pred[pi, :4][:, None], gt[ti, :4][None]).numpy()
                    best_id = np.argmax(ious, -1)
                    best_iou = np.max(ious, -1)
                    detected_set = set()
                    for j in (best_iou > self.iou_thresholds[0]).nonzero()[0]:
                        det = ti[best_id[j]]
                        if det not in detected_set:
                            detected_set.add(det)
                            detected += [det]
                            correct[pi[j]] = best_iou[j] > self.iou_thresholds
                            if len(detected) == gt.shape[0]:
                                break
    
        self.stats += [[correct, pred[..., 4], pred[..., 5], gt[..., 4]]]

    def calculate_mAP(self):
        if len(self.stats) == 0:
            return 0., 0.
        tp, conf, pred_classes, gt_classes = [np.concatenate(x, 0) for x in zip(*self.stats)]
        idx = np.argsort(-conf)
        tp, conf, pred_classes = tp[idx], conf[idx], pred_classes[idx]

        gt_unique_classes = np.unique(gt_classes.astype(np.int32))

        ap = np.zeros((len(self.labels), self.num_of_ious))
        for c in gt_unique_classes:
            i_p = pred_classes == c
            n_g = (gt_classes == c).sum()
            n_p = (i_p).sum()

            if n_p == 0 or n_g == 0:
                continue
            
            tpc = tp[i_p].cumsum(0)
            fpc = (1 - tp[i_p]).cumsum(0)

            recall = tpc / (n_g + self.eps)
            precision = tpc / (tpc + fpc)
 
            for j in range(self.num_of_ious):
                ap[c, j] = self.compute_ap(recall[:, j], precision[:, j])
            
        self.ap = ap
        
        return np.mean(ap[:, 0]), np.mean(ap)

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