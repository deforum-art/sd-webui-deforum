from torch.functional import Tensor
from general_utils import log
from collections import defaultdict
import numpy as np

import torch
from torch.nn import functional as nnf


class BaseMetric(object):

    def __init__(self, metric_names, pred_range=None, gt_index=0, pred_index=0, eval_intermediate=True,
                 eval_validation=True):
        self._names = tuple(metric_names)
        self._eval_intermediate = eval_intermediate
        self._eval_validation = eval_validation

        self._pred_range = pred_range
        self._pred_index = pred_index
        self._gt_index = gt_index

        self.predictions = []
        self.ground_truths = []

    def eval_intermediate(self):
        return self._eval_intermediate

    def eval_validation(self):
        return self._eval_validation

    def names(self):
        return self._names

    def add(self, predictions, ground_truth):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def scores(self):
        # similar to value but returns dict
        value = self.value()
        if type(value) == dict:
            return value
        else:
            assert type(value) in {list, tuple}
            return list(zip(self.names(), self.value()))

    def _get_pred_gt(self, predictions, ground_truth):
        pred = predictions[self._pred_index]
        gt = ground_truth[self._gt_index]

        if self._pred_range is not None:
            pred = pred[:, self._pred_range[0]: self._pred_range[1]]

        return pred, gt


class FixedIntervalMetrics(BaseMetric):

    def __init__(self, sigmoid=False, ignore_mask=False, resize_to=None, 
                 resize_pred=None, n_values=51, custom_threshold=None):


        super().__init__(('ap', 'best_fgiou', 'best_miou', 'fgiou0.5', 'fgiou0.1', 'mean_iou_0p5', 'mean_iou_0p1', 'best_biniou', 'biniou_0.5', 'fgiou_thresh'))
        self.intersections = []
        self.unions = []
        # self.threshold = threshold
        self.sigmoid = sigmoid
        self.resize_to = resize_to
        self.resize_pred = resize_pred  # resize prediction to match ground truth
        self.class_count = defaultdict(lambda: 0)
        self.per_class = defaultdict(lambda : [0,0])
        self.ignore_mask = ignore_mask
        self.custom_threshold = custom_threshold

        self.scores_ap = []
        self.scores_iou = []
        self.gts, self.preds = [], []
        self.classes = []

        # [1:-1] ignores 0 and 1
        self.threshold_values = np.linspace(0, 1, n_values)[1:-1]

        self.metrics = dict(tp=[], fp=[], fn=[], tn=[])

    def add(self, pred, gt):
        
        pred_batch = pred[0].cpu()

        if self.sigmoid:
            pred_batch = torch.sigmoid(pred_batch)

        gt_batch = gt[0].cpu()
        mask_batch = gt[1] if len(gt) > 1 and not self.ignore_mask and gt[1].numel() > 0 else ([None] * len(pred_batch))
        cls_batch = gt[2] if len(gt) > 2 else [None] * len(pred_batch)

        if self.resize_to is not None:
            gt_batch = nnf.interpolate(gt_batch, self.resize_to, mode='nearest')
            pred_batch = nnf.interpolate(pred_batch, self.resize_to, mode='bilinear', align_corners=False)
        
        if isinstance(cls_batch, torch.Tensor):
            cls_batch = cls_batch.cpu().numpy().tolist()

        assert len(gt_batch) == len(pred_batch) == len(cls_batch), f'{len(gt_batch)} {len(pred_batch)} {len(cls_batch)}'

        for predictions, ground_truth, mask, cls in zip(pred_batch, gt_batch, mask_batch, cls_batch):

            if self.resize_pred:
                predictions = nnf.interpolate(predictions.unsqueeze(0).float(), size=ground_truth.size()[-2:], mode='bilinear', align_corners=True)

            p = predictions.flatten()
            g = ground_truth.flatten()

            assert len(p) == len(g)

            if mask is not None:
                m = mask.flatten().bool()
                p = p[m]
                g = g[m]

            p_sorted = p.sort()
            p = p_sorted.values
            g = g[p_sorted.indices]

            tps, fps, fns, tns = [], [], [], []
            for thresh in self.threshold_values:

                valid = torch.where(p > thresh)[0]
                if len(valid) > 0:
                    n = int(valid[0])
                else:
                    n = len(g)

                fn = int(g[:n].sum())
                tp = int(g[n:].sum())
                fns += [fn]
                tns += [n - fn]
                tps += [tp]
                fps += [len(g) - n - tp]

            self.metrics['tp'] += [tps]
            self.metrics['fp'] += [fps]
            self.metrics['fn'] += [fns]
            self.metrics['tn'] += [tns]

            self.classes += [cls.item() if isinstance(cls, torch.Tensor) else cls]

    def value(self):

        import time
        t_start = time.time()   

        if set(self.classes) == set([None]):
            all_classes = None
            log.warning('classes were not provided, cannot compute mIoU')
        else:
            all_classes = set(int(c) for c in self.classes)
            # log.info(f'compute metrics for {len(all_classes)} classes')

        summed = {k: [sum([self.metrics[k][i][j] 
                           for i in range(len(self.metrics[k]))])
                      for j in range(len(self.threshold_values))]
                  for k in self.metrics.keys()}

        if all_classes is not None:

            assert len(self.classes) == len(self.metrics['tp']) == len(self.metrics['fn'])
            # group by class
            metrics_by_class = {c: {k: [] for k in self.metrics.keys()} for c in all_classes}
            for i in range(len(self.metrics['tp'])):
                for k in self.metrics.keys():
                    metrics_by_class[self.classes[i]][k] += [self.metrics[k][i]]
            
            # sum over all instances within the classes
            summed_by_cls = {k: {c: np.array(metrics_by_class[c][k]).sum(0).tolist() for c in all_classes} for k in self.metrics.keys()}


        # Compute average precision

        assert (np.array(summed['fp']) + np.array(summed['tp']) ).sum(), 'no predictions is made'

        # only consider values where a prediction is made
        precisions = [summed['tp'][j] / (1 + summed['tp'][j] + summed['fp'][j]) for j in range(len(self.threshold_values))
                      if summed['tp'][j] + summed['fp'][j] > 0]
        recalls = [summed['tp'][j] / (1 + summed['tp'][j] + summed['fn'][j]) for j in range(len(self.threshold_values))
                           if summed['tp'][j] + summed['fp'][j] > 0]

        # remove duplicate recall-precision-pairs (and sort by recall value)
        recalls, precisions = zip(*sorted(list(set(zip(recalls, precisions))), key=lambda x: x[0]))

        from scipy.integrate import simps
        ap = simps(precisions, recalls)

        # Compute best IoU
        fgiou_scores = [summed['tp'][j] / (1 + summed['tp'][j] + summed['fp'][j] + summed['fn'][j]) for j in range(len(self.threshold_values))]

        biniou_scores = [
            0.5*(summed['tp'][j] / (1 + summed['tp'][j] + summed['fp'][j] + summed['fn'][j])) + 
            0.5*(summed['tn'][j] / (1 + summed['tn'][j] + summed['fn'][j] + summed['fp'][j])) 
            for j in range(len(self.threshold_values))
        ]
        
        index_0p5 = self.threshold_values.tolist().index(0.5)
        index_0p1 = self.threshold_values.tolist().index(0.1)
        index_0p2 = self.threshold_values.tolist().index(0.2)
        index_0p3 = self.threshold_values.tolist().index(0.3)

        if self.custom_threshold is not None:
            index_ct = self.threshold_values.tolist().index(self.custom_threshold)

        if all_classes is not None:
            # mean IoU
            mean_ious = [np.mean([summed_by_cls['tp'][c][j] / (1 + summed_by_cls['tp'][c][j] + summed_by_cls['fp'][c][j] + summed_by_cls['fn'][c][j]) 
                            for c in all_classes])
                        for j in range(len(self.threshold_values))]

            mean_iou_dict = {
                'miou_best': max(mean_ious) if all_classes is not None else None,
                'miou_0.5': mean_ious[index_0p5] if all_classes is not None else None,
                'miou_0.1': mean_ious[index_0p1] if all_classes is not None else None,
                'miou_0.2': mean_ious[index_0p2] if all_classes is not None else None,
                'miou_0.3': mean_ious[index_0p3] if all_classes is not None else None,
                'miou_best_t': self.threshold_values[np.argmax(mean_ious)],
                'mean_iou_ct': mean_ious[index_ct] if all_classes is not None and self.custom_threshold is not None else None,
                'mean_iou_scores': mean_ious,
            }

        print(f'metric computation on {(len(all_classes) if all_classes is not None else "no")} classes took {time.time() - t_start:.1f}s')

        return {
            'ap': ap,

            # fgiou
            'fgiou_best': max(fgiou_scores),
            'fgiou_0.5': fgiou_scores[index_0p5],
            'fgiou_0.1': fgiou_scores[index_0p1],
            'fgiou_0.2': fgiou_scores[index_0p2],
            'fgiou_0.3': fgiou_scores[index_0p3],
            'fgiou_best_t': self.threshold_values[np.argmax(fgiou_scores)],

            # mean iou


            # biniou
            'biniou_best': max(biniou_scores),
            'biniou_0.5': biniou_scores[index_0p5],
            'biniou_0.1': biniou_scores[index_0p1],
            'biniou_0.2': biniou_scores[index_0p2],
            'biniou_0.3': biniou_scores[index_0p3],
            'biniou_best_t': self.threshold_values[np.argmax(biniou_scores)],

            # custom threshold
            'fgiou_ct': fgiou_scores[index_ct] if self.custom_threshold is not None else None,
            'biniou_ct': biniou_scores[index_ct] if self.custom_threshold is not None else None,
            'ct': self.custom_threshold,

            # statistics
            'fgiou_scores': fgiou_scores,
            'biniou_scores': biniou_scores,
            'precision_recall_curve': sorted(list(set(zip(recalls, precisions)))),
            'summed_statistics': summed,
            'summed_by_cls_statistics': summed_by_cls,

            **mean_iou_dict
        }

        # ('ap', 'best_fgiou', 'best_miou', 'fgiou0.5', 'fgiou0.1', 'mean_iou_0p5', 'mean_iou_0p1', 'best_biniou', 'biniou_0.5', 'fgiou_thresh'

        # return ap, best_fgiou, best_mean_iou, iou_0p5, iou_0p1, mean_iou_0p5, mean_iou_0p1, best_biniou, biniou0p5, best_fgiou_thresh, {'summed': summed, 'summed_by_cls': summed_by_cls}

