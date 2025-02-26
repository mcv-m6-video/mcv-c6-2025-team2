import numpy as np
from typing import List

def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Compute VOC AP given precision and recall using the VOC 07 11-point method.
    """
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = np.max(prec[rec >= t]) if np.sum(rec >= t) > 0 else 0
        ap += p / 11.0
    return ap

def voc_iou(pred: List[int], gt: List[List[int]]) -> np.ndarray:
    """
    Compute IoU between a predicted bounding box and multiple ground truth boxes.
    """
    gt = np.array(gt)  # Convert ground truth to numpy array
    if gt.size == 0:
        return np.array([])

    ixmin = np.maximum(gt[:, 0], pred[0])
    iymin = np.maximum(gt[:, 1], pred[1])
    ixmax = np.minimum(gt[:, 2], pred[2])
    iymax = np.minimum(gt[:, 3], pred[3])

    iw = np.maximum(ixmax - ixmin, 0)
    ih = np.maximum(iymax - iymin, 0)
    inters = iw * ih

    # Compute union
    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    gt_areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    uni = pred_area + gt_areas - inters

    return inters / uni  # IoU values for each ground truth box

def voc_eval(preds: List[List[List[int]]], gt: List[List[List[int]]], ovthresh: float = 0.5):
    """
    Evaluate predictions against ground truth bounding boxes using PASCAL VOC metrics.
    """
    class_recs = {}
    npos = 0

    # Convert GT to dictionary
    for i, frame in enumerate(gt):
        class_recs[i] = {
            "bbox": np.array(frame),
            "det": [False] * len(frame)
        }
        npos += len(frame)

    image_ids = []
    confidence = []
    BB = []

    for i, frame in enumerate(preds):
        image_ids += [i] * len(frame)
        confidence += list(np.random.rand(len(frame)))  # Generate random confidence scores
        BB += frame  # Append bounding boxes

    confidence = np.array(confidence)
    BB = np.array(BB)

    if confidence.size > 0:
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind]
        image_ids = [image_ids[i] for i in sorted_ind]

    # Evaluate detections
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    iou_scores = np.zeros(nd)

    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d]
        BBGT = R["bbox"]

        if BBGT.size > 0:
            overlaps = voc_iou(bb, BBGT)
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
            iou_scores[d] = ovmax

            if ovmax > ovthresh:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    # Compute precision and recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec)
    iou_mean = np.mean(iou_scores)

    return ap, rec, prec, iou_mean

def mAP(gt: List[List[List[int]]], preds: List[List[List[int]]], N: int = 10):
    """
    Compute mean Average Precision (mAP) by averaging over multiple runs.
    """
    aps, recs, precs, ious = [], [], [], []
    for _ in range(N):
        ap, rec, prec, iou = voc_eval(preds, gt)
        aps.append(ap)
        recs.append(rec)
        precs.append(prec)
        ious.append(iou)

    return np.mean(aps), np.mean(recs), np.mean(precs), np.mean(ious)
