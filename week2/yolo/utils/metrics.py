import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class Bbox:
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class Annotation:
    bbox: Bbox
    class_name: str

@dataclass
class Prediction:
    bbox: Bbox
    class_name: str
    confidence: Optional[float] = None

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
    gt = np.array(gt)  
    if gt.size == 0:
        return np.array([])

    ixmin = np.maximum(gt[:, 0], pred[0])
    iymin = np.maximum(gt[:, 1], pred[1])
    ixmax = np.minimum(gt[:, 2], pred[2])
    iymax = np.minimum(gt[:, 3], pred[3])

    iw = np.maximum(ixmax - ixmin, 0)
    ih = np.maximum(iymax - iymin, 0)
    inters = iw * ih

    pred_area = (pred[2] - pred[0]) * (pred[3] - pred[1])
    gt_areas = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    uni = pred_area + gt_areas - inters

    return inters / uni

def voc_eval(
    gt: Dict[str, List[Annotation]], 
    preds: Dict[str, List[Prediction]], 
    ovthresh: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth bounding boxes using PASCAL VOC metrics.
    Computes AP per frame instead of per class.
    """
    ap_per_frame = {}

    for image_id in preds.keys():
        # Get ground truth for the frame
        annotations = gt.get(image_id, [])
        bboxes_gt = np.array([[ann.bbox.x1, ann.bbox.y1, ann.bbox.x2, ann.bbox.y2] for ann in annotations])
        detected = [False] * len(annotations)
        npos = len(annotations)

        # Get predictions for the frame
        predictions = preds.get(image_id, [])
        confidence = np.array([pred.confidence for pred in predictions])
        bboxes_pred = np.array([[pred.bbox.x1, pred.bbox.y1, pred.bbox.x2, pred.bbox.y2] for pred in predictions])

        # Sort predictions by confidence
        if confidence.size > 0:
            sorted_ind = np.argsort(-confidence)
            bboxes_pred = bboxes_pred[sorted_ind]
            confidence = confidence[sorted_ind]

        # Evaluate detections
        nd = len(predictions)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            bb = bboxes_pred[d]
            if bboxes_gt.size > 0:
                overlaps = voc_iou(bb, bboxes_gt)
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not detected[jmax]:
                        tp[d] = 1.0
                        detected[jmax] = True
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

        # Compute AP for this frame
        ap_per_frame[image_id] = voc_ap(rec, prec)

    return ap_per_frame


def compute_mAP(
    gt: Dict[str, List[Annotation]],
    preds: Dict[str, List[Prediction]],
    iou_threshold: float = 0.5,
    aggregated: bool = True
) -> float | Dict[str, float]:
    """
    Compute mean Average Precision (mAP) considering confidence scores.
    """
    ap_per_frame = voc_eval(gt, preds, iou_threshold)
    if aggregated:
        return np.mean(list(ap_per_frame.values()))
    return ap_per_frame
