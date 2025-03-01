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

def voc_eval(
    gt: Dict[str, List[Annotation]], 
    preds: Dict[str, List[Prediction]], 
    ovthresh: float = 0.5,
    use_confidence: bool = True
) -> float:
    """
    Evaluate predictions against ground truth bounding boxes using PASCAL VOC metrics.
    """
    class_recs = {}
    npos = 0

    # Convert ground truth annotations to a structured dictionary
    for image_id, annotations in gt.items():
        bboxes = np.array([[ann.bbox.x1, ann.bbox.y1, ann.bbox.x2, ann.bbox.y2] for ann in annotations])
        class_recs[image_id] = {"bbox": bboxes, "det": [False] * len(annotations)}
        npos += len(annotations)

    image_ids = []
    confidence = []
    BB = []

    # Extract predictions
    for image_id, predictions in preds.items():
        for pred in predictions:
            image_ids.append(image_id)
            confidence.append(pred.confidence if use_confidence else np.random.rand())
            BB.append([pred.bbox.x1, pred.bbox.y1, pred.bbox.x2, pred.bbox.y2])

    confidence = np.array(confidence)
    BB = np.array(BB)

    # Sort predictions by confidence
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
        image_id = image_ids[d]
        R = class_recs.get(image_id, {"bbox": np.array([]), "det": []})
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
    # iou_mean = np.mean(iou_scores)

    return ap

def compute_mAP_no_confidence(
    gt: Dict[str, List[Annotation]],
    preds: Dict[str, List[Prediction]],
    iou_threshold: float = 0.5,
    iterations: int = 10
) -> float:
    """
    Compute mean Average Precision (mAP) without considering confidence scores.
    """
    # Compute mAP
    aps = [
        voc_eval(gt, preds, iou_threshold, use_confidence = False)
        for _ in range(iterations)
    ]
    return np.mean(aps)


def compute_mAP_with_confidence(
    gt: Dict[str, List[Annotation]],
    preds: Dict[str, List[Prediction]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Compute mean Average Precision (mAP) considering confidence scores.
    """
    # Compute mAP
    return voc_eval(gt, preds, iou_threshold, use_confidence = True)

        