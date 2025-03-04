import pickle
import os
from typing import Dict, List
from utils.metrics import Annotation, Prediction, Bbox, compute_mAP

def correct_gt_pickle(gt: List[Dict]) -> Dict[str, List[Annotation]]:
    """
    Convert ground truth data from a list of dictionaries to a dictionary of lists.
    """
    gt_dict = {}
    for sample in gt:
        image_id = list(sample.keys())[0]
        for ann in sample[image_id]:
            gt_dict[image_id] = (
                gt_dict.get(image_id, []) +
                [Annotation(bbox=Bbox(*ann['bbox']), class_name=ann['class'])]
            )
    return gt_dict

def correct_preds_pickle(preds: List[Dict]) -> Dict[str, List[Prediction]]:
    """
    Convert predictions data from a list of dictionaries to a dictionary of lists.
    """
    preds_dict = {}
    for sample in preds:
        image_id = list(sample.keys())[0]
        for pred in sample[image_id]:
            preds_dict[image_id] = (
                preds_dict.get(image_id, []) +
                [Prediction(bbox=Bbox(*pred['bbox']), class_name=pred['class'], confidence=pred['conf'])]
            )
    return preds_dict

def read_pickle(file_path: str) -> List[Dict]:
    with open(file_path, 'rb') as f:
        data: List[Dict] = pickle.load(f)
    return data

if __name__ == '__main__':
    DATA_PATH = '/home/alex/Downloads/'
    gt_file = os.path.join(DATA_PATH, 'gt.pkl')
    preds_file = os.path.join(DATA_PATH, 'preds_pred_off-shelf_truck.pkl')

    gt = read_pickle(gt_file)
    gt = correct_gt_pickle(gt)

    preds = read_pickle(preds_file)
    preds = correct_preds_pickle(preds)

    first_image = list(preds.keys())[0]
    mAP = compute_mAP(gt, preds)

    print(mAP)


