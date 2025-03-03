import os
import pickle
from natsort import natsorted

from ultralytics import YOLO
from utils.format_yolo_predictions import format_yolo_predictions

# Load a model
model = YOLO("yolo11m.pt")

# Path to data
data_path = "/home/mmajo/MCV/C6/week2/data/yolo_splits/yolo_A/images/val"
project = 'task1.1_output'
name = 'pred_off-shelf'

# Run inference with the model
# https://docs.ultralytics.com/modes/predict/#inference-arguments
preds = model.predict(
    source = data_path,
    conf = 0.25,  # Minimum confidence threshold for detections
    iou = 0.6,  # IoU threshold for NMS
    imgsz = 640,
    half = True,
    device = 'cuda:1',
    batch = 16,
    max_det = 300,
    visualize = False,
    augment = False,
    agnostic_nms = False,
    classes = [1,2],  # Labels to detect on COCO
    embed = False,
    project = project,
    name = name,
    save = True
)

frames_names = natsorted(os.listdir(data_path))
preds = format_yolo_predictions(preds, frames_names, class_mapping = {1: 'bike', 2: 'car'})

# Save to pkl file
save_dir = os.path.join(project, name)
with open(os.path.join(save_dir, f'preds_{name}.pkl'), 'wb') as f:
    pickle.dump(preds, f)
