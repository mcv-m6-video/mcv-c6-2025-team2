from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")

# Validate the model
# https://docs.ultralytics.com/modes/val/#arguments-for-yolo-model-validation
metrics = model.val(
    data = 'config/task1.1.yaml',
    imgsz = 640,
    #batch = -1,  # Autobatch
    save_json = False,
    save_hybrid = False,
    conf = 0.001,  # Minimum confidence threshold for detections
    iou = 0.6,  # IoU threshold for NMS
    max_det = 300,
    half = True,
    device = 'cuda:0',
    dnn = False,
    plots = True,
    rect = True,
    split = 'val',
    project = 'task1.1_output',
    name = 'val_off-shelf'
)

print('mAP50-95:', metrics.box.map)
print('mAP50:', metrics.box.map50)
print('mAP75:', metrics.box.map75)

# TODO filter classes 1 (bycicle) and 2 (car) in COCO dataset
print('mAP50-95 for each category:', metrics.box.maps)