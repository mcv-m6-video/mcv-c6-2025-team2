# Week 2: Object Detection - Tracking

In this directory, you can find the code used to perform the experiments in [Week 2](https://docs.google.com/presentation/d/1fEmYj3vuOOZ3HEOrYQzoxAS7faIJEvtebDRjRWDJ1Js). Further explanations on the experiments and the corresponding qualitative results can be found in [Week 2 report for Task 1](https://docs.google.com/presentation/d/17lopXVN5yTLmV9D4WNLUpHEstNSaVr50plXjcMj9LnA) and [Week 2 report for Task 2](https://docs.google.com/presentation/d/1xT7DNoxir8k8vc8JpcO5qdK7gY8q_jo5cAr7O9679fg).

## Contents

[▶️ Setup](#▶️-setup)  

[➡️ Task 1: Object Detection](#➡️-task-1-object-detection)  
- [Task 1.1: Off-the-shelf](#task-11-off-the-shelf)  
- [Task 1.2: Fine-tune to your data](#task-12-fine-tune-to-your-data)  
- [Task 1.3: K-Fold Cross-validation](#task-13-k-fold-cross-validation)  

[➡️ Task 2: Object Tracking](#➡️-task-2-object-tracking)  
- [Task 2.1: Tracking by Overlap](#task-21-tracking-by-overlap)  
- [Task 2.2: Tracking with a Kalman Filter](#task-22-tracking-with-a-kalman-filter)  
- [Task 2.3: IDF1, HOTA scores](#task-23-idf1-hota-scores)  

## ▶️ Setup

### Dataset creation
Create a directory `week2/data` and place the data files `vdo.avi` (video) and `ai_challenge_s03_c010-full_annotation.xml` (ground truth annotations) inside it.

Then you can run the code to generate the dataset for YOLO (make sure you are working in `week2` directory):

```
cd week2
python yolo/utils/create_yolo_dataset.py
```

This will create the dataset required for object detection in your `week2/data` directory.

### WandB logging
To log your trainings in WandB, run the following command in your CLI:

```
yolo settings wandb=True
```

### Training YOLOv11
To run training, set the desired data paths in a new or existing `.yaml` configuration file, found in `yolo/config`. Also set the desired parameters in the file `yolo/train_yolov11.py`. Then, you are ready to run training:

```
python yolo/train_yolov11.py
```

### Inference with YOLOv11
To run inference, you just need to adjust your custom parameters in the file `yolo/pred_yolov11.py`, and then run it:

```
python yolo/pred_yolov11.py
```

## ➡️ Task 1: Object Detection
TODO

### Task 1.1: Off-the-shelf
TODO

### Task 1.2: Fine-tune to your data
TODO

### Task 1.3: K-Fold Cross-validation
TODO

## ➡️ Task 2: Object Tracking
TODO

### Task 2.1: Tracking by Overlap
TODO

### Task 2.2: Tracking with a Kalman Filter
TODO

### Task 2.3: IDF1, HOTA scores
TODO
