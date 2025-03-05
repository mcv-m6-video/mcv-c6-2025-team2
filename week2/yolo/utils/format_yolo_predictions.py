import os
import numpy as np
import cv2

def format_yolo_predictions(preds, frames_list, class_mapping = {0: 'car', 1: 'bike'}):
    """
    Formats YOLO predictions into a more readable structure.
    Args:
        preds (list): A list of prediction objects, where each prediction contains bounding boxes, confidence scores, and class indices.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to {0: 'car', 1: 'bike'}.
    Returns:
        list: A list of formatted predictions. Each formatted prediction is a dictionary where the keys are frame identifiers and the values are dictionaries containing:
            - 'bbox' (numpy.ndarray): The bounding box coordinates.
            - 'class' (str): The class name.
            - 'conf' (float): The confidence score.
    """

    formatted_preds = []

    for pred, frame in zip(preds, frames_list):
        formatted_dets = {}
        frame_dets = []

        for box in pred.boxes.cpu():
            for coords, conf, cls in zip(box.xyxy, box.conf, box.cls):
                if cls.item() in class_mapping:
                    det = {'bbox': coords.numpy(), 'class': class_mapping[int(cls.item())], 'conf': conf.item()}
                    frame_dets.append(det)

        formatted_dets[frame] = frame_dets
        formatted_preds.append(formatted_dets)
    
    return formatted_preds

def format_yolo_annotations(annotations_dir, frames_list, class_mapping={0: 'car', 1: 'bike'}, images_dir=None):
    """
    Reads YOLO annotation files and formats them into a structured dictionary.
    
    Args:
        annotations_dir (str): Path to the directory containing YOLO annotation .txt files.
        frames_list (list): List of frame identifiers (e.g., filenames without extensions).
        class_mapping (dict, optional): A dictionary mapping class indices to class names.
        images_dir (str, optional): Path to the directory containing the corresponding images.
    
    Returns:
        list: A list of formatted predictions where each entry corresponds to a frame.
    """
    formatted_preds = []
    
    # Get image dimensions from the first frame
    if images_dir and frames_list:
        first_image_path = os.path.join(images_dir, f"{frames_list[0]}")  # Assuming images are .jpg
        if os.path.exists(first_image_path):
            image = cv2.imread(first_image_path)
            image_height, image_width, _ = image.shape
        else:
            raise FileNotFoundError(f"Image {first_image_path} not found.")
    else:
        raise ValueError("Image directory must be provided and contain at least one frame.")
    
    for frame in frames_list:
        frame_name = frame[:-4]
        annotation_file = os.path.join(annotations_dir, f"{frame_name}.txt")
        formatted_dets = {}
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
            
            detections = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip malformed lines
                
                cls, x_center, y_center, width, height = map(float, parts)
                
                # Convert normalized xywh to absolute xyxy
                x1 = (x_center - width / 2) * image_width
                y1 = (y_center - height / 2) * image_height
                x2 = (x_center + width / 2) * image_width
                y2 = (y_center + height / 2) * image_height
                
                det = {
                    'bbox': np.array([x1, y1, x2, y2]),
                    'class': class_mapping.get(int(cls), 'unknown')
                }
                detections.append(det)
                
            formatted_dets[frame] = detections
        
        formatted_preds.append(formatted_dets)
    
    return formatted_preds
