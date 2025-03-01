import numpy as np

# TODO handle frames with no predictions

def format_yolo_predictions(preds, frames_list, class_mapping = {0: 'car', 1: 'bike'}):
    """
    Formats YOLO predictions into a more readable structure.
    Args:
        preds (list): A list of prediction objects, where each prediction contains bounding boxes, confidence scores, and class indices.
        class_mapping (dict, optional): A dictionary mapping class indices to class names. Defaults to {0: 'car', 1: 'bike'}.
    Returns:
        list: A list of formatted predictions. Each formatted prediction is a dictionary of dictionaries, where each dictionary contains:
            - 'bbox' (numpy.ndarray): The bounding box coordinates.
            - 'class' (str): The class name.
            - 'conf' (float): The confidence score.
    """

    formatted_preds = []

    for pred, frame in zip(preds, frames_list):
        formatted_dets = {}

        for box in pred.boxes.cpu():

            for coords, conf, cls in zip(box.xyxy, box.conf, box.cls):
                det = {'bbox': coords.numpy(), 'class': class_mapping[int(cls.item())], 'conf': conf.item()}

            formatted_dets[frame] = det
        formatted_preds.append(formatted_dets)
    
    return formatted_preds
