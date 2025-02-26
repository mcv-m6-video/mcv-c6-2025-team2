import cv2
import numpy as np
from src.metrics import compute_iou


def remove_noise(video_mask_path, denoised_video_mask_path='denoised_mask.avi', kernel_size=3):
    """
    Removes noise from binary video masks using morphological operations.
    
    Parameters:
    - video_mask_path: Path to the input video mask.
    - kernel_size: Size of the kernel used for morphological operations.
    
    Returns:
    - Path to the denoised video mask.
    """
    cap = cv2.VideoCapture(video_mask_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(denoised_video_mask_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))), False)
    
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        denoised = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        out.write(denoised)
    
    cap.release()
    out.release()
    print(f'Denoised mask video generated. Output saved at: {denoised_video_mask_path}')


def get_bounding_boxes(video_mask_path):
    """
    Extracts bounding boxes from video masks.
    
    Parameters:
    - video_mask_path: Path to the input video mask.
    
    Returns:
    - A list of frames, each containing a list of bounding boxes.
    - Each bounding box is represented as an array with [x-top-left, y-top-left, x-bottom-right, y-bottom-right].
    """
    cap = cv2.VideoCapture(video_mask_path)
    all_bboxes = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gray, connectivity=8)
        
        bboxes = []
        for i in range(1, num_labels):  # Skip the background label (0)
            xtl, ytl, xbr, ybr, _ = stats[i]
            bboxes.append([xtl, ytl, xbr, ybr])
        
        all_bboxes.append(bboxes)
    
    cap.release()
    return all_bboxes


def is_inside(inner_box, outer_box):
    """
    Checks if inner_box is completely inside outer_box.
    
    Parameters:
    - inner_box, outer_box: Lists or arrays representing bounding boxes [x1, y1, x2, y2].
    
    Returns:
    - True if inner_box is inside outer_box, False otherwise.
    """
    x1, y1, x2, y2 = inner_box
    x1_o, y1_o, x2_o, y2_o = outer_box

    return x1 >= x1_o and y1 >= y1_o and x2 <= x2_o and y2 <= y2_o


def compute_area(bbox):
    """
    Computes the area of a bounding box.
    
    Parameters:
    - bbox: A list representing a bounding box in the format [x1, y1, x2, y2].
    
    Returns:
    - The area of the bounding box.
    """
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1) * (y2 - y1))


def non_max_suppression(bboxes_per_frame, iou_threshold=0.5, min_area=500):
    """
    Applies Non-Maximum Suppression (NMS) to a list of bounding boxes per frame.
    Removes bounding boxes with high IoU, those completely inside another, and those below a minimum area.
    
    Parameters:
    - bboxes_per_frame: List of lists where each list contains bounding boxes for a frame.
      Each bounding box is represented as an array [x1, y1, x2, y2].
    - iou_threshold: IoU threshold to determine whether two boxes overlap too much.
    - min_area: Minimum area for a bounding box to be considered valid.
    
    Returns:
    - A list of lists where each sublist contains bounding boxes after applying NMS for each frame.
    """
    
    nms_bboxes_per_frame = []
    
    for bboxes in bboxes_per_frame:
        # Filter out bounding boxes that are too small
        bboxes = [bbox for bbox in bboxes if compute_area(bbox) >= min_area]

        # Sort bounding boxes by the top-left corner's y value (you can change this sorting method if needed)
        bboxes = sorted(bboxes, key=lambda bbox: (bbox[1], bbox[0]))  # Sort by y (and x as a tiebreaker)
        
        selected_bboxes = []
        
        while bboxes:
            # Take the bounding box with the highest score (first in sorted list)
            current_bbox = bboxes.pop(0)
            selected_bboxes.append(current_bbox)
            
            # Filter out boxes that have high IoU or are completely inside another box
            bboxes = [
                bbox for bbox in bboxes 
                if compute_iou(current_bbox, bbox) < iou_threshold and not is_inside(bbox, current_bbox)
            ]
        
        nms_bboxes_per_frame.append(selected_bboxes)
    
    return nms_bboxes_per_frame


def generate_bounding_box_video(predictions, video_path, output_video_path='output_with_boxes.avi', color=(0, 255, 0)):
    """
    Generates a video with bounding boxes drawn on each frame.

    Parameters:
    - predictions: List of lists containing bounding boxes for each frame.
    - video_path: Path to the input video.
    - output_video_path: Path to save the output video with bounding boxes.
    - color: Tuple representing the bounding box color in BGR format.
    - fps: Frames per second of the output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < len(predictions) and predictions[frame_idx]:  # Check if there are detections for this frame
            for bbox in predictions[frame_idx]:
                if len(bbox) == 4:  # Ensure bbox has the correct structure
                    xtl, ytl, xbr, ybr = map(int, bbox)  # Convert to integers
                    cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    print(f'Bounding box video generated. Output saved at: {output_video_path}')