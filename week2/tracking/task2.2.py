import numpy as np
import time
import pickle
import cv2
from tqdm import tqdm
from sort import Sort
import xml.etree.ElementTree as ET

import sys
sys.path.append("week2/") 
from metrics import create_data_for_hota, parse_bboxes_from_xml
from tracking.TrackEval.hota import HOTA


def reformat_detections(seq_dets, start_number=0):
    """
    Convert each frame list to a dictionary so we can easily access the name of the frame.
    """
    detections = {f"{i + start_number:04d}": seq_dets[i] for i in range(len(seq_dets))}
    return detections


def initialize_video_writer(first_frame_path, output_path=None, fps=30):
    """
    Initialize a video writer based on the dimensions of the first frame.
    """
    # Read first frame to get dimensions
    image = cv2.imread(first_frame_path)
    
    if image is None:
        raise ValueError(f"Could not load image {first_frame_path}")
    
    # Get dimensions
    height, width, _ = image.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    if output_path is None:
        output_path = "tracking_results.mp4"
        
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return video_writer, width, height, image


def track_objects_in_frame(detections, tracker):
    """
    Process frame detections and update tracker.
    """
    # Initialize the detection list for this frame
    dets = []

    # Add the frame detections to the `dets` list
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['conf']
        dets.append([x1, y1, x2, y2, confidence])

    dets = np.array(dets)
    
    # If no detections, provide empty array with correct shape
    if len(dets) == 0:
        dets = np.empty((0, 5))

    # Update trackers with detections
    start_time = time.time()
    tracked_objects = tracker.update(dets)
    processing_time = time.time() - start_time
    
    return tracked_objects, processing_time


def visualize_tracked_objects(image, tracked_objects, object_colors):
    """
    Draw tracked objects on the image with their IDs, using distinct colors for each object.
    """
    # Create a copy of the image to avoid modifying the original
    result_image = image.copy()
    
    for d in tracked_objects:
        x1, y1, x2, y2, obj_id = d  # Unpack coordinates and object ID
        obj_id = int(obj_id)
        
        # Assign a unique color if the object ID is new
        if obj_id not in object_colors:
            np.random.seed(obj_id)  # Seed with object ID to ensure consistency
            object_colors[obj_id] = tuple(map(int, np.random.randint(0, 255, size=3)))
        
        color = object_colors[obj_id]
        
        # Draw the detection rectangle
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Add text with the object ID
        cv2.putText(
            result_image, 
            f"ID:{obj_id}", 
            (int(x1), int(y1) - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
    
    return result_image, object_colors


def get_tracking(detections_dict: dict, image_folder: str, 
                 generate_video: bool=False, fps: int=30, output_video_path:str="tracking_results.mp4"):
    """
    Process the detections to get tracking statistics and optionally create a video visualization.
    
    Args:
        detections_dict: Dictionary mapping frame IDs to detection lists
        image_folder: Folder containing the frame images
        generate_video: Boolean indicating whether to generate a video visualization
        fps: Frames per second for the output video
        output_video_path: Path where to save the output video (if generate_video is True)
    
    Returns:
        dict: Dictionary containing tracking statistics including:
            - num_tracker_dets: Total number of tracker detections
            - num_tracker_ids: Total number of unique tracker IDs
            - tracker_ids: List of lists containing tracker IDs for each frame
    """

    mot_tracker = Sort()
    
    # Processing statistics
    total_time = 0
    total_frames = 0
    all_tracker_ids = set()  # All unique tracker IDs across all frames
    tracker_ids_by_frame = []  # IDs in each frame
    all_tracker_detections = 0  # Total detection count
    
    if generate_video:
        first_frame_key = list(detections_dict.keys())[0]
        first_frame_path = f"{image_folder}/frame_{first_frame_key}.jpg"
        video_writer, _, _, _ = initialize_video_writer(first_frame_path, output_video_path, fps)
        object_colors = {}  # Dictionary to store consistent colors per object ID
    

    for frame_key, frame_detections in tqdm(detections_dict.items(), desc="Processing frames", unit="frame"):
        # Process detections and update tracker
        tracked_objects, processing_time = track_objects_in_frame(frame_detections, mot_tracker)
        
        # Update statistics
        total_time += processing_time
        total_frames += 1
        
        # Extract IDs from this frame's tracked objects
        frame_ids = np.array([int(obj[4]) for obj in tracked_objects])
        tracker_ids_by_frame.append(frame_ids)
        
        # Update unique IDs and detection count
        all_tracker_ids.update(frame_ids)
        all_tracker_detections += len(tracked_objects)
        
        if generate_video:
            # Load the corresponding image for the frame
            frame_path = f"{image_folder}/frame_{frame_key}.jpg"
            image = cv2.imread(frame_path)
            
            if image is None:
                print(f"Could not load image {frame_path}")
                continue
            
            # Visualize tracking results
            result_image, object_colors = visualize_tracked_objects(image, tracked_objects, object_colors)
            
            # Write the processed frame to the video
            video_writer.write(result_image)
    
    if generate_video:
        video_writer.release()
        print(f"Tracking visualization saved at {output_video_path}")
    
    # Compile statistics dictionary
    stats = {
        "num_tracker_dets": all_tracker_detections,
        "num_tracker_ids": len(all_tracker_ids),
        "tracker_ids": tracker_ids_by_frame,
    }
    
    return stats


if __name__ == "__main__":
    
    # Read the detections
    with open("week2/tracking/preds_pred_off-shelf.pkl", "rb") as f:
        seq_dets = pickle.load(f)
    
    # Prepare data for tracker
    detections = reformat_detections(seq_dets, start_number=535)
    frames_dir = 'data/AICity_data/frames'

    # Process tracking and optionally create visualization
    tracking_stats = get_tracking(detections, frames_dir, 
                                  generate_video=True, fps=10, output_video_path="object_tracking.mp4")
    
    # Print statistics
    if tracking_stats:
        print("\n=== Tracking Statistics ===")
        print(f"Total tracker detections: {tracking_stats['num_tracker_dets']}")
        print(f"Total unique tracker IDs: {tracking_stats['num_tracker_ids']}")

    # Evaluation - HOTA
    # Bboxes detected and GT
    detected_bboxes = [[d['bbox'] for d in sublist] for sublist in seq_dets]
    file_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C6. Video Analysis/mcv-c6-2025-team2/data/ai_challenge_s03_c010-full_annotation.xml'
    gt_bboxes = parse_bboxes_from_xml(file_path)

    data = create_data_for_hota(gt_bboxes, detected_bboxes, tracking_stats)
    
    # Crear instancia de HOTA
    hota_metric = HOTA()

    # Calcular métricas para una secuencia
    results = hota_metric.eval_sequence(data)

    # Mostrar los resultados
    print(results)