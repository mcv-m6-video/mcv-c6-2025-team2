import numpy as np
import time
import pickle
import cv2
from tqdm import tqdm

from sort import Sort


# Convert each frame list to a dictionary so we can easily access the name of the frame, e.g. '0001'
def reformat_detections(seq_dets, start_number=0):
    detections = {f"{i + start_number:04d}": seq_dets[i] for i in range(len(seq_dets))}
    return detections


def initialize_video_writer(first_frame_path, output_path=None, fps=30):
    """
    Initialize a video writer based on the dimensions of the first frame.
    
    Args:
        first_frame_path: Path to the first frame image
        output_path: Path where to save the output video
        fps: Frames per second for the output video
    
    Returns:
        video_writer: OpenCV VideoWriter object
        width, height: Dimensions of the video
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


def process_frame_detections(detections, tracker):
    """
    Process frame detections and update tracker.
    
    Args:
        detections: List of detections for current frame
        tracker: SORT tracker object
    
    Returns:
        tracked_objects: List of tracked objects with their IDs
        processing_time: Time taken for processing
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
    Draw tracked objects on the image with their IDs, using distinct colors.
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


def track_objects(detections_dict, image_folder, output_video_path=None, fps=30):
    """
    Main function for object tracking visualization.
    """
    first_frame_key = list(detections_dict.keys())[0]
    first_frame_path = f"{image_folder}/frame_{first_frame_key}.jpg"
    
    try:
        video_writer, _, _, _ = initialize_video_writer(first_frame_path, output_video_path, fps)
        mot_tracker = Sort()
        
        total_time = 0
        total_frames = 0
        object_colors = {}  # Dictionary to store consistent colors per object ID
        
        for frame_key, frame_detections in tqdm(detections_dict.items(), desc="Processing frames", unit="frame"):
            frame_path = f"{image_folder}/frame_{frame_key}.jpg"
            image = cv2.imread(frame_path)
            
            if image is None:
                print(f"Could not load image {frame_path}")
                continue
            
            tracked_objects, processing_time = process_frame_detections(frame_detections, mot_tracker)
            total_time += processing_time
            total_frames += 1
            
            result_image, object_colors = visualize_tracked_objects(image, tracked_objects, object_colors)
            video_writer.write(result_image)
        
        video_writer.release()
        
        print(f"Video saved at {output_video_path}")
        print(f"Total processing time: {total_time:.2f} seconds")
        if total_frames > 0:
            print(f"Average processing time per frame: {total_time/total_frames:.4f} seconds")
            print(f"Effective frame rate: {total_frames/total_time:.2f} FPS")
    
    except Exception as e:
        print(f"Error during tracking: {e}")
        try:
            video_writer.release()
        except:
            pass


if __name__ == "__main__":
    
    # Read the detections
    with open("week2/tracking/preds_pred_off-shelf.pkl", "rb") as f:
        seq_dets = pickle.load(f)
    
    detections = reformat_detections(seq_dets, start_number=535)

    # Get frames (they should be extracteded from the video and downloaded as .jpg)
    frames_dir = 'data/AICity_data/frames'

    # Call the tracking function
    track_objects(detections, frames_dir, "object_tracking.mp4", fps=10)
