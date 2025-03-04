import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.metrics import Annotation, Prediction, compute_mAP
from evaluate import read_pickle, correct_gt_pickle, correct_preds_pickle
from typing import List, Dict, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

FRAMES_FOLDER = '/home/alex/Documents/MCV/C6/mcv-c6-2025-team2/week2/data/frames'
DATA_PATH = '/home/alex/Downloads/'
IMAGE_SIZE = (480, 270)

# Function to process a single image
def process_image(image_id: str, gt: Dict[str, List[Annotation]], preds: Dict[str, List[Prediction]]) -> Tuple[str, Image.Image]:
    """Loads an image, draws bounding boxes, resizes it, and converts to PIL format."""
    image_path = os.path.join(FRAMES_FOLDER, image_id)
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        return image_id, None  # Skip missing images

    # Convert BGR to RGB once
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw ground truth boxes (Green)
    for ann in gt.get(image_id, []):
        x1, y1, x2, y2 = map(int, [ann.bbox.x1, ann.bbox.y1, ann.bbox.x2, ann.bbox.y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw prediction boxes (Red)
    for pred in preds.get(image_id, []):
        x1, y1, x2, y2 = map(int, [pred.bbox.x1, pred.bbox.y1, pred.bbox.x2, pred.bbox.y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Resize the image
    resized_image = cv2.resize(image, IMAGE_SIZE)

    return image_id, Image.fromarray(resized_image)  # Convert to PIL

def video_gt_preds(gt: Dict[str, List[Annotation]], preds: Dict[str, List[Prediction]]) -> List[Image.Image]:
    """
    Fully parallelized function to process video frames with ground truth and predictions.
    """
    image_ids = list(gt.keys())  # Get list of image IDs

    # Use ThreadPoolExecutor to process images in parallel
    print('Generating video frames...')
    with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust based on CPU
        results = executor.map(lambda img_id: process_image(img_id, gt, preds), image_ids)

    # Collect valid results
    return [img for _, img in results if img is not None]


def save_to_gif(frames: List[Image.Image], filename: str, sampling: int = 5):
    """
    Saves a list of PIL images as a GIF file.
    """
    print('Saving GIF...')
    frames[0].save(filename, save_all=True, append_images=frames[sampling:-1:sampling], duration=50, loop=0)



def plot_moving_average_video(mAP_dict: Dict[str, float], window_size: int = 10, output_filename="mAP_evolution.mp4"):
    """
    Generates a video showing the evolution of the moving average plot over frames.
    
    Args:
        mAP_dict: Dictionary where keys are frame_ids and values are mAP scores.
        window_size: Size of the moving average window (default: 10).
        output_filename: Output video file name (default: 'mAP_evolution.mp4').
    """
    # Sort frame_ids based on numeric value extracted from the filename
    sorted_frame_ids = sorted(mAP_dict.keys(), key=lambda x: int(x.split('.')[0].split('_')[-1]))  
    
    # Extract numeric frame indices for labeling
    frame_numbers = [int(fid.split('.')[0].split('_')[-1]) for fid in sorted_frame_ids]
    
    # Extract mAP values in sorted order
    mAP_values = [mAP_dict[frame_id] for frame_id in sorted_frame_ids]

    # Compute moving average using Pandas
    mAP_series = pd.Series(mAP_values)
    moving_avg = mAP_series.rolling(window=window_size, min_periods=1).mean()

    # Custom colors for scatter and lines
    dot_color_1 = '#6fbeae'
    line_color_1 = '#274857'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, max(frame_numbers))
    ax.set_ylim(0, 1)  # Adjust as per your mAP values range
    ax.set_xlabel("Frame")
    ax.set_ylabel("mAP")
    ax.set_title("mAP Evolution Over Time (Moving Average)")

    # Initialize scatter plot and line for both mAP and moving average
    sc1, = ax.plot([], [], marker='o', linestyle='', color=dot_color_1, markersize=1.3, alpha=0.4)
    line1, = ax.plot([], [], linestyle='-', color=line_color_1, linewidth=3, alpha=0.8)

    # Update function for animation
    def update(frame):
        # Data for moving average and original mAP values
        x_data = frame_numbers[:frame + 1]
        y_data_1 = mAP_values[:frame + 1]
        y_data_2 = moving_avg[:frame + 1]
        
        # Update scatter and line for mAP and moving average
        sc1.set_data(x_data, y_data_1)
        line1.set_data(x_data, y_data_2)

        return sc1, line1

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(frame_numbers), interval=50, blit=True)

    # Save as video (using ffmpeg)
    ani.save(output_filename, writer="ffmpeg", fps=10)

    # Close the plot
    plt.close(fig)


if __name__ == '__main__':
    gt_file = os.path.join(DATA_PATH, 'gt.pkl')
    preds_file = os.path.join(DATA_PATH, 'preds_pred_off-shelf_truck.pkl')

    gt = read_pickle(gt_file)
    gt = correct_gt_pickle(gt)

    preds = read_pickle(preds_file)
    preds = correct_preds_pickle(preds)

    mAPs = compute_mAP(gt, preds, aggregated=False)
    frames = plot_moving_average_video(mAPs, window_size=10, output_filename='plot.mp4')

    # video_frames = video_gt_preds(gt, preds)
    # save_to_gif(video_frames, 'video.gif')


    