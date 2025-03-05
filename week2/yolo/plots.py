import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from utils.metrics import Annotation, Prediction, compute_mAP
from evaluate import read_pickle, correct_gt_pickle, correct_preds_pickle
from typing import List, Dict, Tuple
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from matplotlib.collections import PolyCollection

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
    frames[0].save(filename, save_all=True, append_images=frames[sampling:-1:sampling], duration=100, loop=0)


def save_moving_average_video(mAP_dict: Dict[str, float], window_size=10, output_filename="mAP_evolution.mp4"):
    """
    Generates a video showing the evolution of the moving average plot over frames, 
    dynamically shading missing training frames as the animation progresses.
    """
    sorted_frame_ids = sorted(mAP_dict.keys(), key=lambda x: int(x.split('.')[0].split('_')[-1]))  
    frame_numbers = [int(fid.split('.')[0].split('_')[-1]) for fid in sorted_frame_ids]

    full_frame_range = np.arange(0, 2141)
    present_frames = set(frame_numbers)
    missing_frames = [num for num in full_frame_range if num not in present_frames]
    mAP_series = pd.Series({num: mAP_dict.get(f"frame_{num:04d}.jpg", np.nan) for num in full_frame_range})
    moving_avg = mAP_series.rolling(window=window_size, min_periods=1).mean()

    dot_color_1 = '#6fbeae'
    line_color_1 = '#274857'
    training_shade_color = '#f4cccc'  # Light red

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlim(0, max(full_frame_range))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("mAP")
    ax.set_title("mAP Evolution Over Time (Moving Average)")

    # Initialize plots
    sc1, = ax.plot([], [], marker='o', linestyle='', color=dot_color_1, markersize=1.3, alpha=0.4)
    line1, = ax.plot([], [], linestyle='-', color=line_color_1, linewidth=3, alpha=0.8)
    line2, = ax.plot([], [], linestyle='-', color=line_color_1, linewidth=3, alpha=0.8)

    # Create a PolyCollection for dynamic shading
    shading_poly = PolyCollection([], facecolor=training_shade_color, alpha=0.5)
    ax.add_collection(shading_poly)

    # Get first element position in numpy array with np.nan
    first_nan = 2141
    try:
        first_nan = np.where(np.isnan(mAP_series))[0][0] + 10
    except IndexError:
        pass

    def update(frame, first_nan):
        x_data = full_frame_range[:frame + 1]  # Full range up to current frame
        y_data_1 = mAP_series[:frame + 1]      # mAP data up to current frame

        # Splitting x_data and moving average into two parts
        x_data1 = full_frame_range[:min(first_nan, frame + 1)]
        x_data2 = full_frame_range[min(first_nan, frame + 1):frame + 1]

        y_data_2 = moving_avg[:min(first_nan, frame + 1)]
        y_data_3 = moving_avg[min(first_nan, frame + 1):frame + 1]

        # Update scatter plot
        sc1.set_data(x_data, y_data_1)

        # Filter valid (non-NaN) indices
        valid_indices1 = ~np.isnan(y_data_2)
        valid_indices2 = ~np.isnan(y_data_3)

        # Ensure x_data1 and x_data2 are non-empty before setting the line data
        if len(x_data1) > 0 and np.any(valid_indices1):
            line1.set_data(np.array(x_data1)[valid_indices1], np.array(y_data_2)[valid_indices1])
        else:
            line1.set_data([], [])  # Clear if no valid data

        if len(x_data2) > 0 and np.any(valid_indices2):
            line2.set_data(np.array(x_data2)[valid_indices2], np.array(y_data_3)[valid_indices2])
        else:
            line2.set_data([], [])  # Clear if no valid data

        # Update the shading dynamically
        if missing_frames:
            current_max_frame = frame + 1
            active_missing_frames = [f for f in missing_frames if f <= current_max_frame]

            if active_missing_frames:
                segments = []
                for k, g in pd.Series(active_missing_frames).groupby(pd.Series(active_missing_frames).diff().ne(1).cumsum()):
                    start, end = g.iloc[0], g.iloc[-1]
                    segments.append([[start, 0], [end, 0], [end, 1], [start, 1]])  # Define shaded area

                shading_poly.set_verts(segments)  # Update shading area

        return sc1, line1, line2, shading_poly

    ani = animation.FuncAnimation(fig, update, frames=len(full_frame_range), interval=50, blit=True, fargs=(first_nan,))
    ani.save(output_filename, writer="ffmpeg", fps=10)
    plt.close(fig)

def generate_gif_from_video(video_path: str, output_path: str, sampling: int = 5):
    """
    Generates a GIF from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()
    cv2.destroyAllWindows()

    save_to_gif(frames, output_path, sampling=sampling)

def delete_video(video_path: str):
    """
    Deletes a video file.
    """
    os.remove(video_path)


def plot_for_slide_7(
    all_maps: List[Dict[str, float]],
    output_filename: str
):
    assert len(all_maps) == 4, "Please provide 4 sets of mAPs."
    
    """
    Plots the predictions for slide 7.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlim(0, 2141)
    ax.set_ylim(0.4, 1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("mAP")
    ax.set_title("mAP Evolution Over Time")

    colors = ['b', 'r', 'g', 'y']
    labels = ['fold1', 'fold2', 'fold3', 'fold4']
    
    # Create scatter plots with labels for the legend
    scatter_plots = [ax.scatter([], [], marker='o', color=color, s=2, alpha=0.25, label=label) 
                     for color, label in zip(colors, labels)]
    
    # Add legend with larger, opaque markers
    legend = ax.legend(loc="lower right", fontsize=10, markerscale=5)  # Increase marker size in legend

    full_frame_range = np.arange(0, 2141)
    mAP_series_list = [pd.Series({num: maps.get(f"frame_{num:04d}.jpg", np.nan) for num in full_frame_range}) 
                       for maps in all_maps]

    def update(frame):
        x_data = full_frame_range[:frame + 1]

        for i, scatter in enumerate(scatter_plots):
            y_data = mAP_series_list[i][:frame + 1]
            valid_indices = ~np.isnan(y_data)
            scatter.set_offsets(np.column_stack((x_data[valid_indices], y_data[valid_indices])))

        return scatter_plots

    ani = animation.FuncAnimation(fig, update, frames=2141, interval=50, blit=True)
    ani.save(output_filename, writer="ffmpeg", fps=10)
    plt.close(fig)



if __name__ == '__main__':

    plot_type = 'slide_7'

    gt_file = os.path.join(DATA_PATH, 'gt.pkl')
    gt = read_pickle(gt_file)
    gt = correct_gt_pickle(gt)

    if plot_type == 'ma':
        name = 'preds_pred_off-shelf_truck'
        preds_file = os.path.join(DATA_PATH, f'{name}.pkl')

        preds = read_pickle(preds_file)
        preds = correct_preds_pickle(preds)

        mAPs = compute_mAP(gt, preds, aggregated=False)
        frames = save_moving_average_video(mAPs, window_size=10, output_filename=f'{name}.mp4')
        generate_gif_from_video(f'{name}.mp4', f'{name}.gif', sampling=10)
        delete_video(f'{name}.mp4')
    elif plot_type == 'slide_7':
        all_maps = []
        for fold in range(4):
            name = f'preds_pred_C_fold{fold}'
            preds_file = os.path.join(DATA_PATH, f'{name}.pkl')
            preds = read_pickle(preds_file)
            preds = correct_preds_pickle(preds)
            mAPs = compute_mAP(gt, preds, aggregated=False)
            all_maps.append(mAPs)

        plot_for_slide_7(all_maps, 'slide7.mp4')
        generate_gif_from_video('slide7.mp4', 'slide7.gif', sampling=10)
        delete_video('slide7.mp4')
    else:
        raise ValueError("Invalid plot type.")


    