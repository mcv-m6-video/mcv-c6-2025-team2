import numpy as np
import cv2
from tqdm import tqdm
from detector.detector import Detector
from metrics import generate_bounding_box_video_from_frames, parse_gt_annotations
from metrics_lastyear import mAP


class GaussianDetector(Detector):

    def __init__(self, **kwargs):
        super(GaussianDetector, self).__init__(**kwargs)

    def _classification(self, alpha: float = 2.0):
        masks = []

        # Iterate over test frames directly from the generator
        for new_frame in tqdm(self.get_test_frames(), 'Classifying test frames'):
            new_frame = np.array(new_frame, dtype=np.float32)  # Ensure correct type
            diff = np.abs(new_frame - self.background_mean)

            # Create the mask: pixel is foreground if |pixel - mean| > threshold * (std+2)
            mask = np.any(diff > (alpha * self.background_std+2), axis=-1).astype(np.uint8)
            masks.append(mask)

        self.masks = np.array(masks)  # Convert to array at the end

if __name__ == "__main__":
    # Background model: Recursive update
    detector = GaussianDetector(
        video_path = 'week1/data/AICity_data/train/S03/c010/vdo.avi',
        image_reduction=4,
        color_space=cv2.COLOR_BGR2Lab,
        mask_cleaning='ISOLATED',
        mask_enhancing='CLOSING'
    )

    alpha = 14.0

    # Load ground truth
    ground_truth = parse_gt_annotations('week1/data/ai_challenge_s03_c010-full_annotation.xml')

    # Get only the ones from test part
    ground_truth = ground_truth[detector.split_index:]

    # Open results file
    results_file = "task1_results.txt"
    with open(results_file, "w") as f:
        f.write(
            "Alpha\tRho\t\tmAP\n"
            "----------------------\n"
        )
    
    detector.train()
    detector.test(alpha=alpha, area_threshold=125)

    predictions = detector.get_predictions()

    # Evaluate predictions
    mAP_score = mAP(predictions, ground_truth)

    print(mAP_score)

    detector.save_masks_as_video(output_path=f'masks_still_{alpha}.mp4')

    # Trash instance detector to get the frames as images to save the video
    detector_trash = GaussianDetector(video_path = 'week1/data/AICity_data/train/S03/c010/vdo.avi')
    frames = detector_trash.get_test_frames()
    generate_bounding_box_video_from_frames(predictions, frames, output_path=f'bbox_still_{alpha}.mp4', ground_truth=ground_truth)
