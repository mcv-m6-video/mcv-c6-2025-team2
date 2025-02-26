import numpy as np
import cv2
from tqdm import tqdm
from .detector.detector import Detector
from .metrics import generate_bounding_box_video_from_frames, parse_gt_annotations
from .metrics_lastyear import mAP

class GMMDetector(Detector):

    def __init__(self, **kwargs):
        super(GMMDetector, self).__init__(**kwargs)

    def _classification(self, alpha: float = 2.0, rho: float = 0):
        """
        Takes the test frames and performs the binary classification: background/foreground,
        based on the background model that has been created.
        """

        masks = []

        # Iterate over test frames directly from the generator
        for new_frame in tqdm(self.get_test_frames(normalized=True), 'Classifying test frames'):
            new_frame = np.array(new_frame, dtype=np.float32)  # Ensure correct type
            diff = np.abs(new_frame - self.background_mean)

            # Create the mask: pixel is foreground if |pixel - mean| > threshold * (std+2)
            mask = np.any(diff >= (alpha * self.background_std + 2/255.), axis=-1).astype(np.uint8)
            masks.append(mask)
            
            # Update mean and variance for background pixels only
            background_pixels = (mask == 0)  # Pixels classified as background
            
            self.background_mean[background_pixels] = (
                rho * new_frame[background_pixels] + (1 - rho) * self.background_mean[background_pixels]
            )

            self.background_std[background_pixels] = np.sqrt(
                rho * (new_frame[background_pixels] - self.background_mean[background_pixels])**2 +
                (1 - rho) * self.background_std[background_pixels]**2
            )

        self.masks = np.array(masks)  # Convert to array at the end

def compute_per_frame_mAP(predictions, ground_truth):
    """
    Compute the mAP for each frame individually.
    """
    frame_mAPs = []
    for pred, gt in zip(predictions, ground_truth):
        frame_mAPs.append(mAP([pred], [gt])[0])  # Compute mAP per frame
    return frame_mAPs

if __name__ == "__main__":
    # Background model: Recursive update
    detector = GMMDetector(
        video_path = 'week1/data/AICity_data/train/S03/c010/vdo.avi',
        image_reduction=4,
        color_space=cv2.COLOR_BGR2Lab,
        mask_cleaning='OPENING',
        mask_enhancing='CLOSING'
    )

    detector.train(method='median')
    detector.test(alpha=7.0, rho=0.0005, area_threshold=125)


    predictions = detector.get_predictions()

    # Load ground truth
    ground_truth = parse_gt_annotations('week1/data/ai_challenge_s03_c010-full_annotation.xml')

    # Get only the ones from test part
    ground_truth = ground_truth[detector.split_index:]

    # Open results file
    results_file = "task2_results.txt"
    with open(results_file, "w") as f:
        f.write(
            "Alpha\tRho\t\tmAP\n"
            "----------------------\n"
        )
    
    # Evaluate predictions
    mAP_score = mAP(predictions, ground_truth)

    print(mAP_score)

    detector.save_masks_as_video(output_path='masks_adaptive.mp4')

    # Trash instance detector to get the frames as images to save the video
    detector_trash = GMMDetector(video_path = 'week1/data/AICity_data/train/S03/c010/vdo.avi')
    frames = detector_trash.get_test_frames()
    generate_bounding_box_video_from_frames(predictions, frames, output_path='bbox_adaptive.mp4')
    frame_mAP_scores = compute_per_frame_mAP(predictions, ground_truth)

    # Save mAP values to a file
    np.savetxt("frame_mAPs_task2.txt", frame_mAP_scores)   
