import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from typing import List, Tuple, Generator, Optional
from cv2.typing import MatLike
from scipy.ndimage import convolve

TRAIN_PROPORTION = 0.25


class Detector:
    def __init__(
        self,
        background_mean: Optional[np.ndarray]=None,
        background_std: Optional[np.ndarray]=None,
        video_path: str = 'week1/data/AICity_data/train/S03/c010/vdo.avi',
        image_reduction: int = 1,
        color_space: Optional[int] = None,
        mask_cleaning: str = 'NAIVE',
        mask_enhancing: str = 'NAIVE'
    ):
        # Ensure video exists and store video path
        self.ensure_video_path(video_path)
        self.video_path = video_path

        # Working variables
        self.background_mean = background_mean  # Tensor for bkg mean (height, width, 3)
        self.background_std = background_std  # Tensor for bkg standard deviation (height, width, 3)
        self.masks = None  # Tensor for the background/foreground mask of each test frame (N, height, width)
        self.objects = None  # List of the detected objects in each frame
        self.masks_cleaned = []

        # Get indexes for training and testing
        self._get_indexes()

        # Deal with image size
        self.image_reduction = image_reduction
        self._original_image_size = self._get_original_image_size()
        self._reduced_image_size = tuple(int(size / image_reduction) for size in self._original_image_size)

        # Other parameters
        self.color_space = color_space
        self.mask_cleaning = mask_cleaning
        self.mask_enhancing = mask_enhancing
    
    @staticmethod
    def ensure_video_path(video_path: str):
        if not video_path.endswith('.avi'):
            raise ValueError("Video path must be an .avi file'")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

    def _get_original_image_size(self) -> Tuple[int, int]:
        video = self._read_video()
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()
        return width, height

    def _read_video(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self.video_path)
    
    def _get_indexes(self) -> int:
        video = self._read_video()
        self.total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        self.split_index = int(self.total_frames * TRAIN_PROPORTION)
    
    def _get_original_position_in_frame(self, *args) -> Tuple[int, ...]:
        return tuple(int(arg * self.image_reduction) for arg in args)
    
    
    def read_frames(self, start: int, end: int, normalized: bool = False) -> Generator[MatLike, None, None]:
        video = self._read_video()
        video.set(cv2.CAP_PROP_POS_FRAMES, start)  # Jump to start frame
        for _ in range(start, end):
            ret, frame = video.read()
            if not ret:
                break
            resized = cv2.resize(frame, self._reduced_image_size)
            resized = resized if self.color_space is None else cv2.cvtColor(resized, self.color_space)
            resized = resized / 255.0 if normalized else resized
            yield resized
        video.release()

    def get_train_frames(self, normalized: bool = False) -> Generator[MatLike, None, None]:
        for frame in self.read_frames(0, self.split_index, normalized):
            yield frame
    
    def get_test_frames(self, normalized: bool = False) -> Generator[MatLike, None, None]:
        for frame in self.read_frames(self.split_index, self.total_frames, normalized):
            yield frame

    def get_all_frames(self, normalized: bool = False) -> Generator[MatLike, None, None]:
        for frame in self.read_frames(0, self.total_frames, normalized):
            yield frame

    @classmethod
    def load_model(cls, file_path: str):
        """
        Loads the background mean and std from a file and returns an instance of the class.

        Parameters:
        - file_path: Path from where the background model should be loaded.

        Returns:
        - An instance of `cls` with the loaded background model.
        """
        data = np.load(file_path)
        return cls(background_mean=data['mean'], background_std=data['std'])
    
    def save_model(self, file_path: str):
        """
        Saves the background mean and std to a file.

        Parameters:
        - file_path: Path where the background model should be saved.
        """
        np.savez(file_path, mean=self.background_mean, std=self.background_std)
        print(f"Background model saved to {file_path}")        

   
    def _clean_mask(self, mask: np.ndarray): # TODO: Improve
        if self.mask_cleaning == "NAIVE":
            return mask
        
        if self.mask_cleaning == "OPENING":
            # Structuring element: Square 3x3
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # Opening
            cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            return cleaned_mask
        
        if self.mask_cleaning == "ISOLATED":
            # Define a 3x3 neighborhood kernel
            kernel = np.array([[1, 1, 1],
                            [1, 0, 1],
                            [1, 1, 1]])

            # Convolve mask to count neighbors for each pixel
            neighbor_count = convolve(mask, kernel, mode='constant', cval=0)

            # Isolated pixels are those where mask == 1 and neighbor_count == 0
            isolated_mask = (mask == 1) & (neighbor_count < 1)

            # Remove isolated pixels
            mask[isolated_mask] = 0
            return mask
        
        raise ValueError(f"Invalid mask cleaning method: {self.mask_cleaning}")

    def _enhance_mask(self, mask: np.ndarray, open_kernel: tuple, close_kernel: tuple):  # Updated to accept two kernels
        if self.mask_enhancing == "NAIVE":
            return mask
        
        if self.mask_enhancing == "CLOSING":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
            enhanced_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return enhanced_mask
        
        # Morphology gridsearch with different kernels for open and close
        open_kernel_cv = np.ones(open_kernel, np.uint8)
        close_kernel_cv = np.ones(close_kernel, np.uint8)

        if self.mask_enhancing == 'opening':
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel_cv)
        
        elif self.mask_enhancing == 'closing':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel_cv)
        
        elif self.mask_enhancing == 'open-close':
            temp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel_cv)
            return cv2.morphologyEx(temp, cv2.MORPH_CLOSE, close_kernel_cv)
        
        elif self.mask_enhancing == 'close-open':
            temp = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel_cv)
            return cv2.morphologyEx(temp, cv2.MORPH_OPEN, open_kernel_cv)
        
        raise ValueError(f"Invalid mask enhancing method: {self.mask_enhancing}")



    def _detection(self, area_threshold: int = 100, open_kernel=(3,3), close_kernel=(3,3)):
        """
        Detects objects in the given binary masks using connected components.
        Visualizes the masks with corresponding bounding boxes.
        """

        self.objects = []

        for frame_index, mask in tqdm(enumerate(self.masks), 'Detecting objects'):
            mask = self._clean_mask(mask)
            mask = self._enhance_mask(mask, open_kernel=open_kernel, close_kernel=close_kernel)
            self.masks_cleaned.append(mask)
            num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            detected_objects = []
            for i in range(1, num_labels):  # Skip background
                if stats[i, cv2.CC_STAT_AREA] > area_threshold:  # Ignore small objects
                    mask_object = (labels == i).astype(np.uint8)
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    bounding_box = self._get_original_position_in_frame(x, y, x+w, y+h)
                    detected_objects.append({
                        'mask': cv2.resize(mask_object, self._original_image_size), 
                        'bounding_box': bounding_box
                    })
            
            self.objects.append(detected_objects)

            #self._plot_mask_with_boxes(cv2.resize(mask, self._original_image_size), detected_objects=detected_objects, frame_index=frame_index)


    def save_masks_as_video(self, output_path="masks.mp4", fps=30, type='raw'):
        """
        Saves the binary masks as a video.

        Parameters:
            output_path (str): Path where the output video will be saved.
            fps (int): Frames per second of the output video.
        """
        if type == 'raw':
            masks = self.masks
        elif type == 'clean':
            masks = self.masks_cleaned

        if not hasattr(self, 'masks') or len(masks) == 0:
            raise ValueError("No masks available to save as video.")

        # Get frame dimensions
        height, width = masks[0].shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

        for mask in masks:
            mask = (mask * 255).astype(np.uint8)  # Convert binary mask to grayscale (0-255)
            out.write(mask)

        out.release()
        print(f"Video saved as {output_path}")


    
    def _plot_mask_with_boxes(self, mask, detected_objects, frame_index):
        """
        Plots the mask with bounding boxes.

        Args:
            mask: The binary mask for the current frame.
            detected_objects: The list of detected objects and their bounding boxes.
            frame_index: The index of the current frame.
        """
        # Crea una imagen en color a partir de la máscara
        mask = cv2.resize(mask, self._original_image_size)
        colored_mask = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR)

        # Dibuja las bounding boxes en la imagen
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['bounding_box']
            cv2.rectangle(colored_mask, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # Muestra la máscara con las bounding boxes
        plt.figure(figsize=(10, 10))
        plt.imshow(colored_mask)
        plt.title(f'Frame {frame_index + 1}')
        plt.axis('off')  # Oculta los ejes
        plt.savefig(f"frame_{frame_index}")


    def train(self, method='mean'):
        """
        Estimates the background model using the training frames.
        
        Parameters:
        - method: Method to estimate the background model. Options: 'mean', 'median'.
        """
        assert method in ['mean', 'median'], f"Method {method} not implemented"
        
        # Get the train frames
        frames_list = list(self.get_train_frames(normalized=True))  # N frames
        frames = np.array(frames_list, dtype=np.float32)  # (N, height, width, 3)
        
        # Compute mean and std across the temporal axis
        print('Computing mean...', end='\r')
        if method == 'mean':
            mean = np.mean(frames, axis=0)
        elif method == 'median':
            mean = np.median(frames, axis=0)

        print('Computing std...', end='\r')
        std = np.std(frames, axis=0) #mean=mean)

        self.background_mean = mean  # (height, width, 3)
        self.background_std = std  # (height, width, 3)


    def test(self, alpha: float = 4, area_threshold: int = 100, open_kernel = (3,3), close_kernel=(3,3), **kwargs):
        self._classification(alpha=alpha, **kwargs)
        self._detection(area_threshold=area_threshold, open_kernel=open_kernel, close_kernel=close_kernel)


    def _classification(alpha: float = 2.0, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")
    

    def get_predictions(self):
        assert self.objects is not None, "No objects detected yet, please call test first"
        predictions = []
        for objects in self.objects:
            frame_objects = []

            for object in objects:
                frame_objects.append(object['bounding_box'])
            
            predictions.append(frame_objects)
        
        return predictions
