import random
import gc
from typing import List
import cv2
import numpy as np
from tqdm import tqdm
from task1 import GaussianDetector
from task2 import GMMDetector
from metrics import parse_gt_annotations
from metrics_lastyear import mAP


def task1_search(num_experiments: int = 10, alpha_range: tuple = (1, 15)):
    """"
    Performs an random search of the Gaussian background subtraction model using different alpha values.

    Parameters:
        - num_experiments (int): Number of different alpha values to test.
        - alpha_range (tuple): Range of values to search for the optimal alpha [min, max].

    # TODO: Incloure també a la search l'àrea de threshold pel connected components??
    
    Outputs:
        - Saves the results to a file named "task1_results.txt" with the format:
            Alpha    mAP
            ---------------
            2.35    0.6721
            ...
    """

    # Initialize background model and train
    detector = GaussianDetector(
        video_path='/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/AICity_data/train/S03/c010/vdo.avi', 
        image_reduction=4,
        color_space=cv2.COLOR_BGR2Lab,
        mask_cleaning='ISOLATED',
        mask_enhancing='NAIVE'
    )

    detector.train()

    mean_0 = detector.background_mean
    std_0 = detector.background_std

    # Load ground truth
    ground_truth = parse_gt_annotations('/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/ai_challenge_s03_c010-full_annotation.xml')

    # Open results file
    results_file = "task1_results_new.txt"
    with open(results_file, "w") as f:
        f.write(
            "Alpha\tmAP\n"
            "---------------\n"
        )

    # Loop N times with random alpha
    for _ in tqdm(range(num_experiments), desc="Testing different alpha values"):
        
        # Generate a random alpha in the input range 
        alpha = random.uniform(alpha_range[0], alpha_range[1])

        # Reset the values of the detector to the trained ones
        detector.background_mean = mean_0
        detector.background_std = std_0

        detector.masks = None
        detector.objects = None

        # Load ground truth
        ground_truth = parse_gt_annotations('/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/ai_challenge_s03_c010-full_annotation.xml')
        ground_truth = ground_truth[detector.split_index:]  # Get only the ones from test part
        
        # Test with the random alpha
        detector.test(alpha=alpha, area_threshold=50)

        predictions = detector.get_predictions()

        # Evaluate predictions
        map = mAP(predictions, ground_truth)[0]

        # Save results to .txt
        with open(results_file, "a") as f:
            f.write(f"{alpha:.2f}\t{map:.4f}\n")

        print(f"Alpha: {alpha:.2f}, mAP: {map:.4f}\n")

        del predictions
        gc.collect()


def task2_gridsearch(alpha_values: List[int], rho_values: List[float], num_exec=1):
    # Load ground truth
    ground_truth = parse_gt_annotations('/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/ai_challenge_s03_c010-full_annotation.xml')
    ground_truth = ground_truth[535:]  # Get only the ones from test part

    # Open results file
    results_file = "task2_results_acotat7.txt"
    with open(results_file, "w") as f:
        f.write(
            "Alpha\tRho\t\tmAP\n"
            "----------------------\n"
        )

    # Run the grid search
    for alpha in tqdm(alpha_values, desc="Alpha search"):
        for rho in tqdm(rho_values, desc="Rho search", leave=False):
            for i in tqdm(range(num_exec), desc="Execution", leave=False):
                
                detector = GMMDetector(
                    video_path='/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/AICity_data/train/S03/c010/vdo.avi', 
                    image_reduction=4,
                    color_space=cv2.COLOR_BGR2Lab,
                    mask_cleaning='OPENING',
                    mask_enhancing='CLOSING'
                )

                detector.train(method='median')


                # Test with the current alpha and rho
                detector.test(alpha=alpha, rho=rho, area_threshold=125)

                predictions = detector.get_predictions()

                # Evaluate predictions
                map_score = mAP(predictions, ground_truth)[0]

                # Save results to .txt
                with open(results_file, "a") as f:
                    f.write(f"{alpha:.2f}\t{rho:.4f}\t{map_score:.4f}\n")

                print(f"Alpha: {alpha:.2f}, Rho: {rho:.4f}, mAP: {map_score:.4f}")

                del predictions
                gc.collect()


def task1_morphology_search(kernel_range: tuple = (3, 15), num_kernels: int = 5):
    """
    Evaluates different morphological operations with systematically chosen kernel sizes.

    Parameters:
        - kernel_range (tuple): Min and max kernel size (must be odd).
        - num_kernels (int): Number of kernel sizes to test per operation.

    Outputs:
        - Saves the results to "morphology_results.txt" with the format:
            Operation    Kernel Size    mAP
            --------------------------------
            opening      (3,3)          0.6721
            closing      (5,5)          0.6903
            ...
    """

    morph_operations = ['opening', 'closing', 'open-close', 'close-open']

    # Generate `num_kernels` evenly spaced odd values between kernel_range
    #kernel_sizes = np.linspace(kernel_range[0], kernel_range[1], num_kernels, dtype=int)
    #kernel_sizes = [(k, k) for k in kernel_sizes if k % 2 == 1]  # Ensure odd values
    
    open_kernel_sizes = [(3, 3), (7, 7), (11, 11)]
    close_kernel_sizes = [(15, 15), (20,20), (25,25), (30, 30), (40,40)]


    # Initialize detector and train
    detector = GaussianDetector(
        video_path='/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/AICity_data/train/S03/c010/vdo.avi',
        image_reduction=4,
        color_space=cv2.COLOR_BGR2Lab,
        mask_cleaning='OPENING',
        mask_enhancing='NAIVE'
    )
    detector.train(method='median')

    mean_0 = detector.background_mean
    std_0 = detector.background_std

    # Load ground truth
    ground_truth = parse_gt_annotations('/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/ai_challenge_s03_c010-full_annotation.xml')
    ground_truth = ground_truth[detector.split_index:]  # Use only the test part

    # Open results file
    results_file = "morphology_results_ir4_task1_ar.txt"
    with open(results_file, "w") as f:
        f.write(
            "Operation\tKernel Size\tmAP\n"
            "--------------------------------\n"
        )

    # Loop through operations
    for operation in morph_operations:
        # Handle combinations of kernel sizes for open-close and close-open operations
        if operation in ['open-close', 'close-open']:
            # Nested loop to try all combinations of open and close kernel sizes
            for open_kernel in open_kernel_sizes:
                for close_kernel in close_kernel_sizes:
                    # Reset detector to original trained values
                    detector.background_mean = mean_0
                    detector.background_std = std_0
                    detector.masks = None
                    detector.objects = None

                    # Set the operation
                    detector.mask_enhancing = operation  
                    detector.test(alpha=8.0, area_threshold=50, open_kernel=open_kernel, close_kernel=close_kernel)

                    predictions = detector.get_predictions()

                    # Evaluate performance
                    map_score = mAP(predictions, ground_truth)[0]

                    # Save results
                    with open(results_file, "a") as f:
                        f.write(f"{operation}\t{open_kernel}-{close_kernel}\t{map_score:.4f}\n")

                    print(f"Operation: {operation}, Open Kernel: {open_kernel}, Close Kernel: {close_kernel}, mAP: {map_score:.4f}")

                    # Cleanup
                    del predictions
                    gc.collect()

        
        if operation == 'closing':
            # For operations like opening and closing, we only need one kernel size for both open and close
            for kernel_size in close_kernel_sizes:
                # Reset detector to original trained values
                detector.background_mean = mean_0
                detector.background_std = std_0
                detector.masks = None
                detector.objects = None

                # Set the operation and test
                detector.mask_enhancing = operation  
                detector.test(alpha=8.0, area_threshold=50, open_kernel=kernel_size, close_kernel=kernel_size)

                predictions = detector.get_predictions()

                # Evaluate performance
                map_score = mAP(predictions, ground_truth)[0]

                # Save results
                with open(results_file, "a") as f:
                    f.write(f"{operation}\t{kernel_size}\t{map_score:.4f}\n")

                print(f"Operation: {operation}, Kernel Size: {kernel_size}, mAP: {map_score:.4f}")

                # Cleanup
                del predictions
                gc.collect()

        elif operation == 'opening':
            # For operations like opening and closing, we only need one kernel size for both open and close
            for kernel_size in open_kernel_sizes:
                # Reset detector to original trained values
                detector.background_mean = mean_0
                detector.background_std = std_0
                detector.masks = None
                detector.objects = None

                # Set the operation and test
                detector.mask_enhancing = operation  
                detector.test(alpha=8.0, area_threshold=50, open_kernel=kernel_size, close_kernel=kernel_size)

                predictions = detector.get_predictions()

                # Evaluate performance
                map_score = mAP(predictions, ground_truth)[0]

                # Save results
                with open(results_file, "a") as f:
                    f.write(f"{operation}\t{kernel_size}\t{map_score:.4f}\n")

                print(f"Operation: {operation}, Kernel Size: {kernel_size}, mAP: {map_score:.4f}")

                # Cleanup
                del predictions
                gc.collect()


def task2_morphology_search(kernel_range: tuple = (3, 15), num_kernels: int = 5):
    """
    Evaluates different morphological operations with systematically chosen kernel sizes.

    Parameters:
        - kernel_range (tuple): Min and max kernel size (must be odd).
        - num_kernels (int): Number of kernel sizes to test per operation.

    Outputs:
        - Saves the results to "morphology_results.txt" with the format:
            Operation    Kernel Size    mAP
            --------------------------------
            opening      (3,3)          0.6721
            closing      (5,5)          0.6903
            ...
    """

    morph_operations = ['opening', 'closing', 'open-close', 'close-open']

    # Generate `num_kernels` evenly spaced odd values between kernel_range
    #kernel_sizes = np.linspace(kernel_range[0], kernel_range[1], num_kernels, dtype=int)
    #kernel_sizes = [(k, k) for k in kernel_sizes if k % 2 == 1]  # Ensure odd values
    
    open_kernel_sizes = [(3, 3), (7, 7), (11, 11)]
    close_kernel_sizes = [(15, 15), (20,20), (25,25), (30, 30), (40,40)]


    # Initialize detector and train
    detector = GMMDetector(
        video_path='/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/AICity_data/train/S03/c010/vdo.avi', 
        image_reduction=4,
        color_space=cv2.COLOR_BGR2Lab,
        mask_cleaning='OPENING',
        mask_enhancing='CLOSING'
    )
    detector.train(method='median')

    mean_0 = detector.background_mean
    std_0 = detector.background_std

    # Load ground truth
    ground_truth = parse_gt_annotations('/Users/abriil/Uni/master/C6/Project1/project-c6/week1/data/ai_challenge_s03_c010-full_annotation.xml')
    ground_truth = ground_truth[detector.split_index:]  # Use only the test part

    # Open results file
    results_file = "morphology_results_ir4_task1_ar.txt"
    with open(results_file, "w") as f:
        f.write(
            "Operation\tKernel Size\tmAP\n"
            "--------------------------------\n"
        )

    # Loop through operations
    for operation in morph_operations:
        # Handle combinations of kernel sizes for open-close and close-open operations
        if operation in ['open-close', 'close-open']:
            # Nested loop to try all combinations of open and close kernel sizes
            for open_kernel in open_kernel_sizes:
                for close_kernel in close_kernel_sizes:
                    # Reset detector to original trained values
                    detector.background_mean = mean_0
                    detector.background_std = std_0
                    detector.masks = None
                    detector.objects = None

                    # Set the operation
                    detector.mask_enhancing = operation  
                    detector.test(alpha=8.0, rho=0.00, area_threshold=50, open_kernel=open_kernel, close_kernel=close_kernel)

                    predictions = detector.get_predictions()

                    # Evaluate performance
                    map_score = mAP(predictions, ground_truth)[0]

                    # Save results
                    with open(results_file, "a") as f:
                        f.write(f"{operation}\t{open_kernel}-{close_kernel}\t{map_score:.4f}\n")

                    print(f"Operation: {operation}, Open Kernel: {open_kernel}, Close Kernel: {close_kernel}, mAP: {map_score:.4f}")

                    # Cleanup
                    del predictions
                    gc.collect()

        
        if operation == 'closing':
            # For operations like opening and closing, we only need one kernel size for both open and close
            for kernel_size in close_kernel_sizes:
                # Reset detector to original trained values
                detector.background_mean = mean_0
                detector.background_std = std_0
                detector.masks = None
                detector.objects = None

                # Set the operation and test
                detector.mask_enhancing = operation  
                detector.test(alpha=8.0, rho=0.00, area_threshold=50, open_kernel=kernel_size, close_kernel=kernel_size)

                predictions = detector.get_predictions()

                # Evaluate performance
                map_score = mAP(predictions, ground_truth)[0]

                # Save results
                with open(results_file, "a") as f:
                    f.write(f"{operation}\t{kernel_size}\t{map_score:.4f}\n")

                print(f"Operation: {operation}, Kernel Size: {kernel_size}, mAP: {map_score:.4f}")

                # Cleanup
                del predictions
                gc.collect()

        elif operation == 'opening':
            # For operations like opening and closing, we only need one kernel size for both open and close
            for kernel_size in open_kernel_sizes:
                # Reset detector to original trained values
                detector.background_mean = mean_0
                detector.background_std = std_0
                detector.masks = None
                detector.objects = None

                # Set the operation and test
                detector.mask_enhancing = operation  
                detector.test(alpha=8.0, rho=0.00, area_threshold=50, open_kernel=kernel_size, close_kernel=kernel_size)

                predictions = detector.get_predictions()

                # Evaluate performance
                map_score = mAP(predictions, ground_truth)[0]

                # Save results
                with open(results_file, "a") as f:
                    f.write(f"{operation}\t{kernel_size}\t{map_score:.4f}\n")

                print(f"Operation: {operation}, Kernel Size: {kernel_size}, mAP: {map_score:.4f}")

                # Cleanup
                del predictions
                gc.collect()


if __name__ == "__main__":
    #task1_search(num_experiments=15)
    task2_gridsearch(
        alpha_values=[6.5, 6.7, 6.9, 7.1, 7.3, 7.5],
        rho_values=[0, 0.0001, 0.0005, 0.001, 0.005, 0.01],
    )
    #task1_morphology_search()
    #task2_morphology_search()
