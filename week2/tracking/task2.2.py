import numpy as np
import time
import pickle
import cv2
from tqdm import tqdm
from sort import Sort
import xml.etree.ElementTree as ET
import os
import pickle
import json  # Para guardar los resultados en formato JSON

import sys
sys.path.append("week2/") 
from metrics import create_data_for_hota, parse_bboxes_from_xml
from tracking.TrackEval.hota import HOTA
from TrackEval.identity import Identity


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
    Los cuadros se dibujan con mayor grosor y los IDs se muestran en un tamaño de fuente mayor.
    """
    # Crear una copia de la imagen para no modificar la original
    result_image = image.copy()
    
    for d in tracked_objects:
        x1, y1, x2, y2, obj_id = d  # Desempaquetar coordenadas e ID del objeto
        obj_id = int(obj_id)
        
        # Asignar un color único si el objeto es nuevo
        if obj_id not in object_colors:
            np.random.seed(obj_id)  # Semilla con el objeto ID para asegurar consistencia
            object_colors[obj_id] = tuple(map(int, np.random.randint(0, 255, size=3)))
        
        color = object_colors[obj_id]
        
        # Dibujar el rectángulo de detección con mayor grosor (por ejemplo, 8 píxeles)
        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)
        
        # Añadir el texto con el ID del objeto en tamaño mayor (fuente 1.0 y grosor 3)
        cv2.putText(
            result_image, 
            f"ID:{obj_id}", 
            (int(x1), int(y1) - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            color, 
            3
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
    all_tracker_ids = []  # All unique tracker IDs across all frames
    tracker_ids_by_frame = []  # IDs in each frame
    tracker_bboxes_by_frame = []  # IDs in each frame
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

        # Extract bboxes from this frame's tracked objects
        frame_bboxes = [obj[:4] for obj in tracked_objects]
        tracker_bboxes_by_frame.append(frame_bboxes)
        
        # Extract IDs from this frame's tracked objects
        frame_ids = np.array([int(obj[4]-1) for obj in tracked_objects])
        tracker_ids_by_frame.append(frame_ids)
        
        # Update unique IDs and detection count
        all_tracker_ids.extend(frame_ids)  
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
        "num_tracker_ids": max(np.unique(all_tracker_ids))+1,
        "tracker_bboxes": tracker_bboxes_by_frame,
        "tracker_ids": tracker_ids_by_frame
    }
    
    return stats


def process_detections_in_folder(detections_folder, frames_dir, gt_file_path, output_results_path):
    # Cargar las anotaciones de GT
    gt_bboxes = parse_bboxes_from_xml(gt_file_path)

    # Inicializar una lista para almacenar los resultados de HOTA e IDF1
    all_results = []

    # Iterar sobre los archivos en la carpeta de detecciones
    for filename in tqdm(os.listdir(detections_folder), desc="Processing detection files", unit="file"):
        if filename.endswith('.pkl'):
            # Cargar las detecciones desde el archivo
            detection_file_path = os.path.join(detections_folder, filename)
            with open(detection_file_path, "rb") as f:
                detections = pickle.load(f)

            # Obtener los resultados de tracking
            tracking_stats = get_tracking(detections, frames_dir, 
                                        generate_video=False, fps=10, output_video_path=None)

            if tracking_stats:
                print(f"=== Tracking Statistics for {filename} ===")
                print(f"Total tracker detections: {tracking_stats['num_tracker_dets']}")
                print(f"Total unique tracker IDs: {tracking_stats['num_tracker_ids']}")

            # Crear los datos para la evaluación de HOTA
            data = create_data_for_hota(gt_bboxes, tracking_stats)
            
            # Calcular HOTA y IDF1
            hota_metric = HOTA()
            iden = Identity()

            # Evaluar las métricas
            hota_result = hota_metric.eval_sequence(data)
            idf1_result = iden.eval_sequence(data)

            # Almacenar los resultados en el formato deseado
            result = {
                'filename': filename,
                'HOTA(0)': hota_result['HOTA(0)'],
                'IDF1': idf1_result['IDF1']
            }
            all_results.append(result)

    # Guardar los resultados en un archivo JSON
    with open(output_results_path, 'w') as result_file:
        json.dump(all_results, result_file, indent=4)


if __name__ == "__main__":
    # # Definir las rutas
    # detections_folder = "week2/tracking/to_reformat"  # La carpeta con los archivos .pkl
    # frames_dir = 'data/AICity_data/frames'  # La carpeta de frames
    # gt_file_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C6. Video Analysis/mcv-c6-2025-team2/data/ai_challenge_s03_c010-full_annotation.xml'  # El archivo de GT
    # output_results_path = 'tracking_results_original.json'  # El archivo donde se guardarán los resultados

    # # Llamar a la función para procesar las detecciones
    # process_detections_in_folder(detections_folder, frames_dir, gt_file_path, output_results_path)

    with open("/Users/arnaubarrera/Desktop/MSc Computer Vision/C6. Video Analysis/mcv-c6-2025-team2/week2/tracking/to_reformat/preds_pred_B_fold3_wholevid.pkl", "rb") as f:
        detections = pickle.load(f)
    
    # Prepare data for tracker
    # detections = reformat_detections(seq_dets, start_number=535)
    frames_dir = 'data/AICity_data/frames'

    # Process tracking and optionally create visualization
    tracking_stats = get_tracking(detections, frames_dir, 
                                  generate_video=True, fps=30, output_video_path="object_tracking.mp4")
    
    # Print statistics
    if tracking_stats:
        print("\n=== Tracking Statistics ===")
        print(f"Total tracker detections: {tracking_stats['num_tracker_dets']}")
        print(f"Total unique tracker IDs: {tracking_stats['num_tracker_ids']}")

    # Evaluation - HOTA
    # Bboxes detected and GT
    file_path = '/Users/arnaubarrera/Desktop/MSc Computer Vision/C6. Video Analysis/mcv-c6-2025-team2/data/ai_challenge_s03_c010-full_annotation.xml'
    gt_bboxes = parse_bboxes_from_xml(file_path)

    data = create_data_for_hota(gt_bboxes, tracking_stats)
    
    # Compute HOTA and IDF1
    hota_metric = HOTA()
    iden = Identity()

    results = []
    results.append(hota_metric.eval_sequence(data))
    results.append(iden.eval_sequence(data))

    print(results)
