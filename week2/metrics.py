import numpy as np
from collections import defaultdict
import xml.etree.ElementTree as ET
import numpy as np


########################
####### TASK 2.3 #######
########################

def parse_xml_HOTA(file_path: str) -> dict:
    "Reads the GT file and returns a dictionary prepared for the HOTA metric."

    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Inicializamos los datos que necesitamos
    gt_ids = defaultdict(list)
    num_gt_dets = 0
    unique_ids = set()

    # Iteramos a través de los tracks
    for track in root.findall('track'):
        track_id = int(track.get('id'))
        unique_ids.add(track_id)  # Agregar track_id al conjunto de IDs únicos
        frames = []

        # Iteramos a través de las cajas dentro de cada track
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            # Almacenamos el ID del track en cada frame
            gt_ids[frame].append(track_id)
            num_gt_dets += 1

        # No es necesario contar el número de IDs únicos aquí porque lo hacemos con el conjunto unique_ids

    num_gt_ids = len(unique_ids)  # Número total de IDs únicos

    # Creamos el diccionario con los resultados
    result = {
        "num_gt_dets": num_gt_dets,
        "num_gt_ids": num_gt_ids,
        "gt_ids": [np.array(gt_ids[frame]) for frame in sorted(gt_ids.keys()) if frame >= 535]
    }

    return result

def parse_bboxes_from_xml(file_path):
    # Parsear el archivo XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Crear un diccionario para almacenar las coordenadas por frame
    frames_dict = {}

    # Iterar sobre los elementos <track>
    for track in root.findall('track'):
        # Iterar sobre los elementos <box> dentro de cada <track>
        for box in track.findall('box'):
            # Obtener el número de frame
            frame_number = int(box.get('frame'))

            # Obtener las coordenadas del bounding box
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))

            # Si el frame no está en el diccionario, añadirlo
            if frame_number not in frames_dict:
                frames_dict[frame_number] = []

            # Añadir las coordenadas del bounding box al frame correspondiente
            frames_dict[frame_number].append([xtl, ytl, xbr, ybr])

    # Crear una lista de listas ordenada por los frames
    min_frame = 535
    max_frame = max(frames_dict.keys())  # Obtener el último frame
    frame_list = []
    
    for i in range(min_frame, max_frame+1):
        frame_list.append(frames_dict.get(i, []))  # Si no hay cajas para un frame, poner una lista vacía

    return frame_list


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou

def compute_similarity_matrices(gt_bboxes_per_frame, detected_bboxes_per_frame):
    """
    Computes IoU similarity matrices for all frames.

    :param gt_bboxes_per_frame: List of lists containing GT bounding boxes per frame.
    :param detected_bboxes_per_frame: List of lists containing detected bounding boxes per frame.
    :return: List of 2D numpy arrays representing IoU matrices for each frame.
    """
    similarity_matrices = []
    
    for gt_bboxes, detected_bboxes in zip(gt_bboxes_per_frame, detected_bboxes_per_frame):
        num_gt = len(gt_bboxes)
        num_det = len(detected_bboxes)
        similarity_matrix = np.zeros((num_gt, num_det))

        for i, gt_bbox in enumerate(gt_bboxes):
            for j, det_bbox in enumerate(detected_bboxes):
                similarity_matrix[i, j] = iou(gt_bbox, det_bbox)
        
        similarity_matrices.append(similarity_matrix)
    
    return similarity_matrices

    
def create_data_for_hota(gt_bboxes, tracking_stats):
   
    # Compute stats from GT
    gt_stats = parse_xml_HOTA("data/ai_challenge_s03_c010-full_annotation.xml")

    # Compute similarity metric btwn GT and detected bboxes
    similarity_scores = compute_similarity_matrices(gt_bboxes, tracking_stats['tracker_bboxes'])

    data = {
        "num_tracker_dets": tracking_stats["num_tracker_dets"],
        "num_gt_dets": gt_stats["num_gt_dets"],
        
        "num_tracker_ids": tracking_stats["num_tracker_ids"],
        "num_gt_ids": gt_stats["num_gt_ids"],

        "gt_ids": gt_stats["gt_ids"],  # IDs de objetos en cada frame (GT)
        "tracker_ids": tracking_stats["tracker_ids"],  # IDs de Tracker en cada frame

        "similarity_scores": similarity_scores
    }

    return data