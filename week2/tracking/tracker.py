import cv2
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict

class Tracker:
    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.tracks = {}  # Stores object ID -> bbox
        self.next_id = 0
        self.object_colors = {}
        self.gt_detections = []
        self.tracking_data = {
            "num_tracker_dets": 0,
            "num_gt_dets": 0,
            "num_gt_ids": 0,
            "num_tracker_ids": 0,
            "gt_ids": [],
            "tracker_ids": [],
            "similarity_scores": []
        }
    
    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def update(self, detections, gt_detections):
        new_tracks = {}
        assigned = set()
        tracker_ids = []
        
        self.tracking_data["similarity_scores"].append(self.compute_similarity_matrix(gt_detections, [d['bbox'] for d in detections]))
        
        for track_id, prev_box in self.tracks.items():
            best_match = None
            best_iou = self.iou_threshold
            
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                
                iou_score = self.iou(prev_box, det['bbox'])
                #print(iou_score)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_match = i
            
            if best_match is not None:
                new_tracks[track_id] = detections[best_match]['bbox']
                assigned.add(best_match)
                tracker_ids.append(track_id)
        
        for i, det in enumerate(detections):
            if i not in assigned:
                new_tracks[self.next_id] = det['bbox']
                tracker_ids.append(self.next_id)
                self.next_id += 1
        
        self.tracking_data["num_tracker_dets"] += len(tracker_ids)
        self.tracking_data["tracker_ids"].append(np.array(tracker_ids))

        self.tracks = new_tracks
        return self.tracks

    def draw_tracks(self, frame):
        for track_id, bbox in self.tracks.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    def compute_similarity_matrix(self, gt_bboxes, detected_bboxes):
        """
        Computes IoU similarity between all ground truth and detected bounding boxes in a specific frame.

        :param frame_gt_bboxes: List of ground truth bounding boxes in the frame (format: [x1, y1, x2, y2]).
        :param frame_detected_bboxes: List of detected bounding boxes in the frame (format: [x1, y1, x2, y2]).
        :return: A matrix (2D list) containing IoU scores between GT and detected bboxes.
        """
        num_gt = len(gt_bboxes)
        num_det = len(detected_bboxes)

        similarity_matrix = np.zeros((num_gt, num_det))

        for i, gt_bbox in enumerate(gt_bboxes):
            for j, det_bbox in enumerate(detected_bboxes):
                similarity_matrix[i, j] = self.iou(gt_bbox, det_bbox)

        return similarity_matrix

    
    def load_gt_annotations_xml(self, file_path, split_index):
        # Parsear el archivo XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Crear un diccionario para almacenar las coordenadas por frame
        frames_dict = {}

        gt_ids = defaultdict(list)
        num_gt_dets = 0
        unique_ids = set()

        # Iterar sobre los elementos <track>
        for track in root.findall('track'):
            track_id = int(track.get('id'))
            unique_ids.add(track_id) 
            frames = []

            for box in track.findall('box'):
                # Obtener el número de frame
                frame_number = int(box.get('frame'))

                # Solo procesar frames desde el 535 en adelante
                if frame_number >= split_index:
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

                    gt_ids[frame_number].append(track_id)
                    num_gt_dets += 1

            num_gt_ids = len(unique_ids)  # Número total de IDs únicos
            
        # Creamos el diccionario con los resultados
        self.tracking_data["num_gt_dets"] = num_gt_dets
        self.tracking_data["num_gt_ids"] = num_gt_ids,
        self.tracking_data["gt_ids"] = [np.array(gt_ids[frame]) for frame in sorted(gt_ids.keys()) if frame >= split_index]

        # Crear una lista de listas ordenada por los frames desde el 535 en adelante
        if frames_dict:
            max_frame = max(frames_dict.keys())  # Obtener el último frame presente
            frame_list = []

            for i in range(split_index, max_frame + 1):
                frame_list.append(frames_dict.get(i, []))  
            self.gt_detections = frame_list
            #return frame_list
        else:
            self.gt_detections = []
            #return []  

    def draw_tracks2(self, frame, object_colors):
        result_image = frame.copy()

        for track_id, bbox in self.tracks.items():
            x1, y1, x2, y2 = map(int, bbox)
            # Assign a unique color if the object ID is new
            if track_id not in object_colors:
                np.random.seed(track_id)  # Seed with object ID to ensure consistency
                object_colors[track_id] = tuple(map(int, np.random.randint(0, 255, size=3)))
            
            color = object_colors[track_id]
            
            # Draw the detection rectangle
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add text with the object ID
            cv2.putText(
                result_image, 
                f"ID:{track_id}", 
                (int(x1), int(y1) - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )
        
        return result_image, object_colors
    
    def visualize_tracked_objects(self, image, tracked_objects, object_colors):
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