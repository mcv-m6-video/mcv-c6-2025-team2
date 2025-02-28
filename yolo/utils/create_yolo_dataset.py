import os
import cv2
import xml.etree.ElementTree as ET
import random
from natsort import natsorted

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    os.makedirs(output_folder, exist_ok=True)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        frame_filename = os.path.join(output_folder, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'")

def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img.shape[1], img.shape[0]  

def parse_xml(xml_file, output_dir, image_path, class_mapping):
    image_width, image_height = get_image_dimensions(image_path)
    os.makedirs(output_dir, exist_ok=True)
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    for track in root.findall("track"):
        label = track.get("label")
        if label not in class_mapping:
            continue
        class_id = class_mapping[label]
        
        for box in track.findall("box"):
            frame = int(box.get("frame"))
            xtl, ytl, xbr, ybr = map(float, [box.get("xtl"), box.get("ytl"), box.get("xbr"), box.get("ybr")])
            
            x_center = ((xtl + xbr) / 2) / image_width
            y_center = ((ytl + ybr) / 2) / image_height
            width = (xbr - xtl) / image_width
            height = (ybr - ytl) / image_height
            
            txt_filename = os.path.join(output_dir, f"frame_{frame:04d}.txt")
            with open(txt_filename, "a") as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def create_symlinks(strategy, images_dir, labels_dir, output_base_dir, k=4):
    image_files = natsorted([f for f in os.listdir(images_dir) if f.endswith(".jpg")])
    num_images = len(image_files)

    random.seed(42)
    
    if strategy == "A":
        output_dir = os.path.join(output_base_dir, "yolo_A")
        os.makedirs(output_dir, exist_ok=True)
        for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        
        split_idx = num_images // 4
        train_images, val_images = image_files[:split_idx], image_files[split_idx:]

        for img in train_images:
            label = img.replace(".jpg", ".txt")
            os.symlink(os.path.join(images_dir, img), os.path.join(output_dir, "images/train", img))
            os.symlink(os.path.join(labels_dir, label), os.path.join(output_dir, "labels/train", label))
        for img in val_images:
            label = img.replace(".jpg", ".txt")
            os.symlink(os.path.join(images_dir, img), os.path.join(output_dir, "images/val", img))
            os.symlink(os.path.join(labels_dir, label), os.path.join(output_dir, "labels/val", label))
        
    elif strategy in ["B", "C"]:
        fold_size = num_images // k
        for fold in range(k):
            output_dir = os.path.join(output_base_dir, f"yolo_{strategy}_fold{fold}")
            os.makedirs(output_dir, exist_ok=True)
            for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
                os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
            
            if strategy == "B":
                train_start = fold * fold_size
                train_end = train_start + fold_size if fold != k - 1 else num_images
                train_images = image_files[train_start:train_end]
                val_images = image_files[:train_start] + image_files[train_end:]
            else:
                random.shuffle(image_files)
                train_images, val_images = image_files[:fold_size], image_files[fold_size:]

            for img in train_images:
                label = img.replace(".jpg", ".txt")
                os.symlink(os.path.join(images_dir, img), os.path.join(output_dir, "images/train", img))
                os.symlink(os.path.join(labels_dir, label), os.path.join(output_dir, "labels/train", label))
            for img in val_images:
                label = img.replace(".jpg", ".txt")
                os.symlink(os.path.join(images_dir, img), os.path.join(output_dir, "images/val", img))
                os.symlink(os.path.join(labels_dir, label), os.path.join(output_dir, "labels/val", label))

if __name__ == "__main__":
    data_path = "../data"
    video_file = os.path.join(data_path, "vdo.avi")
    frames_dir = os.path.join(data_path, "frames")
    annotations_file = os.path.join(data_path, "ai_challenge_s03_c010-full_annotation.xml")
    yolo_annotations_dir = os.path.join(data_path, "yolo_annotations")
    yolo_splits_dir = os.path.join(data_path, "yolo_splits")
    image_sample = os.path.join(frames_dir, "frame_0000.jpg")
    
    class_mapping = {"car": 0, "bike": 1}
    
    print("Starting frame extraction...")
    extract_frames(video_file, frames_dir)
    
    print("Converting annotations...")
    parse_xml(annotations_file, yolo_annotations_dir, image_sample, class_mapping)
    
    print("Creating dataset splits...")
    for strategy in ["A", "B", "C"]:
        create_symlinks(strategy, frames_dir, yolo_annotations_dir, yolo_splits_dir)
    
    print("YOLO dataset created.")
