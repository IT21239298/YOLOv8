from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import Counter
from PIL import Image
import pickle
import operator

class LoadVideo:
    def __init__(self, path, img_size=(1088, 608)):
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

def get_FrameLabels(frame):
    frame_width = frame.shape[1]
    text_scale = max(1, frame_width / 1600.)
    text_thickness = 1 if text_scale > 1.1 else 1
    line_thickness = max(1, int(frame_width / 500.))
    return text_scale, text_thickness, line_thickness

def cv2_addBox(id, frame, left, top, right, bottom, line_thickness, text_thickness, text_scale):
    color = get_color(abs(id))
    cv2.rectangle(frame, (left, top), (right, bottom), color=color, thickness=line_thickness)
    cv2.putText(frame, 'ID-' + str(id), (left, top - 12), 0, text_scale, (0, 0, 255), thickness=text_thickness, lineType=cv2.LINE_AA)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def write_results(filename, data_type, w_frame_id, w_track_id, w_x1, w_y1, w_x2, w_y2, w_wid, w_hgt):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{x2},{y2},{w},{h}\n'
    else:
        raise ValueError(data_type)
    with open(filename, 'a') as f:
        line = save_format.format(frame=w_frame_id, id=w_track_id, x1=w_x1, y1=w_y1, x2=w_x2, y2=w_y2, w=w_wid, h=w_hgt)
        f.write(line)

def calculate_iou(box1, box2):
    # Calculate intersection over union between two boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / float(box1_area + box2_area - intersection)

def process_images(input_dir, output_dir):
    model = YOLO('yolov8l.pt')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracking variables
    next_id = 0  # Counter for generating unique IDs
    track_cnt = dict()  # Dictionary to store tracking information
    previous_boxes = []  # Store previous frame's boxes
    previous_ids = []   # Store previous frame's IDs
    total_unique_persons = 0  # Counter for unique persons
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)])
    
    # Create tracking results file
    filename = os.path.join(output_dir, 'tracking_results.txt')
    open(filename, 'w').close()

    for frame_cnt, image_file in enumerate(image_files):
        input_path = os.path.join(input_dir, image_file)
        frame = cv2.imread(input_path)
        
        if frame is None:
            print(f"Failed to read image: {image_file}")
            continue
            
        # Person detection using YOLO
        results = model(frame, classes=[0])
        
        current_boxes = []
        current_ids = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            # Process each detection
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                if x1 >= 0 and y1 >= 0 and y2 < frame.shape[0] and x2 < frame.shape[1]:
                    current_box = [x1, y1, x2, y2]
                    matched = False
                    
                    # Try to match with previous frame's boxes
                    if previous_boxes:
                        max_iou = 0
                        best_match_idx = -1
                        
                        for i, prev_box in enumerate(previous_boxes):
                            iou = calculate_iou(current_box, prev_box)
                            if iou > 0.5 and iou > max_iou:  # IOU threshold of 0.5
                                max_iou = iou
                                best_match_idx = i
                        
                        if best_match_idx >= 0:
                            track_id = previous_ids[best_match_idx]
                            matched = True
                    
                    # If no match found, assign new ID
                    if not matched:
                        track_id = next_id
                        next_id += 1
                        total_unique_persons += 1
                    
                    current_boxes.append(current_box)
                    current_ids.append(track_id)
                    
                    # Update tracking information
                    if track_id not in track_cnt:
                        track_cnt[track_id] = [[frame_cnt, x1, y1, x2, y2]]
                    else:
                        track_cnt[track_id].append([frame_cnt, x1, y1, x2, y2])
                    
                    # Draw boxes and write results
                    text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
                    cv2_addBox(
                        track_id,
                        frame,
                        x1, y1, x2, y2,
                        line_thickness,
                        text_thickness,
                        text_scale
                    )
                    
                    write_results(
                        filename,
                        'mot',
                        frame_cnt + 1,
                        str(track_id),
                        x1, y1, x2, y2,
                        frame.shape[1],
                        frame.shape[0]
                    )
        
        # Update previous frame information
        previous_boxes = current_boxes
        previous_ids = current_ids
        
        # Save processed image
        output_path = os.path.join(output_dir, f'processed_{image_file}')
        cv2.imwrite(output_path, frame)
        print(f"Processed and saved: {output_path}")
    
    print("Processing complete!")
    print(f"Total frames processed: {len(image_files)}")
    print(f"Total unique persons detected: {total_unique_persons}")
    
    # Save tracking information
    with open(os.path.join(output_dir, 'tracking_info.pkl'), 'wb') as f:
        pickle.dump(track_cnt, f)

def main():
    input_dir = 'input_images'
    output_dir = 'output_images'
    process_images(input_dir, output_dir)

if __name__ == "__main__":
    main()