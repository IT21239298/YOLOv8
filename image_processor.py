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

def process_images(input_dir, output_dir):
    model = YOLO('yolov8l.pt')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tracking variables
    track_cnt = dict()
    images_by_id = dict()
    ids_per_frame = []
    exist_ids = set()
    final_fuse_id = dict()
    frame_cnt = 0
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]
    
    # Create tracking results file
    filename = os.path.join(output_dir, 'tracking_results.txt')
    open(filename, 'w').close()

    for image_file in image_files:
        input_path = os.path.join(input_dir, image_file)
        frame = cv2.imread(input_path)
        
        if frame is None:
            print(f"Failed to read image: {image_file}")
            continue
            
        image = Image.fromarray(frame[..., ::-1])
        
        # Person detection using YOLO
        results = model(frame, classes=[0])
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            tmp_ids = []
            
            # Process each detection
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                
                if x1 >= 0 and y1 >= 0 and y2 < frame.shape[0] and x2 < frame.shape[1]:
                    track_id = frame_cnt * 1000 + i  # Generate unique ID
                    tmp_ids.append(track_id)
                    
                    if track_id not in track_cnt:
                        track_cnt[track_id] = [[frame_cnt, x1, y1, x2, y2, area]]
                        images_by_id[track_id] = [frame[y1:y2, x1:x2]]
                    else:
                        track_cnt[track_id].append([frame_cnt, x1, y1, x2, y2, area])
                        images_by_id[track_id].append(frame[y1:y2, x1:x2])
            
            ids_per_frame.append(set(tmp_ids))
            
            # Re-ID process similar to your reference code
            if len(exist_ids) == 0:
                for i in tmp_ids:
                    final_fuse_id[i] = [i]
                exist_ids = exist_ids.union(set(tmp_ids))
            else:
                new_ids = set(tmp_ids) - exist_ids
                for nid in new_ids:
                    final_fuse_id[nid] = [nid]
                    exist_ids.add(nid)
            
            # Draw boxes and write results
            text_scale, text_thickness, line_thickness = get_FrameLabels(frame)
            
            for box, track_id in zip(boxes, tmp_ids):
                x1, y1, x2, y2 = map(int, box)
                final_id = next((k for k, v in final_fuse_id.items() if track_id in v), track_id)
                
                cv2_addBox(
                    final_id,
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
                    str(final_id),
                    x1, y1, x2, y2,
                    frame.shape[1],
                    frame.shape[0]
                )
        
        # Save processed image
        output_path = os.path.join(output_dir, f'processed_{image_file}')
        cv2.imwrite(output_path, frame)
        print(f"Processed and saved: {output_path}")
        frame_cnt += 1
    
    print("Processing complete!")
    print(f"Total frames processed: {frame_cnt}")
    print(f"Total unique IDs: {len(final_fuse_id)}")

def main():
    input_dir = 'input_images'
    output_dir = 'output_images'
    process_images(input_dir, output_dir)

if __name__ == "__main__":
    main()