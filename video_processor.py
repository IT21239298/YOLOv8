from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

class PersonTracker:
    """
    Class to track and maintain consistent IDs for people across video frames.
    Uses appearance-based features for person re-identification.
    """
    def __init__(self, max_disappeared=30, similarity_threshold=0.7):
        # Initialize tracker variables
        self.next_id = 1  # Start IDs from 1
        self.persons = {}  # Dictionary to store person data
        self.features_by_id = {}  # Store feature history for each ID
        self.similarity_threshold = similarity_threshold  # Threshold for matching persons
        self.active_persons = set()  # Track persons in current frame
        self.total_unique = 0  # Counter for total unique persons
        self.last_seen = {}  # Track when each person was last seen
        self.confirmed_persons = set()  # Set of confirmed unique persons
        
    def get_features(self, frame, box):
        """
        Extract features from person detection using color histogram.
        Args:
            frame: Current video frame
            box: Bounding box coordinates [x1, y1, x2, y2]
        Returns:
            Normalized color histogram features or None if invalid box
        """
        x1, y1, x2, y2 = map(int, box)
        if x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0]:
            person_img = frame[y1:y2, x1:x2]
            person_img = cv2.resize(person_img, (64, 128))
            hist = cv2.calcHist([person_img], [0, 1, 2], None, [8, 8, 8], 
                              [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        return None

    def match_person(self, features, frame_count):
        """
        Match current detection with existing persons using feature similarity.
        Args:
            features: Extracted features from current detection
            frame_count: Current frame number for tracking last seen
        Returns:
            Best matching person ID or None if no match found
        """
        best_match = None
        best_score = -1

        # Only consider persons seen in the last 50 frames
        recent_persons = {pid: feats for pid, feats in self.features_by_id.items() 
                         if frame_count - self.last_seen.get(pid, 0) < 50}

        for person_id, stored_features in recent_persons.items():
            # Compare current features with stored feature history
            score = cv2.compareHist(features, np.mean(stored_features, axis=0), 
                                  cv2.HISTCMP_CORREL)
            if score > self.similarity_threshold and score > best_score:
                best_score = score
                best_match = person_id

        return best_match

    def update(self, frame, boxes, frame_count):
        """
        Update tracker with new detections in current frame.
        Args:
            frame: Current video frame
            boxes: List of detection bounding boxes
            frame_count: Current frame number
        Returns:
            List of tuples (person_id, box) for current detections
        """
        self.active_persons.clear()
        current_detections = []

        for box in boxes:
            features = self.get_features(frame, box)
            if features is None:
                continue

            person_id = self.match_person(features, frame_count)

            if person_id is None:
                # New person detected
                person_id = self.next_id
                self.next_id += 1
                self.features_by_id[person_id] = []
                self.total_unique += 1
                self.confirmed_persons.add(person_id)

            # Update feature history
            if person_id in self.features_by_id:
                self.features_by_id[person_id].append(features)
                if len(self.features_by_id[person_id]) > 30:
                    self.features_by_id[person_id].pop(0)
            else:
                self.features_by_id[person_id] = [features]

            # Update tracking info
            self.active_persons.add(person_id)
            self.last_seen[person_id] = frame_count
            current_detections.append((person_id, box))

        return current_detections

    def get_counts(self):
        """
        Get current person counts.
        Returns:
            Tuple of (current_count, total_unique)
        """
        return len(self.active_persons), len(self.confirmed_persons)

def get_color(idx):
    """
    Generate consistent color for visualization based on ID.
    Args:
        idx: Person ID
    Returns:
        BGR color tuple
    """
    idx = idx * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

def process_video(input_video_path, output_dir):
    """
    Process video file for person detection and tracking.
    Args:
        input_video_path: Path to input video file
        output_dir: Directory to save processed video
    """
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize YOLO model and person tracker
    model = YOLO('yolov8l.pt')
    person_tracker = PersonTracker()

    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Setup video writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'processed_video_{timestamp}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    # Process video frame by frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run person detection
        results = model(frame, classes=[0])  # class 0 is person

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            detections = person_tracker.update(frame, boxes, frame_count)
            
            # Draw detection boxes and IDs
            for person_id, box in detections:
                x1, y1, x2, y2 = map(int, box)
                color = get_color(person_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'Person {person_id}', (x1, y1 - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Get and display counts
        current_count, total_unique = person_tracker.get_counts()
        cv2.putText(frame, f'Frame: {frame_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Current Count: {current_count}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Total Unique: {total_unique}', (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # Release resources
    cap.release()
    out.release()

    # Print processing summary
    print(f"Video processing complete!")
    print(f"Total frames processed: {frame_count}")
    current_count, total_unique = person_tracker.get_counts()
    print(f"Total unique persons detected: {total_unique}")
    print(f"Output saved to: {output_path}")

    # Convert to MP4 format
    try:
        print("Converting to MP4 format...")
        mp4_output_path = output_path.replace('.avi', '.mp4')
        cap = cv2.VideoCapture(output_path)
        fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
        out_mp4 = cv2.VideoWriter(mp4_output_path, fourcc_mp4, fps, 
                                 (frame_width, frame_height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out_mp4.write(frame)
            
        cap.release()
        out_mp4.release()
        print(f"MP4 version saved to: {mp4_output_path}")
    except Exception as e:
        print(f"Could not convert to MP4: {e}")

def main():
    """Main function to handle video processing"""
    video_dir = 'input_videos'
    output_dir = 'output_videos'

    # Check input directory
    if not os.path.exists(video_dir):
        print(f"Input directory '{video_dir}' does not exist!")
        return

    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]

    if not videos:
        print("No video files found in input directory!")
        return

    # Process each video file
    for video_file in videos:
        print(f"\nProcessing video: {video_file}")
        input_path = os.path.join(video_dir, video_file)
        process_video(input_path, output_dir)

if __name__ == "__main__":
    main()