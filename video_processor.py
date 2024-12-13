from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime

class PersonTracker:
    def __init__(self, max_disappeared=30, similarity_threshold=0.7):
        self.next_id = 1
        self.persons = {}
        self.features_by_id = {}
        self.similarity_threshold = similarity_threshold
        self.active_persons = set()
        self.last_seen = {}
        self.confirmed_persons = set()
        # New variables for better tracking
        self.person_history = {}  # Store movement history
        self.min_frames_to_confirm = 5  # Minimum frames to confirm as unique person
        self.frame_counts = {}  # Track how many frames each ID has been seen
        
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
        """Enhanced matching with temporal consistency check"""
        best_match = None
        best_score = -1

        # Consider only recently seen persons
        recent_persons = {pid: feats for pid, feats in self.features_by_id.items() 
                         if frame_count - self.last_seen.get(pid, 0) < 50}

        for person_id, stored_features in recent_persons.items():
            # Compare features
            score = cv2.compareHist(features, np.mean(stored_features, axis=0), 
                                  cv2.HISTCMP_CORREL)
            
            # Add temporal consistency check
            if score > self.similarity_threshold and score > best_score:
                best_score = score
                best_match = person_id

        return best_match

    def update(self, frame, boxes, frame_count):
        """Updated tracking method with better unique person confirmation"""
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
                self.frame_counts[person_id] = 0
                self.person_history[person_id] = []

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
            self.frame_counts[person_id] = self.frame_counts.get(person_id, 0) + 1
            
            # Store position history
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            self.person_history[person_id].append((x_center, y_center))
            
            # Confirm as unique person if seen for enough frames
            if (self.frame_counts[person_id] >= self.min_frames_to_confirm and 
                person_id not in self.confirmed_persons):
                # Check if movement pattern suggests a new person
                if self._validate_unique_person(person_id):
                    self.confirmed_persons.add(person_id)

            current_detections.append((person_id, box))

        return current_detections
    
    def _validate_unique_person(self, person_id):
        """Validate if this is likely a unique person based on movement pattern"""
        if len(self.person_history[person_id]) < self.min_frames_to_confirm:
            return False
            
        # Calculate movement consistency
        positions = np.array(self.person_history[person_id])
        if len(positions) >= 2:
            # Calculate total distance moved
            distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
            total_distance = np.sum(distances)
            
            # If person has moved significantly, more likely to be a real detection
            return total_distance > 50  # Minimum movement threshold
            
        return False

    def get_counts(self):
        """Get current and total unique person counts"""
        # Only count persons as unique if they've been confirmed
        current_count = len(self.active_persons)
        total_unique = len(self.confirmed_persons)
        return current_count, total_unique

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