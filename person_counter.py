from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import pickle
import os

class PersonDatabase:
    def __init__(self, feature_db_path='person_features.pkl'):
        self.feature_db_path = feature_db_path
        self.person_features = self.load_database()
        self.next_id = max(self.person_features.keys()) + 1 if self.person_features else 0
        self.active_persons = set()  # Currently visible persons
        self.all_detected_ids = set()  # All IDs ever detected

    def load_database(self):
        if os.path.exists(self.feature_db_path):
            with open(self.feature_db_path, 'rb') as f:
                return pickle.load(f)
        return {}

    def save_database(self):
        with open(self.feature_db_path, 'wb') as f:
            pickle.dump(self.person_features, f)

    def extract_features(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        person_img = frame[y1:y2, x1:x2]
        if person_img.size == 0:
            return None
        
        # Resize for consistent feature extraction
        person_img = cv2.resize(person_img, (64, 128))
        
        # Calculate color histogram features
        hist = cv2.calcHist([person_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def match_person(self, features, threshold=0.7):
        if not self.person_features:
            return None

        best_match = None
        best_score = -1

        for person_id, stored_features in self.person_features.items():
            score = cv2.compareHist(features, stored_features, cv2.HISTCMP_CORREL)
            if score > threshold and score > best_score:
                best_score = score
                best_match = person_id

        return best_match

    def update_person(self, frame, box):
        features = self.extract_features(frame, box)
        if features is None:
            return None

        # Try to match with existing person
        person_id = self.match_person(features)

        # If no match found, create new person
        if person_id is None:
            person_id = self.next_id
            self.next_id += 1
            self.person_features[person_id] = features

        self.active_persons.add(person_id)
        self.all_detected_ids.add(person_id)
        return person_id

    def cleanup_frame(self):
        self.active_persons.clear()

def run_detection():
    model = YOLO('yolov8l.pt')
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or video path
    person_db = PersonDatabase()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 detection
        results = model(frame, classes=[0])  # class 0 is person

        # Clear active persons for new frame
        person_db.cleanup_frame()

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            # Process each detection
            for box in boxes:
                person_id = person_db.update_person(frame, box)
                if person_id is not None:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw current ID
                    label = f"Person {person_id}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display all previously detected IDs
        y_pos = 30
        cv2.putText(frame, "All Detected IDs:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos += 30
        id_text = ", ".join(f"ID-{id}" for id in sorted(person_db.all_detected_ids))
        cv2.putText(frame, id_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display active IDs
        y_pos += 30
        cv2.putText(frame, "Currently Active IDs:", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        active_id_text = ", ".join(f"ID-{id}" for id in sorted(person_db.active_persons))
        cv2.putText(frame, active_id_text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    person_db.save_database()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()