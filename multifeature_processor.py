import cv2
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance
class MultiFeatureTracker:
   def __init__(self):
       # Initialize detectors
       self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
       self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
       self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
       # Initialize background subtractor for movement detection
       self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
       # Initialize tracker properties
       self.next_person_id = 0
       self.tracked_persons = {}
       self.max_disappeared = 30
       # Parameters for histogram comparison
       self.hist_bins = 32
       self.hist_range = [0, 180]
   def calculate_color_histogram(self, frame, bbox):
       """Calculate color histogram for the given bounding box region."""
       x, y, w, h = bbox
       roi = frame[y:y+h, x:x+w]
       # Convert ROI to HSV color space
       hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
       # Calculate histogram for hue channel
       hist = cv2.calcHist([hsv_roi], [0], None, [self.hist_bins], self.hist_range)
       cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
       return hist
   def detect_motion(self, frame):
       """Detect moving regions in the frame."""
       fg_mask = self.bg_subtractor.apply(frame)
       # Apply morphological operations to remove noise
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
       fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
       fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
       # Find contours of moving regions
       contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
       motion_regions = []
       for contour in contours:
           if cv2.contourArea(contour) > 500:  # Filter small movements
               x, y, w, h = cv2.boundingRect(contour)
               motion_regions.append((x, y, w, h))
       return motion_regions
   def detect_all_features(self, frame):
       """Detect persons using multiple methods."""
       gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       # Detect faces
       faces = self.face_cascade.detectMultiScale(
           gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
       # Detect full bodies
       bodies = self.body_cascade.detectMultiScale(
           gray, scaleFactor=1.1, minNeighbors=3, minSize=(80, 200))
       # Detect upper bodies
       upper_bodies = self.upper_body_cascade.detectMultiScale(
           gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))
       # Detect motion
       motion_regions = self.detect_motion(frame)
       # Combine all detections
       all_detections = []
       # Add faces with detection type
       for bbox in faces:
           all_detections.append((*bbox, 'face'))
       # Add bodies with detection type
       for bbox in bodies:
           all_detections.append((*bbox, 'body'))
       # Add upper bodies with detection type
       for bbox in upper_bodies:
           all_detections.append((*bbox, 'upper_body'))
       # Add motion regions with detection type
       for bbox in motion_regions:
           all_detections.append((*bbox, 'motion'))
       return all_detections
   def update(self, frame):
       # Detect all features in current frame
       current_detections = self.detect_all_features(frame)
       # If no detections are found
       if len(current_detections) == 0:
           for person_id in list(self.tracked_persons.keys()):
               self.tracked_persons[person_id]["disappeared"] += 1
               if self.tracked_persons[person_id]["disappeared"] > self.max_disappeared:
                   del self.tracked_persons[person_id]
           return self.tracked_persons
       # Calculate centroids for current detections
       current_centroids = np.array([[x + w//2, y + h//2] for (x, y, w, h, _) in current_detections])
       # If we're not tracking anyone, register all detections
       if len(self.tracked_persons) == 0:
           for i in range(len(current_detections)):
               self.register_person(frame, current_detections[i], current_centroids[i])
       # Otherwise, match current detections to existing persons
       else:
           person_ids = list(self.tracked_persons.keys())
           tracked_centroids = np.array([self.tracked_persons[person_id]["centroid"]
                                       for person_id in person_ids])
           # Compute distances between tracked centroids and current centroids
           D = euclidean_distances(tracked_centroids, current_centroids)
           # Calculate histogram similarities for overlapping regions
           H = np.zeros((len(tracked_centroids), len(current_centroids)))
           for i, person_id in enumerate(person_ids):
               hist1 = self.tracked_persons[person_id]["histogram"]
               for j, detection in enumerate(current_detections):
                   hist2 = self.calculate_color_histogram(frame, detection[:4])
                   H[i, j] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
           # Combine distance and histogram metrics
           similarity_matrix = D * (2 - H)  # Lower values indicate better matches
           rows = similarity_matrix.min(axis=1).argsort()
           cols = similarity_matrix.argmin(axis=1)[rows]
           used_rows = set()
           used_cols = set()
           for (row, col) in zip(rows, cols):
               if row in used_rows or col in used_cols:
                   continue
               if similarity_matrix[row, col] > 100:  # Adjusted threshold
                   continue
               person_id = person_ids[row]
               self.tracked_persons[person_id].update({
                   "bbox": current_detections[col][:4],
                   "centroid": current_centroids[col],
                   "detection_type": current_detections[col][4],
                   "disappeared": 0,
                   "histogram": self.calculate_color_histogram(frame, current_detections[col][:4])
               })
               used_rows.add(row)
               used_cols.add(col)
           # Register new detections
           unused_cols = set(range(0, similarity_matrix.shape[1])).difference(used_cols)
           for col in unused_cols:
               self.register_person(frame, current_detections[col], current_centroids[col])
       return self.tracked_persons
   def register_person(self, frame, detection, centroid):
       """Register a new person."""
       bbox = detection[:4]
       detection_type = detection[4]
       self.tracked_persons[self.next_person_id] = {
           "bbox": bbox,
           "centroid": centroid,
           "detection_type": detection_type,
           "disappeared": 0,
           "histogram": self.calculate_color_histogram(frame, bbox)
       }
       self.next_person_id += 1
def main():
   cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path
   tracker = MultiFeatureTracker()
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       persons = tracker.update(frame)
       # Draw tracking results
       for person_id, data in persons.items():
           if data["disappeared"] == 0:
               (x, y, w, h) = data["bbox"]
               # Different colors for different detection types
               color = {
                   'face': (0, 255, 0),      # Green
                   'body': (255, 0, 0),      # Blue
                   'upper_body': (0, 0, 255), # Red
                   'motion': (255, 255, 0)    # Cyan
               }.get(data["detection_type"], (255, 255, 255))
               cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
               cv2.putText(frame, f"ID: {person_id} ({data['detection_type']})",
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
       cv2.imshow("Multi-Feature Tracking", frame)
       if cv2.waitKey(1) & 0xFF == ord('q'):
           break
   cap.release()
   cv2.destroyAllWindows()
if __name__ == "__main__":
   main()