import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

# Load YOLOv8 model
model_path = 'C:/Users/Neha KB/Desktop/humanhead/best1.pt'
model = YOLO(model_path)  # Load the model using YOLO from Ultralytics

# Configuration
input_folder = 'C:/Users/Neha KB/Desktop/humanhead/res_crowd'
output_folder = 'C:/Users/Neha KB/Desktop/humanhead/output'
csv_file_path = os.path.join(output_folder, 'counts.csv')
window_size = 640  # Size of the sliding window
stride = 160       # Stride of the sliding window
confidence_threshold = 0.5  # Confidence threshold for detection
nms_threshold = 0.3 # NMS threshold

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create or clear the CSV file and write headers
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Image Name', 'Head Count', 'Person Count'])

def sliding_window(image, window_size, stride):
    windows = []
    (h, w) = image.shape[:2]
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            window = image[y:y + window_size, x:x + window_size]
            windows.append((x, y, window))
    return windows

def detect_objects(image):
    # Convert BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get detections
    results = model(rgb_image)  # Model inference
    
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confs = result.boxes.conf.cpu().numpy()  # Get confidences
        classes = result.boxes.cls.cpu().numpy()  # Get class IDs
        
        # Convert detections to dictionary format
        for box, conf, cls in zip(boxes, confs, classes):
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'confidence': conf,
                    'class': int(cls),
                    'counted': False  # Track if head has been counted
                })
    
    return detections

def apply_nms(detections, threshold):
    boxes = np.array([[d['xmin'], d['ymin'], d['xmax'], d['ymax']] for d in detections])
    confidences = np.array([d['confidence'] for d in detections])
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), confidence_threshold, threshold)
    if len(indices) > 0:
        indices = indices.flatten()
        filtered_detections = [detections[i] for i in indices]
    else:
        filtered_detections = []
    
    return filtered_detections

def draw_boxes(image, detections):
    for detection in detections:
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = detection['class']
        
        # Draw bounding box
        color = (0, 255, 0) if cls == 1 else (255, 0, 0)  # Green for persons, Red for heads
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw confidence
        label = f'{conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image

def is_head_near_person(head, person):
    hx1, hy1, hx2, hy2 = head['xmin'], head['ymin'], head['xmax'], head['ymax']
    px1, py1, px2, py2 = person['xmin'], person['ymin'], person['xmax'], person['ymax']
    
    # Check if the head is within the bounding box of the person
    if (hx1 >= px1 and hx2 <= px2 and hy1 >= py1 and hy2 <= py2):
        return True
    return False

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        # Apply sliding window
        windows = sliding_window(image, window_size, stride)
        
        # Detect objects in each window
        all_detections = []
        for (x, y, window) in windows:
            detections = detect_objects(window)
            
            # Adjust detection coordinates to the full image scale
            for det in detections:
                det['xmin'] += x
                det['ymin'] += y
                det['xmax'] += x
                det['ymax'] += y
            
            all_detections.extend(detections)
        
        # Apply Non-Maximum Suppression to reduce duplicate detections
        filtered_detections = apply_nms(all_detections, nms_threshold)
        
        # Count heads and persons separately
        head_detections = [det for det in filtered_detections if det['class'] == 0]
        person_detections = [det for det in filtered_detections if det['class'] == 1]
        
        # Ensure each detected person has an associated head count
        for person in person_detections:
            person_region = {
                'xmin': person['xmin'],
                'ymin': person['ymin'],
                'xmax': person['xmax'],
                'ymax': person['ymax']
            }
            head_found = False
            for head in head_detections:
                if is_head_near_person(head, person) and not head['counted']:
                    head_found = True
                    head['counted'] = True  # Mark this head as counted
            
            if not head_found:
                # If no head was found for the person, still count one head
                head_detections.append({
                    'xmin': person_region['xmin'],
                    'ymin': person_region['ymin'],
                    'xmax': person_region['xmax'],
                    'ymax': person_region['ymax'],
                    'confidence': 1.0,  # Assign high confidence for the artificial head
                    'class': 0,
                    'counted': True
                })
        
        # Filter out heads already counted
        head_detections = [head for head in head_detections if 'counted' not in head]
        
        # Combine all detections for final drawing
        final_detections = person_detections + head_detections
        
        # Draw bounding boxes on the original image
        image_with_boxes = draw_boxes(image.copy(), final_detections)
        
        # Save results
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_detected.jpg")
        cv2.imwrite(output_path, image_with_boxes)

        # Write counts to CSV
        head_count = len(head_detections)
        person_count = len(person_detections)
        
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([filename, head_count, person_count])

        # Print counts for debugging
        print(f"Processed {filename} - Heads: {head_count}, Persons: {person_count}")
