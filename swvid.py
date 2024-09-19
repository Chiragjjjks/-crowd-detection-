import cv2
import numpy as np
import os
import csv
from ultralytics import YOLO

# Load YOLOv8 model
model_path = 'C:/Users/Neha KB/Desktop/humanhead/best1.pt'
model = YOLO(model_path)

# Configuration
input_image_path = 'C:/Users/Neha KB/Desktop/humanhead/res_crowd/c4.jpeg'
output_video_path = 'C:/Users/Neha KB/Desktop/humanhead/outputvid/sliding_window_video.mp4'
csv_file_path = 'C:/Users/Neha KB/Desktop/humanhead/outputvid/box_counts.csv'
window_size = 640
stride = 160
confidence_threshold = 0.5
nms_threshold = 0.3
slowmo_factor = 5  # How much slower you want the video (e.g., 2 means 2x slower)


# Create output folder if it doesn't exist
output_folder = os.path.dirname(output_video_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create or clear the CSV file and write headers
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame Number', 'Window X', 'Window Y', 'Head Count', 'Person Count'])

def sliding_window(image, window_size, stride):
    windows = []
    (h, w) = image.shape[:2]

    # Add padding to ensure the window covers the entire image
    padded_image = cv2.copyMakeBorder(image, 0, window_size - (h % window_size), 0, window_size - (w % window_size), 
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))

    padded_h, padded_w = padded_image.shape[:2]

    # Process each window
    for y in range(0, padded_h - window_size + 1, stride):
        for x in range(0, padded_w - window_size + 1, stride):
            window = padded_image[y:y + window_size, x:x + window_size]
            windows.append((x, y, window))
    
    return windows

def detect_objects(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(rgb_image)
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
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
                    'counted': False
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

def is_head_near_person(head, person):
    hx1, hy1, hx2, hy2 = head['xmin'], head['ymin'], head['xmax'], head['ymax']
    px1, py1, px2, py2 = person['xmin'], person['ymin'], person['xmax'], person['ymax']
    return (hx1 >= px1 and hx2 <= px2 and hy1 >= py1 and hy2 <= py2)

def draw_boxes(image, detections, windows):
    for (x, y, _) in windows:
        cv2.rectangle(image, (x, y), (x + window_size, y + window_size), (0, 0, 255), 2)  # Red for windows
    
    for detection in detections:
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = detection['class']
        color = (0, 255, 0) if cls == 1 else (255, 0, 0)  # Green for persons, Red for heads
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f'{conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image

def process_windows(image, window_size, stride):
    windows = sliding_window(image, window_size, stride)
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10 / slowmo_factor, (image.shape[1], image.shape[0]))
    
    for frame_num, (x, y, window) in enumerate(windows):
        detections = detect_objects(window)
        for det in detections:
            det['xmin'] += x
            det['ymin'] += y
            det['xmax'] += x
            det['ymax'] += y
        
        filtered_detections = apply_nms(detections, nms_threshold)
        head_detections = [det for det in filtered_detections if det['class'] == 0]
        person_detections = [det for det in filtered_detections if det['class'] == 1]
        
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
                    head['counted'] = True
            if not head_found:
                head_detections.append({
                    'xmin': person_region['xmin'],
                    'ymin': person_region['ymin'],
                    'xmax': person_region['xmax'],
                    'ymax': person_region['ymax'],
                    'confidence': 1.0,
                    'class': 0,
                    'counted': True
                })
        
        head_detections = [head for head in head_detections if head.get('counted', False)]
        final_detections = person_detections + head_detections
        
        image_with_boxes = draw_boxes(image.copy(), final_detections, windows)
        for _ in range(slowmo_factor):  # Duplicate frames for slow motion
            video_writer.write(image_with_boxes)
        
        head_count = len(head_detections)
        person_count = len(person_detections)
        with open(csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([frame_num, x, y, head_count, person_count])
    
    video_writer.release()

# Process the image and create the video
if os.path.exists(input_image_path):
    image = cv2.imread(input_image_path)
    process_windows(image, window_size, stride)
else:
    print(f"Image not found at {input_image_path}")
