from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model_path = 'C:/Users/Neha KB/Desktop/humanhead/best1.pt'
model = YOLO(model_path)  # Load the model using YOLO from Ultralytics

# Configuration
input_folder = 'C:/Users/Neha KB/Desktop/humanhead/res_crowd'
output_folder = 'C:/Users/Neha KB/Desktop/humanhead/outputshow'
window_size = 640  # Size of the sliding window
stride = 320       # Stride of the sliding window

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
            x1, y1, x2, y2 = box.astype(int)
            detections.append({
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
                'confidence': conf,
                'class': int(cls)
            })
    
    return detections

def draw_boxes(image, detections):
    for detection in detections:
        x1, y1, x2, y2 = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        conf = detection['confidence']
        cls = int(detection['class'])
        label = f'{model.names[cls]} {conf:.2f}'
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

# Process each image in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        
        # Apply sliding window
        windows = sliding_window(image, window_size, stride)
        
        # Detect objects in each window
        for (x, y, window) in windows:
            detections = detect_objects(window)
            
            # Draw bounding boxes on the window
            window_with_boxes = draw_boxes(window.copy(), detections)
            
            # Save results
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_detected.jpg")
            cv2.imwrite(output_path, window_with_boxes)

        print(f"Processed {filename}")