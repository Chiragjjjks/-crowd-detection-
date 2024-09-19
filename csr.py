import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model with the trained weights
model = YOLO('C:/Users/Neha KB/Desktop/humanhead/best1.pt')

# Define the directory containing images
image_folder = 'C:/Users/Neha KB/Desktop/humanhead/res_crowd/'

# List all files in the directory
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Set a confidence threshold
confidence_threshold = 0.5

for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)
    img = cv2.imread(img_path)
    
    # Perform inference
    results = model(img)

    # Process results
    detections = results[0]  # Get the first result from the list

    # Extract the bounding boxes, confidences, and class IDs
    boxes = detections.boxes.xyxy.numpy()  # Get bounding boxes in [x1, y1, x2, y2] format
    confidences = detections.boxes.conf.numpy()  # Confidence scores for each box
    class_ids = detections.boxes.cls.numpy()  # Class IDs for each detected object

    # Convert to DataFrame-like structure for easier handling
    detections_df = pd.DataFrame({
        'xmin': boxes[:, 0],
        'ymin': boxes[:, 1],
        'xmax': boxes[:, 2],
        'ymax': boxes[:, 3],
        'confidence': confidences,
        'class': class_ids
    })

    # Filter detections by confidence and class (assuming class 0 is for heads)
    filtered_detections = detections_df[detections_df['confidence'] > confidence_threshold]
    head_detections = filtered_detections[filtered_detections['class'] == 0]  # Filter by head class (0)
    head_count = len(head_detections)

    # Print the number of heads detected
    print(f'File: {image_file}, Total heads detected: {head_count}')

    # Optional: Draw bounding boxes on the image
    for _, row in head_detections.iterrows():
        x1, y1, x2, y2, conf, cls = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']), row['confidence'], row['class']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'Head {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Save the output image with bounding boxes
    output_path = os.path.join('C:/Users/Neha KB/Desktop/humanhead/outputnew/', image_file)
    cv2.imwrite(output_path, img)
