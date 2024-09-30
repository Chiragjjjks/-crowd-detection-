import os
import torch
from ultralytics import YOLO
import cv2

# Path to your custom-trained YOLOv8 weights
model_path = 'C:/Users/Neha KB/Desktop/14K_DS/930.pt'

# Path to your validation images folder
validation_images_folder = 'C:/Users/Neha KB/Desktop/14K_DS/valid_s/images'

# Load the YOLOv8 model with custom weights
model = YOLO(model_path)

# Directory to save the detection label .txt files
labels_output_folder = 'C:/Users/Neha KB/Desktop/14K_DS/predicted_s/labels'
os.makedirs(labels_output_folder, exist_ok=True)

# Iterate over all images in the validation folder
for image_name in os.listdir(validation_images_folder):
    # Read each image
    image_path = os.path.join(validation_images_folder, image_name)
    image = cv2.imread(image_path)
    height, width, _ = image.shape  # Get the image dimensions
    
    # Run the YOLO model on the image
    results = model.predict(source=image_path, save=False, show=False)
    
    # Get the bounding boxes, class labels, and confidence scores
    bboxes = results[0].boxes.xyxy  # (x1, y1, x2, y2) bounding box format
    class_ids = results[0].boxes.cls  # class labels
    
    # Prepare the output label file for this image
    label_filename = image_name.replace('.jpg', '.txt')  # Ensure correct filename
    label_file_path = os.path.join(labels_output_folder, label_filename)
    
    # Open the label file for writing
    with open(label_file_path, 'w') as label_file:
        # Loop over all bounding boxes and class labels
        for i in range(len(bboxes)):
            bbox = bboxes[i].cpu().numpy()
            class_id = int(class_ids[i].cpu().item())

            # Convert (x1, y1, x2, y2) to YOLO format (x_center, y_center, width, height)
            x1, y1, x2, y2 = bbox
            box_width = x2 - x1
            box_height = y2 - y1
            x_center = x1 + box_width / 2
            y_center = y1 + box_height / 2
            
            # Normalize the values (scale to range 0-1)
            x_center /= width
            y_center /= height
            box_width /= width
            box_height /= height

            # Write the label in YOLO format: class_id x_center y_center width height
            label_file.write(f'{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n')
    
    print(f'Label file saved for {image_name}: {label_file_path}')

print(f'Detection and label generation completed! Check the label files in {labels_output_folder}.')

