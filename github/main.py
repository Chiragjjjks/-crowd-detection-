from ultralytics import YOLO
import torch

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()

# Initialize the model
model = YOLO('yolov8n.pt')  # Ensure this file is in your working directory or provide the full path

# Define training parameters
training_params = {
    'data': '/home/pirllabs/projectcrowd/data.yaml',  # Path to dataset YAML file
    'epochs': 50,  # Number of training epochs
    'batch': 8,  # Batch size
    'imgsz': 640,  # Image size (must be square)
    'lr0': 0.001,  # Initial learning rate
    'optimizer': 'Adam',  # Optimizer choice
    'project': 'crowd_detection_project',  # Directory for saving results
    'name': 'custom_experiment',  # Experiment name
    'exist_ok': True,  # Overwrite existing results
    'save_period': 10,  # Save model checkpoints every 10 epochs
    'cos_lr': True,  # Use a cosine learning rate scheduler
    'device': '0'  # Specify GPU device ID; set to 'cpu' if you want to use CPU
}

# Train the model
results = model.train(**training_params)

# Print training results
print("Training complete.")
print(f"Best model saved at: {results.best_model}")
print(f"Results saved at: {results.save_dir}")
