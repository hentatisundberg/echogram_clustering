

from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

# Empty cache of images (in shell...)
#find /home/jonas/Documents/vscode/Auklab_OD -name "*.cache" -type f -delete


#RUN EXAMPLE
"""
python3 code/train.py
"""

# Load a COCO-pretrained YOLO model
model = YOLO("models/yolo26n-sem.pt")

# Specify dataset
prefix = "echograms-2"
dataset_version = 3

# Complete path to the dataset YAML file
dataset_yaml = f"roboflow_download/{prefix}-{dataset_version}/data.yaml"

# Train the model on the dataset
results = model.train(data=dataset_yaml, batch=8, epochs=100, imgsz=1024, device = [0, 1])

# Save the model
model.save(f'models/auklab_yolo26n-sem_{prefix}_{dataset_version}.pt')
