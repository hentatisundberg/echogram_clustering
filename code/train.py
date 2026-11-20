

from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

# Empty cache of images (in shell...)
#find /home/jonas/Documents/vscode/Auklab_OD -name "*.cache" -type f -delete


#RUN EXAMPLE
"""
python3 code/train.py 2>&1 | tee data/yolo_train_$(date +%Y%m%d_%H%M%S).log
"""

# Load a COCO-pretrained YOLO model
model = YOLO("models/yolo26n-seg.pt")

# Specify dataset
prefix = "echogram_segment"
dataset_version = 1

# Complete path to the dataset YAML file
dataset_yaml = f"dataset/dataset_{prefix}_{dataset_version}.yaml"

# Train the model on the dataset
results = model.train(data=dataset_yaml, batch=8, epochs=100, imgsz=680, device = [0, 1])

# Save the model
model.save(f'models/auklab_yolo26n-seg_{prefix}_{dataset_version}.pt')
