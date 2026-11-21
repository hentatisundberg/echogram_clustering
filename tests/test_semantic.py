
from ultralytics import YOLO
import torch
import numpy as np

model = YOLO("runs/semantic/train/weights/best.pt")

image = "data/select1/0001_SLUAquaSailor2020V2-Phase0-D20260403-T234115-0_raw_clean.png"

result = model.predict(
    image,
    imgsz=1024,
    device=0,
    verbose=False
)[0]

# This is the correct output!
class_map = result.semantic_mask.data.cpu().numpy()

print("Class map shape:", class_map.shape)
print("Data type:", class_map.dtype)
print("Unique classes:", np.unique(class_map))

print("\nCLASS DISTRIBUTION")

for class_id, name in model.names.items():

    pixels = np.sum(class_map == class_id)
    percentage = 100 * pixels / class_map.size

    print(
        f"{class_id}: {name:20s} "
        f"{pixels:>10,} pixels "
        f"({percentage:6.2f}%)"
    )
