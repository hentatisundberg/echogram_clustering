from ultralytics import YOLO

model = YOLO("runs/semantic/train/weights/best.pt")

print("Task:", model.task)
print("Model type:", type(model.model))
print("Names:", model.names)

print("\nModel architecture:")
print(model.model)