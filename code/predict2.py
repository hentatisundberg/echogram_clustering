
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
import torch

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "runs/segment/train-2/weights/best.pt"
IMAGE_FOLDER = "data/select1/"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 1024

# Visualization
MASK_ALPHA = 0.30       # Mask transparency
DRAW_CONTOURS = True
CONTOUR_THICKNESS = 1
SHOW_LABELS = True
SHOW_CONFIDENCE = False
FONT_SCALE = 0.1
FONT_THICKNESS = 1

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------

model = YOLO(MODEL_PATH)

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

image_dir = Path(IMAGE_FOLDER)

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
    image_paths.extend(image_dir.glob(ext))

image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images")

if not image_paths:
    raise RuntimeError("No images found!")

# --------------------------------------------------
# MASK VISUALIZATION
# --------------------------------------------------

def draw_masks(image, result):

    annotated = image.copy()

    if result.masks is None:
        return annotated

    masks = result.masks.data.cpu().numpy()

    boxes = result.boxes

    if boxes is not None:
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
    else:
        classes = np.zeros(len(masks), dtype=int)
        confidences = np.zeros(len(masks))

    # Generate colors for classes
    colors = [
        (0, 255, 0),    # Green
        (255, 100, 0),  # Blue-ish
        (0, 150, 255),  # Orange
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]

    h, w = image.shape[:2]

    for i, mask in enumerate(masks):

        # Resize mask to original image dimensions
        mask = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        binary_mask = mask > 0.5

        color = colors[classes[i] % len(colors)]

        # Transparent mask overlay
        overlay = annotated.copy()
        overlay[binary_mask] = color

        annotated = cv2.addWeighted(
            annotated,
            1 - MASK_ALPHA,
            overlay,
            MASK_ALPHA,
            0
        )

        # Thin mask contours
        if DRAW_CONTOURS:
            contours, _ = cv2.findContours(
                binary_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                annotated,
                contours,
                -1,
                color,
                CONTOUR_THICKNESS
            )

        # Small class label
        if SHOW_LABELS and np.any(binary_mask):

            ys, xs = np.where(binary_mask)

            x = int(xs.min())
            y = int(ys.min())

            label = model.names[classes[i]]

            if SHOW_CONFIDENCE:
                label += f" {confidences[i]:.2f}"

            cv2.putText(
                annotated,
                label,
                (x, max(y - 3, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                color,
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    return annotated


# --------------------------------------------------
# INTERACTIVE VIEWER
# --------------------------------------------------

index = 0

while True:

    image_path = image_paths[index]

    print(f"Processing {index+1}/{len(image_paths)}: {image_path.name}")

    # Load original image
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        index = (index + 1) % len(image_paths)
        continue

    # Run prediction
    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=device,
        retina_masks=True,
        verbose=False
    )

    result = results[0]

    # Draw masks only
    annotated = draw_masks(image, result)

    # Resize for screen display
    h, w = annotated.shape[:2]

    scale = min(
        SCREEN_WIDTH / w,
        SCREEN_HEIGHT / h,
        1.0
    )

    if scale < 1:
        annotated = cv2.resize(
            annotated,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    # Small image information
    cv2.putText(
        annotated,
        f"{index+1}/{len(image_paths)} - {image_path.name}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    cv2.imshow("YOLO Segmentation Viewer", annotated)

    key = cv2.waitKey(0) & 0xFF

    if key in [ord('q'), 27]:
        break

    elif key in [ord('n'), 83, 3]:
        index = (index + 1) % len(image_paths)

    elif key in [ord('p'), 81, 2]:
        index = (index - 1) % len(image_paths)

cv2.destroyAllWindows()


