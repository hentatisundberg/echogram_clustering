
#!/usr/bin/env python3

"""
predict_semantic.py

Semantic segmentation prediction and visualization.

Supports:
    - Ultralytics-style Results objects containing semantic class masks
    - Per-class masks (one mask per class)
    - Instance masks with associated class IDs (fallback)
    - Class-map outputs, if exposed by the Results object

Usage:
    python predict_semantic.py
    python predict_semantic.py --show
    python predict_semantic.py --show --mode semantic
    python predict_semantic.py --mode masks
    python predict_semantic.py --debug

Keyboard controls when --show is enabled:
    n / right arrow : next image
    p / left arrow  : previous image
    q / ESC         : quit
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for site_packages in (PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"):
    if site_packages.is_dir():
        sys.path.insert(0, str(site_packages))

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = PROJECT_ROOT / "runs/semantic/train/weights/best.pt"
IMAGE_FOLDER = PROJECT_ROOT / "data/select1/"
OUTPUT_FOLDER = PROJECT_ROOT / "data/predictions_semantic/"

IMAGE_SIZE = 1024

# Semantic segmentation visualization
MASK_ALPHA = 0.40
DRAW_CONTOURS = True
CONTOUR_THICKNESS = 1
SHOW_LABELS = True
FONT_SCALE = 0.45
FONT_THICKNESS = 1

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1000

# Colors are BGR (OpenCV format)
CLASS_COLORS = [
    (0, 255, 0),       # Class 0: green
    (255, 100, 0),     # Class 1: blue
    (0, 150, 255),     # Class 2: orange
    (255, 0, 255),     # Class 3: magenta
    (0, 255, 255),     # Class 4: yellow
    (255, 255, 0),     # Class 5: cyan
    (255, 0, 0),       # Class 6: red
    (150, 0, 255),     # Class 7: purple
]

# If your class 0 is background, set this True.
# Background will not be overlaid or outlined.
BACKGROUND_CLASS = 0
IGNORE_BACKGROUND = True


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Semantic segmentation inference and visualization."
)

parser.add_argument(
    "--show",
    action="store_true",
    help="Open interactive image viewer."
)

parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path(OUTPUT_FOLDER),
    help="Output directory for annotated images."
)

parser.add_argument(
    "--mode",
    choices=["auto", "semantic", "masks"],
    default="auto",
    help=(
        "auto: automatically detect output format; "
        "semantic: interpret masks as semantic classes; "
        "masks: interpret masks as separate masks with class IDs."
    )
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="Print detailed model output information."
)

parser.add_argument(
    "--model",
    type=Path,
    default=MODEL_PATH,
    help="Path to the semantic segmentation weights.",
)

parser.add_argument(
    "--image-dir",
    type=Path,
    default=IMAGE_FOLDER,
    help="Directory containing images to predict.",
)

args = parser.parse_args()


# ============================================================
# INITIALIZE MODEL
# ============================================================

model = YOLO(str(args.model))

device = 0 if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("SEMANTIC SEGMENTATION PREDICTION")
print("=" * 60)
print(f"Model:       {args.model}")
print(f"Task:        {model.task}")
print(f"Classes:     {model.names}")
print(f"Device:     {device}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Mode:       {args.mode}")
print("=" * 60)


# ============================================================
# IMAGE COLLECTION
# ============================================================

image_dir = args.image_dir

image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
    image_paths.extend(image_dir.glob(ext))

image_paths = sorted(image_paths)

print(f"Found {len(image_paths)} images")

if not image_paths:
    raise RuntimeError(f"No images found in {image_dir}")


# ============================================================
# UTILITIES
# ============================================================

def get_class_name(class_id):
    """Return class name from model.names."""
    if isinstance(model.names, dict):
        return model.names.get(class_id, f"class_{class_id}")

    if isinstance(model.names, list):
        if class_id < len(model.names):
            return model.names[class_id]

    return f"class_{class_id}"


def get_class_color(class_id):
    """Get a class-specific BGR color."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def resize_class_map(class_map, width, height):
    """Resize categorical class map using nearest-neighbor interpolation."""
    return cv2.resize(
        class_map.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )


def print_result_debug(result):
    """Print information about the Results object."""

    print("\n--- RESULT DEBUG ---")
    print("Result type:", type(result))
    print("Result keys/attributes:")

    for attribute in [
        "masks",
        "boxes",
        "probs",
        "semantic_mask",
        "orig_shape",
    ]:
        value = getattr(result, attribute, None)

        print(f"  {attribute}: {type(value)}")

        if value is not None:
            if hasattr(value, "shape"):
                print("    shape:", value.shape)

            elif hasattr(value, "data"):
                print("    data shape:", value.data.shape)

    if getattr(result, "masks", None) is not None:
        print("Mask tensor shape:", result.masks.data.shape)

    if getattr(result, "boxes", None) is not None:
        print("Box tensor shape:", result.boxes.data.shape)
        print("Classes:", result.boxes.cls)
        print("Confidences:", result.boxes.conf)

    print("-------------------\n")


# ============================================================
# EXTRACT SEMANTIC PREDICTION
# ============================================================


def extract_class_map(result, image_shape=None):
    """Extract a 2-D class map from semantic or instance-style results."""
    if isinstance(result, (list, tuple)):
        if not result:
            return None
        result = result[0]

    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is not None:
        class_map = getattr(semantic_mask, "data", semantic_mask)
        if torch.is_tensor(class_map):
            class_map = class_map.detach().cpu().numpy()
        class_map = np.asarray(class_map)
        if class_map.ndim == 3 and class_map.shape[0] == 1:
            class_map = class_map[0]
        if class_map.ndim != 2:
            raise RuntimeError(
                f"Unexpected semantic mask shape: {class_map.shape}"
            )
        return class_map.astype(np.uint8)

    masks = getattr(result, "masks", None)
    if masks is None:
        if args.mode in ("semantic", "masks"):
            print("No segmentation predictions returned!")
        return None

    if args.mode == "semantic" or args.mode == "auto":
        class_map_attr = getattr(result, "class_map", None)
        if class_map_attr is not None:
            class_map = getattr(class_map_attr, "data", class_map_attr)
            if torch.is_tensor(class_map):
                class_map = class_map.detach().cpu().numpy()
            return np.asarray(class_map).astype(np.uint8)

    mask_data = getattr(masks, "data", masks)
    if torch.is_tensor(mask_data):
        mask_data = mask_data.detach().cpu().numpy()
    mask_data = np.asarray(mask_data)

    if mask_data.ndim == 2:
        return (mask_data > 0.5).astype(np.uint8)
    if mask_data.ndim != 3:
        raise RuntimeError(f"Unexpected mask tensor shape: {mask_data.shape}")

    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "cls", None) is not None:
        class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    else:
        class_ids = np.zeros(mask_data.shape[0], dtype=int)

    if image_shape is not None:
        h, w = image_shape[:2]
    else:
        h, w = mask_data.shape[-2:]

    class_map = np.zeros((h, w), dtype=np.uint8)
    for mask, class_id in zip(mask_data, class_ids):
        binary_mask = cv2.resize(
            mask.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0.5
        class_map[binary_mask] = int(class_id)

    return class_map

# ============================================================
# VISUALIZATION
# ============================================================

def draw_semantic_prediction(image, class_map):

    annotated = image.copy()

    if class_map is None:
        return annotated

    h, w = image.shape[:2]

    class_map = resize_class_map(
        class_map,
        w,
        h
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = image.copy()

    unique_classes = np.unique(class_map)

    for class_id in unique_classes:

        class_id = int(class_id)

        if IGNORE_BACKGROUND and class_id == BACKGROUND_CLASS:
            continue

        binary_mask = class_map == class_id

        if not np.any(binary_mask):
            continue

        color = get_class_color(class_id)

        overlay[binary_mask] = color

    annotated = cv2.addWeighted(
        image,
        1 - MASK_ALPHA,
        overlay,
        MASK_ALPHA,
        0
    )


    # --------------------------------------------------------
    # Draw contours
    # --------------------------------------------------------

    if DRAW_CONTOURS:

        for class_id in unique_classes:

            class_id = int(class_id)

            if IGNORE_BACKGROUND and class_id == BACKGROUND_CLASS:
                continue

            binary_mask = (
                class_map == class_id
            ).astype(np.uint8)

            if not np.any(binary_mask):
                continue

            contours, _ = cv2.findContours(
                binary_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(
                annotated,
                contours,
                -1,
                get_class_color(class_id),
                CONTOUR_THICKNESS
            )


    # --------------------------------------------------------
    # Class labels
    # --------------------------------------------------------

    if SHOW_LABELS:

        for class_id in unique_classes:

            class_id = int(class_id)

            if IGNORE_BACKGROUND and class_id == BACKGROUND_CLASS:
                continue

            binary_mask = class_map == class_id

            if not np.any(binary_mask):
                continue

            ys, xs = np.where(binary_mask)

            x = int(xs.min())
            y = int(ys.min())

            label = get_class_name(class_id)

            cv2.putText(
                annotated,
                label,
                (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                get_class_color(class_id),
                FONT_THICKNESS,
                cv2.LINE_AA
            )

    return annotated


# ============================================================
# INTERACTIVE PREDICTION
# ============================================================

args.output_dir.mkdir(
    parents=True,
    exist_ok=True
)

index = 0

while index < len(image_paths):

    image_path = image_paths[index]

    print(
        f"\nProcessing {index + 1}/{len(image_paths)}: "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        print("Could not read image:", image_path)

        index += 1

        continue


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        device=device,
        verbose=False
    )

    # model.predict() returns a list; take the first result
    result = results[0]

    class_map = extract_class_map(
        result,
        image.shape
    )

    if args.debug:
        print_result_debug(result)


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    annotated = draw_semantic_prediction(
        image,
        class_map
    )


    # --------------------------------------------------------
    # Image information
    # --------------------------------------------------------

    status = (
        "SEMANTIC"
        if class_map is not None
        else "NO SEGMENTATION"
    )

    cv2.putText(
        annotated,
        f"{index + 1}/{len(image_paths)} | {status} | {image_path.name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0) if class_map is not None else (0, 0, 255),
        1,
        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = (
        args.output_dir
        / f"{image_path.stem}_semantic.png"
    )

    success = cv2.imwrite(
        str(output_path),
        annotated
    )

    if not success:
        raise RuntimeError(
            f"Could not save image: {output_path}"
        )

    print("Saved:", output_path)


    # --------------------------------------------------------
    # Interactive viewer
    # --------------------------------------------------------

    if args.show:

        display = annotated.copy()

        h, w = display.shape[:2]

        scale = min(
            SCREEN_WIDTH / w,
            SCREEN_HEIGHT / h,
            1.0
        )

        if scale < 1:

            display = cv2.resize(
                display,
                (
                    int(w * scale),
                    int(h * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        cv2.imshow(
            "Semantic Segmentation Viewer",
            display
        )

        key = cv2.waitKey(0) & 0xFF

        # Quit
        if key in [ord("q"), 27]:

            break

        # Previous image
        elif key in [
            ord("p"),
            81,
            2,
            2424832
        ]:

            index = max(
                index - 1,
                0
            )

            continue

        # Next image
        elif key in [
            ord("n"),
            83,
            3,
            2555904
        ]:

            index += 1

            continue


    # --------------------------------------------------------
    # Next image
    # --------------------------------------------------------

    index += 1


# ============================================================
# CLEANUP
# ============================================================

if args.show:
    cv2.destroyAllWindows()

print("\nPrediction finished.")
