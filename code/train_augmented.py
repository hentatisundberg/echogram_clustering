
from pathlib import Path
import shutil
import yaml
import cv2
import numpy as np
import torch

from tqdm import tqdm
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_YAML = Path(
    "roboflow_download/echograms-2-3/data.yaml"
)

# If the YAML is actually directly inside echograms-2-3,
# change the path above to:
# "roboflow-download/echograms-2-3/data.yaml"

AUGMENTED_DATASET = Path("data/echogram_edge_augmented")

BASE_MODEL = "models/yolo26n-seg.pt"

IMAGE_FOLDER = Path("data/select1/")

# Training
IMAGE_SIZE = 1024
EPOCHS = 100
BATCH = 8
WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Edge augmentation
PAD_FRACTION = 0.20

# Inference
INFERENCE_PAD_FRACTION = 0.20
CONF = 0.25
IOU = 0.70

MASK_ALPHA = 0.25
DRAW_CONTOURS = True

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900

# Set True to prepare data and train.
# Set False to skip training and use an existing best.pt.
RUN_TRAINING = True

# Set True to inspect the predictions after training.
RUN_VIEWER = True

SEED = 42


# ============================================================
# 2. DATASET PATH RESOLUTION
# ============================================================

def resolve_dataset():

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Cannot find dataset YAML: {DATA_YAML.resolve()}"
        )

    with open(DATA_YAML, "r") as f:
        config = yaml.safe_load(f)

    yaml_dir = DATA_YAML.resolve().parent

    root = Path(config.get("path", yaml_dir))

    if not root.is_absolute():
        root = (yaml_dir / root).resolve()

    splits = {}

    for split in ["train", "val", "valid", "test"]:

        if split not in config:
            continue

        value = config[split]

        if isinstance(value, list):
            raise NotImplementedError(
                "Multiple image directories in one split "
                "are not supported by this script."
            )

        path = Path(value)

        if not path.is_absolute():
            path = root / path

        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {split} directory: {path}"
            )

        # Handle paths pointing directly to images/
        if path.name == "images":
            image_dir = path
        else:
            image_dir = path / "images"

        if not image_dir.exists():
            raise FileNotFoundError(
                f"Missing images directory: {image_dir}"
            )

        splits[split] = image_dir

    if "train" not in splits:
        raise ValueError("No training split found.")

    if "val" not in splits and "valid" not in splits:
        raise ValueError("No validation split found.")

    return config, splits


# ============================================================
# 3. TRANSFORM YOLO POLYGON LABELS
# ============================================================

def read_labels(label_path):

    instances = []

    if not label_path.exists():
        return instances

    with open(label_path, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) < 7:
                continue

            class_id = int(values[0])

            coords = np.asarray(
                values[1:],
                dtype=np.float32
            ).reshape(-1, 2)

            instances.append((class_id, coords))

    return instances


def write_labels(label_path, instances, width, height):

    with open(label_path, "w") as f:

        for class_id, polygon in instances:

            polygon = polygon.copy()

            polygon[:, 0] /= width
            polygon[:, 1] /= height

            polygon = np.clip(polygon, 0.0, 1.0)

            coords = " ".join(
                f"{v:.6f}"
                for v in polygon.flatten()
            )

            f.write(f"{class_id} {coords}\n")


def pad_labels(instances, width, height, pad):

    padded_instances = []

    for class_id, polygon in instances:

        # Original polygon, shifted to the right
        original = polygon.copy()
        original[:, 0] += pad

        padded_instances.append(
            (class_id, original)
        )

        # Reflected copy on the left
        left = polygon.copy()
        left[:, 0] = pad - left[:, 0]

        padded_instances.append(
            (class_id, left)
        )

        # Reflected copy on the right
        right = polygon.copy()
        right[:, 0] = pad + 2 * width - 1 - right[:, 0]

        padded_instances.append(
            (class_id, right)
        )

    return padded_instances


# ============================================================
# 4. CREATE EDGE-AUGMENTED DATASET
# ============================================================

def prepare_dataset():

    print("\n" + "=" * 60)
    print("PREPARING EDGE-AUGMENTED DATASET")
    print("=" * 60)

    config, splits = resolve_dataset()

    print("Original YAML:", DATA_YAML.resolve())
    print("Output:", AUGMENTED_DATASET.resolve())

    if AUGMENTED_DATASET.exists():
        print("\nOutput dataset already exists.")
        print("Delete it manually if you want to regenerate.")
        return AUGMENTED_DATASET / "data.yaml"

    AUGMENTED_DATASET.mkdir(
        parents=True,
        exist_ok=True
    )

    # Process original dataset splits
    for split, image_dir in splits.items():

        print(f"\nProcessing split: {split}")

        source_label_dir = (
            image_dir.parent / "labels"
        )

        output_image_dir = (
            AUGMENTED_DATASET / split / "images"
        )

        output_label_dir = (
            AUGMENTED_DATASET / split / "labels"
        )

        output_image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_label_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in [
                ".jpg", ".jpeg", ".png",
                ".tif", ".tiff"
            ]
        )

        print(f"Found {len(image_files)} images")

        for image_path in tqdm(
            image_files,
            desc=f"Copying {split}"
        ):

            label_path = (
                source_label_dir /
                (image_path.stem + ".txt")
            )

            # Copy original image and label
            shutil.copy2(
                image_path,
                output_image_dir / image_path.name
            )

            if label_path.exists():
                shutil.copy2(
                    label_path,
                    output_label_dir / label_path.name
                )
            else:
                # Empty label file for negative samples
                (output_label_dir /
                 (image_path.stem + ".txt")).touch()

            # Augment TRAIN only.
            # Validation and test remain original.
            if split != "train":
                continue

            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            h, w = image.shape[:2]
            pad = int(w * PAD_FRACTION)

            padded_image = cv2.copyMakeBorder(
                image,
                0, 0,
                pad, pad,
                cv2.BORDER_REFLECT_101
            )

            instances = read_labels(label_path)

            padded_instances = pad_labels(
                instances,
                w,
                h,
                pad
            )

            padded_image_path = (
                output_image_dir /
                (image_path.stem + "_pad" + image_path.suffix)
            )

            padded_label_path = (
                output_label_dir /
                (image_path.stem + "_pad.txt")
            )

            cv2.imwrite(
                str(padded_image_path),
                padded_image
            )

            write_labels(
                padded_label_path,
                padded_instances,
                padded_image.shape[1],
                padded_image.shape[0]
            )

    # Generate new YAML
    new_config = dict(config)

    new_config["path"] = str(
        AUGMENTED_DATASET.resolve()
    )

    for split in splits:
        new_config[split] = f"{split}/images"

    output_yaml = AUGMENTED_DATASET / "data.yaml"

    with open(output_yaml, "w") as f:
        yaml.safe_dump(
            new_config,
            f,
            sort_keys=False
        )

    print("\nDataset preparation complete!")
    print("New YAML:", output_yaml.resolve())

    return output_yaml


# ============================================================
# 5. TRAIN MODEL
# ============================================================

def train_model(dataset_yaml):

    print("\n" + "=" * 60)
    print("STARTING YOLO26 SEGMENTATION TRAINING")
    print("=" * 60)

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Batch size:", BATCH)
    print("Epochs:", EPOCHS)

    model = YOLO(BASE_MODEL)

    results = model.train(

        data=str(dataset_yaml),

        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH,
        device=DEVICE,

        project="runs/echogram_segmentation",
        name="edge_augmented_v1",

        patience=20,

        save=True,
        plots=True,

        # Preserve aspect ratios during batching
        rect=True,

        # Conservative augmentation
        degrees=0,
        translate=0.05,
        scale=0.20,
        fliplr=0.0,
        flipud=0.0,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.10,

        workers=WORKERS,
        seed=SEED,

        exist_ok=True
    )

    print("\nTraining completed!")

    return Path(
        "runs/echogram_segmentation/"
        "edge_augmented_v1/weights/best.pt"
    )


# ============================================================
# 6. INFERENCE WITH HORIZONTAL PADDING
# ============================================================

def predict_with_padding(model, image):

    h, w = image.shape[:2]

    pad = int(w * INFERENCE_PAD_FRACTION)

    padded = cv2.copyMakeBorder(
        image,
        0, 0,
        pad, pad,
        cv2.BORDER_REFLECT_101
    )

    result = model.predict(
        source=padded,
        conf=CONF,
        iou=IOU,
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        retina_masks=True,
        verbose=False
    )[0]

    if result.masks is None:
        return np.zeros((h, w), dtype=np.uint8)

    masks = result.masks.data.cpu().numpy()

    # Ultralytics returns masks at original padded-image
    # dimensions with retina_masks=True.
    if masks.shape[1:] != padded.shape[:2]:
        raise RuntimeError(
            f"Unexpected mask shape {masks.shape}. "
            f"Expected spatial shape {padded.shape[:2]}"
        )

    cropped_masks = masks[
        :,
        :,
        pad:pad + w
    ]

    combined = np.any(
        cropped_masks > 0.5,
        axis=0
    ).astype(np.uint8)

    return combined


# ============================================================
# 7. VISUALIZATION
# ============================================================

def create_overlay(image, mask):

    overlay = image.copy()

    overlay[mask > 0] = (0, 255, 0)

    annotated = cv2.addWeighted(
        image,
        1 - MASK_ALPHA,
        overlay,
        MASK_ALPHA,
        0
    )

    if DRAW_CONTOURS:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            annotated,
            contours,
            -1,
            (0, 255, 0),
            1
        )

    return annotated


# ============================================================
# 8. INTERACTIVE VIEWER
# ============================================================

def run_viewer(model_path):

    print("\n" + "=" * 60)
    print("STARTING INTERACTIVE VIEWER")
    print("=" * 60)

    model = YOLO(str(model_path))

    image_files = sorted(
        p for p in IMAGE_FOLDER.iterdir()
        if p.suffix.lower() in [
            ".jpg", ".jpeg", ".png",
            ".tif", ".tiff"
        ]
    )

    if not image_files:
        raise RuntimeError(
            f"No images found in {IMAGE_FOLDER.resolve()}"
        )

    index = 0

    while True:

        image_path = image_files[index]

        print(
            f"Processing {index + 1}/{len(image_files)}: "
            f"{image_path.name}"
        )

        image = cv2.imread(str(image_path))

        if image is None:
            index = (index + 1) % len(image_files)
            continue

        mask = predict_with_padding(
            model,
            image
        )

        annotated = create_overlay(
            image,
            mask
        )

        h, w = annotated.shape[:2]

        scale = min(
            SCREEN_WIDTH / w,
            SCREEN_HEIGHT / h,
            1.0
        )

        if scale < 1:

            annotated = cv2.resize(
                annotated,
                (
                    int(w * scale),
                    int(h * scale)
                ),
                interpolation=cv2.INTER_AREA
            )

        cv2.putText(
            annotated,
            f"{index + 1}/{len(image_files)} - {image_path.name}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        cv2.imshow(
            "Edge-Augmented YOLO Segmentation",
            annotated
        )

        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), 27]:
            break

        elif key in [ord("n"), 83, 3]:
            index = (index + 1) % len(image_files)

        elif key in [ord("p"), 81, 2]:
            index = (index - 1) % len(image_files)

    cv2.destroyAllWindows()


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":

    print("\nEDGE-AUGMENTED ECHOGRAM SEGMENTATION")
    print("Working directory:", Path.cwd())
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", DEVICE)

    best_model = Path(
        "runs/echogram_segmentation/"
        "edge_augmented_v1/weights/best.pt"
    )

    if RUN_TRAINING:

        dataset_yaml = prepare_dataset()

        best_model = train_model(
            dataset_yaml
        )

    if RUN_VIEWER:

        if not best_model.exists():
            raise FileNotFoundError(
                f"Trained model not found: {best_model}"
            )

        run_viewer(best_model)

    print("\nALL DONE!")
