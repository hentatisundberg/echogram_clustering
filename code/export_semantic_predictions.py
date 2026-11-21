#!/usr/bin/env python3

"""Run semantic-segmentation inference and export review data for Roboflow.

Each input image is copied to images/<name>.png and receives:
- masks/<name>_mask.png: indexed class-mask PNG
- masks/<name>_mask_color.png: colorized class-mask PNG
- overlays/<name>_overlay.png: mask overlaid on the source image

The indexed mask pixels use model class IDs, such as 0, 1, and 2. The generated
classes.txt, manifest.csv, and manifest.json files document the ID-to-name mapping.

Example:
    python3 code/export_semantic_predictions.py \
        --input data/batch2 \
        --output data/roboflow_review \
        --model runs/semantic/train/weights/best.pt \
        --zip
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for site_packages in (PROJECT_ROOT / ".venv" / "lib").glob("python*/site-packages"):
    if site_packages.is_dir():
        sys.path.insert(0, str(site_packages))

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODEL = PROJECT_ROOT / "runs/semantic/train/weights/best.pt"
CLASS_COLORS = [
    (0, 0, 0),
    (0, 255, 0),
    (255, 100, 0),
    (0, 150, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 0),
]


def find_images(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if input_dir.is_file():
        if input_dir.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported input image: {input_dir}")
        return [input_dir]

    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_class_names(model) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


def get_device(requested: str) -> str | int:
    if requested != "auto":
        return int(requested) if requested.isdigit() else requested
    return 0 if torch.cuda.is_available() else "cpu"


def extract_class_map(result, original_shape: tuple[int, int]) -> np.ndarray:
    height, width = original_shape

    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is not None:
        class_map = getattr(semantic_mask, "data", semantic_mask)
        if torch.is_tensor(class_map):
            class_map = class_map.detach().cpu().numpy()
        class_map = np.asarray(class_map)
        if class_map.ndim == 3:
            if class_map.shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected semantic mask shape: {class_map.shape}"
                )
            class_map = class_map[0]
        if class_map.ndim != 2:
            raise RuntimeError(
                f"Unexpected semantic mask shape: {class_map.shape}"
            )
    else:
        masks = getattr(result, "masks", None)
        if masks is None:
            return np.zeros((height, width), dtype=np.uint8)

        mask_data = getattr(masks, "data", masks)
        if torch.is_tensor(mask_data):
            mask_data = mask_data.detach().cpu().numpy()
        mask_data = np.asarray(mask_data)
        if mask_data.ndim != 3:
            raise RuntimeError(f"Unexpected mask tensor shape: {mask_data.shape}")

        boxes = getattr(result, "boxes", None)
        if boxes is not None and getattr(boxes, "cls", None) is not None:
            class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
        else:
            class_ids = np.zeros(mask_data.shape[0], dtype=np.int32)

        class_map = np.zeros((height, width), dtype=np.uint8)
        for mask, class_id in zip(mask_data, class_ids):
            binary_mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0.5
            class_map[binary_mask] = int(class_id)

    return cv2.resize(
        class_map.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )


def make_color_mask(class_map: np.ndarray) -> np.ndarray:
    height, width = class_map.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in np.unique(class_map):
        class_id = int(class_id)
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        color_mask[class_map == class_id] = color
    return color_mask


def make_overlay(image_bgr: np.ndarray, class_map: np.ndarray, alpha: float) -> np.ndarray:
    color_mask_rgb = make_color_mask(class_map)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    foreground = np.any(color_mask_bgr != 0, axis=2)
    overlay = image_bgr.copy()
    overlay[foreground] = cv2.addWeighted(
        image_bgr,
        1 - alpha,
        color_mask_bgr,
        alpha,
        0,
    )[foreground]
    return overlay


def output_stem(input_path: Path, input_dir: Path) -> str:
    if input_dir.is_file():
        relative = Path(input_path.stem)
    else:
        relative = input_path.relative_to(input_dir).with_suffix("")
    parts = [part for part in relative.parts if part not in ("", ".")]
    return "__".join(parts)


def write_review_outputs(
    model,
    class_names: dict[int, str],
    image_paths: list[Path],
    input_dir: Path,
    output_dir: Path,
    device: str | int,
    image_size: int,
    alpha: float,
    overwrite: bool,
) -> list[dict]:
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    for directory in (images_dir, masks_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for index, image_path in enumerate(image_paths, start=1):
        stem = output_stem(image_path, input_dir)
        image_output = images_dir / f"{stem}.png"
        mask_output = masks_dir / f"{stem}_mask.png"
        color_output = masks_dir / f"{stem}_mask_color.png"
        overlay_output = overlays_dir / f"{stem}_overlay.png"

        if not overwrite and all(
            path.exists()
            for path in (image_output, mask_output, color_output, overlay_output)
        ):
            print(f"[{index}/{len(image_paths)}] Skipping existing: {image_path.name}")
            continue

        pil_image = Image.open(image_path).convert("RGB")
        image_bgr = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)
        height, width = image_bgr.shape[:2]

        results = model.predict(
            source=str(image_path),
            imgsz=image_size,
            device=device,
            verbose=False,
        )
        result = results[0]
        class_map = extract_class_map(result, (height, width))

        pil_image.save(image_output, format="PNG")
        cv2.imwrite(str(mask_output), class_map)
        cv2.imwrite(
            str(color_output),
            cv2.cvtColor(make_color_mask(class_map), cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(overlay_output),
            make_overlay(image_bgr, class_map, alpha),
        )

        pixel_counts = {}
        for class_id in np.unique(class_map):
            class_id = int(class_id)
            pixel_counts[str(class_id)] = int(np.count_nonzero(class_map == class_id))

        record = {
            "source_image": str(image_path),
            "review_image": str(image_output),
            "mask": str(mask_output),
            "color_mask": str(color_output),
            "overlay": str(overlay_output),
            "width": width,
            "height": height,
            "pixel_counts": json.dumps(pixel_counts, sort_keys=True),
        }
        records.append(record)
        print(f"[{index}/{len(image_paths)}] Exported: {image_path.name}")

    return records


def write_metadata(
    output_dir: Path,
    class_names: dict[int, str],
    records: list[dict],
) -> None:
    max_class_id = max(class_names, default=-1)
    ordered_names = [
        class_names.get(class_id, f"class_{class_id}")
        for class_id in range(max_class_id + 1)
    ]
    (output_dir / "classes.txt").write_text(
        "\n".join(ordered_names) + ("\n" if ordered_names else ""),
        encoding="utf-8",
    )
    (output_dir / "class_mapping.json").write_text(
        json.dumps(
            {str(class_id): name for class_id, name in sorted(class_names.items())},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    (output_dir / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )

    if records:
        fieldnames = list(records[0].keys())
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        (output_dir / "manifest.csv").write_text(
            "source_image,review_image,mask,color_mask,overlay,width,height,pixel_counts\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export semantic-segmentation predictions for Roboflow annotation review."
    )
    parser.add_argument(
        "--input",
        "--image-dir",
        dest="input",
        type=Path,
        required=True,
        help="Input image or directory. Directories are searched recursively.",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        type=Path,
        default=PROJECT_ROOT / "data/roboflow_review",
        help="Directory for images, masks, overlays, and manifests.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Semantic-segmentation model weights.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, 0, 1, cuda:0, etc.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.40,
        help="Mask overlay opacity.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess and replace existing review outputs.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a ZIP archive of the output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_absolute():
        args.model = PROJECT_ROOT / args.model
    if not args.output.is_absolute():
        args.output = PROJECT_ROOT / args.output
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be between 0 and 1")

    image_paths = find_images(args.input)
    if not image_paths:
        raise RuntimeError(f"No supported images found in {args.input}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        image_paths = image_paths[: args.limit]

    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    args.output.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))
    device = get_device(args.device)
    class_names = get_class_names(model)

    print("=" * 60)
    print("SEMANTIC PREDICTIONS FOR ROBOFLOW REVIEW")
    print("=" * 60)
    print(f"Model:  {args.model}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}")
    print(f"Task:   {model.task}")
    print("Classes:")
    for class_id, name in sorted(class_names.items()):
        print(f"  {class_id}: {name}")
    print("=" * 60)

    records = write_review_outputs(
        model=model,
        class_names=class_names,
        image_paths=image_paths,
        input_dir=args.input,
        output_dir=args.output,
        device=device,
        image_size=args.imgsz,
        alpha=args.alpha,
        overwrite=args.overwrite,
    )
    write_metadata(args.output, class_names, records)

    print("\nDone.")
    print(f"Images:  {args.output / 'images'}")
    print(f"Masks:   {args.output / 'masks'}")
    print(f"Overlays:{args.output / 'overlays'}")

    if args.zip:
        archive = shutil.make_archive(
            str(args.output),
            "zip",
            root_dir=args.output,
        )
        print(f"ZIP:     {archive}")


if __name__ == "__main__":
    main()
