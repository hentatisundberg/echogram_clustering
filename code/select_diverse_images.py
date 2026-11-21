#!/usr/bin/env python3


"""
RUN EXAMPLE: 
python3 code/select_diverse_images.py --input data/all --output data/batch2 --n 100 --method fps
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import KMeans

ImageFile.LOAD_TRUNCATED_IMAGES = True

# --------------------------------------------------
# Settings
# --------------------------------------------------

MODEL_NAME = "facebook/dinov2-small"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


# --------------------------------------------------
# Find images
# --------------------------------------------------

def find_images(input_dir):
    input_dir = Path(input_dir)

    files = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    return sorted(files)


# --------------------------------------------------
# Extract image embeddings using DINOv2
# --------------------------------------------------

def extract_embeddings(images, batch_size=32):

    device = get_device()
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    embeddings = []
    valid_images = []

    for start in tqdm(range(0, len(images), batch_size),
                      desc="Extracting embeddings"):

        batch_paths = images[start:start + batch_size]
        batch_pil = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_pil.append(img)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        if not batch_pil:
            continue

        inputs = processor(
            images=batch_pil,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

            # CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :]

            # Normalize embeddings
            emb = torch.nn.functional.normalize(emb, dim=1)

        embeddings.append(emb.cpu().numpy())

        # Keep filenames corresponding to valid images
        valid_images.extend(batch_paths[:len(batch_pil)])

    embeddings = np.vstack(embeddings)

    return valid_images, embeddings


# --------------------------------------------------
# Select diverse images: farthest-point sampling
# --------------------------------------------------

def select_fps(embeddings, n_select):

    n_images = embeddings.shape[0]

    if n_select > n_images:
        raise ValueError("Requested more images than available.")

    # Embeddings are already L2 normalized
    embeddings = embeddings.astype(np.float32)

    # Start with image closest to the mean embedding
    mean_embedding = embeddings.mean(axis=0)

    first = np.argmax(embeddings @ mean_embedding)

    selected = [first]

    # Minimum distance to any selected image
    min_dist = 1 - embeddings @ embeddings[first]

    for _ in tqdm(range(n_select - 1), desc="Selecting diverse images"):

        # Select image farthest from all previously selected images
        next_idx = np.argmax(min_dist)

        selected.append(next_idx)

        # Update distance to nearest selected image
        new_dist = 1 - embeddings @ embeddings[next_idx]

        min_dist = np.minimum(min_dist, new_dist)

        # Prevent selecting the same image twice
        min_dist[selected] = -1

    return selected


# --------------------------------------------------
# Select images using K-means
# --------------------------------------------------

def select_kmeans(embeddings, n_select):

    print("Running K-means...")

    kmeans = KMeans(
        n_clusters=n_select,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(embeddings)

    selected = []

    for cluster in range(n_select):

        members = np.where(cluster_labels == cluster)[0]

        if len(members) == 0:
            continue

        center = kmeans.cluster_centers_[cluster]

        # Pick image closest to cluster center
        distances = np.linalg.norm(
            embeddings[members] - center,
            axis=1
        )

        selected.append(members[np.argmin(distances)])

    return selected, cluster_labels


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Select diverse images for annotation."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing images"
    )

    parser.add_argument(
        "--output",
        default="selected_images",
        help="Output folder"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of images to select"
    )

    parser.add_argument(
        "--method",
        choices=["fps", "kmeans"],
        default="fps",
        help="Selection method"
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Embedding batch size"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find images
    images = find_images(input_dir)

    print(f"Found {len(images)} images")

    if len(images) == 0:
        raise ValueError("No images found.")

    if args.n > len(images):
        raise ValueError("Requested more images than available.")

    # Extract embeddings
    images, embeddings = extract_embeddings(
        images,
        batch_size=args.batch
    )

    print(f"Successfully processed {len(images)} images")

    # Selection
    if args.method == "fps":

        selected = select_fps(
            embeddings,
            args.n
        )

        cluster_labels = np.full(len(images), -1)

    elif args.method == "kmeans":

        selected, cluster_labels = select_kmeans(
            embeddings,
            args.n
        )

    # Copy selected images and save metadata
    records = []

    for rank, idx in enumerate(selected, start=1):

        src = images[idx]

        # Avoid filename collisions
        dst_name = f"{rank:04d}_{src.name}"
        dst = output_dir / dst_name

        shutil.copy2(src, dst)

        records.append({
            "selection_rank": rank,
            "original_file": str(src),
            "selected_file": str(dst),
            "cluster": int(cluster_labels[idx])
                if args.method == "kmeans" else None
        })

    # Save CSV
    df = pd.DataFrame(records)

    csv_path = output_dir / "selection.csv"
    df.to_csv(csv_path, index=False)

    print("\nDone!")
    print(f"Selected {len(selected)} images")
    print(f"Output folder: {output_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()