"""Core logic for the echogram active-learning loop.

Workspace layout (default: data/al):

    config.json               classes, pool dirs, current model, settings
    history.csv               one row per training round
    dataset/
        data.yaml             ultralytics semantic dataset (PNG masks)
        images/{train,val}/   training images
        masks/{train,val}/    uint8 PNG class-id masks, 255 = ignore
    queue/
        queue.csv             images selected for annotation + status
        images/<stem>.png     RGB copy of the source image
        pred/<stem>.png       predicted class map
        conf/<stem>.png       per-pixel confidence (max softmax * 255)
    runs/round_XX/            ultralytics training runs

Class ids follow the trained model: when it was trained on Roboflow polygons,
ultralytics appended a "background" class after the labelled classes, e.g.
0 near field, 1 seafloor, 2 surface turbidity, 3 background.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IGNORE_ID = 255
QUEUE_FIELDS = [
    "stem",
    "source",
    "round",
    "score",
    "mean_conf",
    "low_conf_frac",
    "status",  # todo | done | skipped
    "split",
]

_queue_lock = threading.Lock()
_log_lock = threading.Lock()


def now_iso() -> str:
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# Workspace
# ----------------------------------------------------------------------------


@dataclass
class Workspace:
    root: Path
    config: dict = field(default_factory=dict)

    # --- paths ---------------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def data_yaml(self) -> Path:
        return self.dataset_dir / "data.yaml"

    @property
    def queue_dir(self) -> Path:
        return self.root / "queue"

    @property
    def queue_csv(self) -> Path:
        return self.queue_dir / "queue.csv"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def history_csv(self) -> Path:
        return self.root / "history.csv"

    def image_dir(self, split: str) -> Path:
        return self.dataset_dir / "images" / split

    def mask_dir(self, split: str) -> Path:
        return self.dataset_dir / "masks" / split

    # --- config --------------------------------------------------------------
    @property
    def classes(self) -> dict[int, str]:
        return {int(k): v for k, v in self.config["classes"].items()}

    @property
    def current_model(self) -> Path | None:
        model = self.config.get("current_model")
        return resolve_path(model) if model else None

    @classmethod
    def load(cls, root: Path) -> "Workspace":
        root = resolve_path(root)
        ws = cls(root)
        if not ws.config_path.exists():
            raise SystemExit(
                f"No active-learning workspace at {root}. Run `python code/al.py init` first."
            )
        ws.config = json.loads(ws.config_path.read_text())
        return ws

    def save_config(self) -> None:
        atomic_write_text(self.config_path, json.dumps(self.config, indent=2) + "\n")

    def write_data_yaml(self) -> None:
        import yaml

        names = self.classes
        data = {
            "path": str(self.dataset_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "masks_dir": "masks",
            "nc": len(names),
            "names": {k: names[k] for k in sorted(names)},
        }
        self.data_yaml.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.data_yaml, yaml.safe_dump(data, sort_keys=False))

    def ensure_dirs(self) -> None:
        for split in ("train", "val"):
            self.image_dir(split).mkdir(parents=True, exist_ok=True)
            self.mask_dir(split).mkdir(parents=True, exist_ok=True)
        for sub in ("images", "pred", "conf"):
            (self.queue_dir / sub).mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    # --- dataset bookkeeping -------------------------------------------------
    def labeled_stems(self) -> dict[str, str]:
        """Return {stem: split} for every image that has a mask in the dataset."""
        out = {}
        for split in ("train", "val"):
            for mask in self.mask_dir(split).glob("*.png"):
                out[mask.stem] = split
        return out

    def find_label(self, stem: str) -> tuple[Path, Path, str] | None:
        """Return (image, mask, split) for a labelled stem, or None."""
        for split in ("train", "val"):
            mask = self.mask_dir(split) / f"{stem}.png"
            if mask.exists():
                image = find_image_with_stem(self.image_dir(split), stem)
                if image is not None:
                    return image, mask, split
        return None

    def split_for(self, stem: str) -> str:
        """Deterministic train/val assignment so an image never changes split."""
        frac = float(self.config.get("val_fraction", 0.15))
        h = int(hashlib.sha1(stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "val" if h < frac else "train"

    def save_label(self, stem: str, image_src: Path, mask: np.ndarray, split: str) -> Path:
        """Write a corrected mask (and its image) into the dataset."""
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split: {split}")
        # If the image was previously stored in the other split, move it.
        existing = self.find_label(stem)
        if existing is not None and existing[2] != split:
            old_img, old_mask, _ = existing
            old_mask.unlink(missing_ok=True)
            if old_img.resolve() != image_src.resolve():
                old_img.unlink(missing_ok=True)
            else:
                dst = self.image_dir(split) / old_img.name
                shutil.move(old_img, dst)
                image_src = dst

        img_dst = self.image_dir(split) / f"{stem}{image_src.suffix.lower()}"
        if image_src.resolve() != img_dst.resolve():
            shutil.copy2(image_src, img_dst)
        mask_dst = self.mask_dir(split) / f"{stem}.png"
        write_png_atomic(mask_dst, mask.astype(np.uint8))
        return mask_dst

    # --- queue ---------------------------------------------------------------
    def read_queue(self) -> list[dict]:
        if not self.queue_csv.exists():
            return []
        with self.queue_csv.open(newline="") as f:
            return list(csv.DictReader(f))

    def write_queue(self, rows: list[dict]) -> None:
        tmp = self.queue_csv.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=QUEUE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, self.queue_csv)

    def update_queue_item(self, stem: str, **changes) -> None:
        with _queue_lock:
            rows = self.read_queue()
            for row in rows:
                if row["stem"] == stem:
                    row.update({k: str(v) for k, v in changes.items()})
            self.write_queue(rows)

    # --- activity log ----------------------------------------------------------
    @property
    def log_path(self) -> Path:
        return self.root / "log.jsonl"

    def log(self, event: str, **details) -> None:
        """Append one event to log.jsonl (never raises: logging must not break the loop)."""
        rec = {"time": now_iso(), "event": event, **details}
        try:
            with _log_lock, self.log_path.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            print(f"Warning: could not write activity log: {e}")

    def read_log(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        out = []
        for line in self.log_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out


def init_workspace(
    root: Path,
    classes: dict[int, str],
    pool: list[Path],
    model: Path | None,
    imgsz: int,
    val_fraction: float,
) -> Workspace:
    root = resolve_path(root)
    ws = Workspace(root)
    if ws.config_path.exists():
        ws = Workspace.load(root)
        print(f"Workspace already exists at {root}; updating settings.")
    ws.config.update(
        {
            "classes": {str(k): v for k, v in sorted(classes.items())},
            "pool": [str(rel_to_project(p)) for p in pool],
            "imgsz": imgsz,
            "val_fraction": val_fraction,
        }
    )
    if model is not None:
        ws.config["current_model"] = str(rel_to_project(model))
    ws.config.setdefault("round", 0)
    ws.ensure_dirs()
    ws.save_config()
    ws.write_data_yaml()
    return ws


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def resolve_path(p: str | Path) -> Path:
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def rel_to_project(p: Path) -> Path:
    p = resolve_path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return p


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_png_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), array):
        raise RuntimeError(f"Could not write {path}")
    os.replace(tmp, path)


def find_images(dirs: list[Path]) -> list[Path]:
    out = []
    for d in dirs:
        d = resolve_path(d)
        if d.is_file() and d.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(d)
        elif d.is_dir():
            out.extend(
                p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        else:
            print(f"Warning: pool path not found: {d}")
    return sorted(out)


def find_image_with_stem(directory: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        p = directory / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_rgb(path: Path) -> np.ndarray:
    """Read any supported image as an RGB uint8 array (drops alpha)."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def get_device(requested: str = "auto"):
    import torch

    if requested != "auto":
        if "," in requested:
            return [int(x) for x in requested.split(",")]
        return int(requested) if requested.isdigit() else requested
    return 0 if torch.cuda.is_available() else "cpu"


def model_class_names(model) -> dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


# ----------------------------------------------------------------------------
# Importing the existing Roboflow polygon dataset
# ----------------------------------------------------------------------------


def import_roboflow(ws: Workspace, roboflow_dir: Path) -> int:
    """Rasterise YOLO polygon labels into PNG masks, exactly like ultralytics'
    PolygonSemanticDataset does (larger polygons first, uncovered = background)."""
    import yaml
    from ultralytics.data.utils import polygons2masks_overlap

    roboflow_dir = resolve_path(roboflow_dir)
    cfg = yaml.safe_load((roboflow_dir / "data.yaml").read_text())
    rf_names = cfg["names"]
    rf_names = rf_names if isinstance(rf_names, list) else [rf_names[k] for k in sorted(rf_names)]
    classes = ws.classes
    bg_ids = [k for k, v in classes.items() if v == "background"]
    if not bg_ids:
        raise SystemExit("Workspace classes have no 'background' class; cannot import polygons.")
    bg = bg_ids[0]
    for i, name in enumerate(rf_names):
        if classes.get(i) != name:
            raise SystemExit(
                f"Class mismatch: Roboflow class {i}={name!r} but workspace has {classes.get(i)!r}"
            )

    split_map = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    n = 0
    for rf_split, split in split_map.items():
        img_dir = roboflow_dir / rf_split / "images"
        lbl_dir = roboflow_dir / rf_split / "labels"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Skipping unreadable {img_path}")
                continue
            h, w = image.shape[:2]
            mask = np.full((h, w), bg, dtype=np.uint8)
            lbl = lbl_dir / f"{img_path.stem}.txt"
            cls_ids, polys = [], []
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    vals = line.split()
                    if len(vals) < 7:
                        continue
                    cls_ids.append(int(vals[0]))
                    polys.append(
                        np.asarray(vals[1:], dtype=np.float32).reshape(-1, 2)
                        * np.array([w, h], dtype=np.float32)
                    )
            if polys:
                inst, sorted_idx = polygons2masks_overlap((h, w), polys, downsample_ratio=1)
                fg = inst > 0
                cls_arr = np.asarray(cls_ids, dtype=np.int32)[sorted_idx]
                mask[fg] = cls_arr[inst[fg] - 1].astype(np.uint8)
            stem = img_path.stem
            shutil.copy2(img_path, ws.image_dir(split) / img_path.name)
            write_png_atomic(ws.mask_dir(split) / f"{stem}.png", mask)
            n += 1
    return n


# ----------------------------------------------------------------------------
# Inference with uncertainty
# ----------------------------------------------------------------------------


def make_uncertainty_predictor():
    """Build a SemanticSegmentationPredictor subclass that also returns softmax
    confidence. Defined lazily so importing this module doesn't import torch."""
    import torch
    import torch.nn.functional as F
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor
    from ultralytics.utils import ops

    class UncertaintyPredictor(SemanticSegmentationPredictor):
        def postprocess(self, preds, img, orig_imgs):
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            results = []
            for i, (pred, orig_img) in enumerate(zip(preds, orig_imgs)):
                img_path = self.batch[0][i] if isinstance(self.batch[0], list) else self.batch[0]
                if pred.ndim == 2:
                    raise RuntimeError("Model outputs a baked class map; need logits for uncertainty.")
                pred = pred[None].float()
                if pred.shape[2:] != img.shape[2:]:
                    pred = F.interpolate(pred, img.shape[2:], mode="bilinear", align_corners=False)
                logits = ops.scale_masks(pred, orig_img.shape[:2])[0]  # [nc, H, W]
                if logits.shape[0] == 1:
                    p = torch.sigmoid(logits)
                    probs = torch.cat([1 - p, p], 0)
                else:
                    probs = logits.softmax(0)
                conf, class_map = probs.max(0)
                ent = -(probs.clamp_min(1e-8).log() * probs).sum(0) / np.log(probs.shape[0])
                r = Results(orig_img, path=img_path, names=self.model.names, semantic_mask=class_map.to(torch.uint8))
                r.conf_map = conf.cpu().numpy()
                r.entropy_map = ent.cpu().numpy()
                results.append(r)
            return results

    return UncertaintyPredictor


@dataclass
class Prediction:
    class_map: np.ndarray  # uint8 [H, W]
    conf_map: np.ndarray  # float32 [H, W] in [0, 1]
    score: float  # mean normalised entropy (higher = more uncertain)
    mean_conf: float
    low_conf_frac: float


class Predictor:
    def __init__(self, model_path: Path, imgsz: int = 1024, device: str = "auto"):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        if self.model.task != "semantic":
            raise SystemExit(f"{model_path} is a '{self.model.task}' model, expected 'semantic'.")
        self.imgsz = imgsz
        self.device = get_device(device)
        self.predictor_cls = make_uncertainty_predictor()
        self.names = model_class_names(self.model)

    def __call__(self, image_rgb: np.ndarray, low_conf: float = 0.6) -> Prediction:
        bgr = np.ascontiguousarray(image_rgb[..., ::-1])
        r = self.model.predict(
            bgr,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            predictor=self.predictor_cls,
        )[0]
        class_map = r.semantic_mask
        class_map = getattr(class_map, "data", class_map)
        class_map = class_map.cpu().numpy().astype(np.uint8)
        conf = r.conf_map.astype(np.float32)
        return Prediction(
            class_map=class_map,
            conf_map=conf,
            score=float(r.entropy_map.mean()),
            mean_conf=float(conf.mean()),
            low_conf_frac=float((conf < low_conf).mean()),
        )


def unique_stem(path: Path, taken: set[str]) -> str:
    stem = path.stem
    if stem not in taken:
        return stem
    h = hashlib.sha1(str(path).encode()).hexdigest()[:6]
    return f"{stem}__{h}"


def select_images(
    ws: Workspace,
    model_path: Path,
    n: int,
    strategy: str = "uncertainty",
    sample: int | None = None,
    device: str = "auto",
    seed: int = 0,
    random_fraction: float = 0.25,
    pool_dirs: list[Path] | None = None,
) -> list[dict]:
    """Score unlabelled pool images and add the N most informative to the queue.

    pool_dirs: folders (or image files) to draw from; defaults to the workspace pool.
    """
    rng = random.Random(seed)
    labeled = set(ws.labeled_stems())
    queue = ws.read_queue()
    queued_sources = {resolve_path(r["source"]).resolve() for r in queue}
    queued_stems = {r["stem"] for r in queue}

    if pool_dirs is None:
        pool_dirs = [Path(d) for d in ws.config["pool"]]
    print("Pool: " + ", ".join(str(rel_to_project(d)) for d in pool_dirs))
    pool = [
        p
        for p in find_images(pool_dirs)
        if p.resolve() not in queued_sources and p.stem not in labeled
    ]
    print(f"Unlabelled pool images: {len(pool)}")
    if not pool:
        return []
    if sample and len(pool) > sample:
        pool = rng.sample(pool, sample)
        print(f"Scoring a random sample of {len(pool)}")

    if strategy == "random":
        candidates = rng.sample(pool, min(n, len(pool)))
        to_score = candidates
    else:
        to_score = pool

    predictor = Predictor(model_path, ws.config.get("imgsz", 1024), device)
    if predictor.names != ws.classes:
        print(f"WARNING: model classes {predictor.names} != workspace classes {ws.classes}")

    from tqdm import tqdm

    # Keep only scalar scores here; large pools would not fit in memory otherwise.
    scored = []
    for path in tqdm(to_score, desc="Scoring pool"):
        try:
            pred = predictor(read_rgb(path))
        except Exception as e:  # noqa: BLE001
            print(f"Skipping {path}: {e}")
            continue
        pred.class_map = pred.conf_map = None
        scored.append((path, pred))

    if strategy == "uncertainty":
        scored.sort(key=lambda t: t[1].score, reverse=True)
        chosen = scored[:n]
    elif strategy == "mixed":
        # Mostly the most uncertain, plus some random images to avoid
        # only ever looking at hard edge cases.
        scored.sort(key=lambda t: t[1].score, reverse=True)
        n_rand = int(round(n * random_fraction))
        chosen = scored[: n - n_rand]
        rest = scored[n - n_rand :]
        chosen += rng.sample(rest, min(n_rand, len(rest)))
    else:
        chosen = scored

    rnd = int(ws.config.get("round", 0))
    new_rows = []
    taken = labeled | queued_stems
    for path, _ in chosen:
        rgb = read_rgb(path)
        pred = predictor(rgb)
        stem = unique_stem(path, taken)
        taken.add(stem)
        write_png_atomic(ws.queue_dir / "images" / f"{stem}.png", rgb[..., ::-1])
        write_png_atomic(ws.queue_dir / "pred" / f"{stem}.png", pred.class_map)
        write_png_atomic(
            ws.queue_dir / "conf" / f"{stem}.png",
            (pred.conf_map * 255).clip(0, 255).astype(np.uint8),
        )
        new_rows.append(
            {
                "stem": stem,
                "source": str(rel_to_project(path)),
                "round": rnd,
                "score": f"{pred.score:.5f}",
                "mean_conf": f"{pred.mean_conf:.4f}",
                "low_conf_frac": f"{pred.low_conf_frac:.4f}",
                "status": "todo",
                "split": ws.split_for(stem),
            }
        )

    with _queue_lock:
        ws.write_queue(ws.read_queue() + new_rows)
    ws.log(
        "select",
        round=rnd,
        model=str(rel_to_project(model_path)),
        strategy=strategy,
        pool=[str(rel_to_project(d)) for d in pool_dirs],
        n_scored=len(scored),
        n_queued=len(new_rows),
        pool_remaining=len(pool) - len(new_rows) if not sample else None,
    )

    # Save all scores so you can inspect the distribution later.
    scores_csv = ws.queue_dir / f"scores_round{rnd:02d}.csv"
    with scores_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "score", "mean_conf", "low_conf_frac"])
        for path, pred in sorted(scored, key=lambda t: -t[1].score):
            w.writerow([path, f"{pred.score:.5f}", f"{pred.mean_conf:.4f}", f"{pred.low_conf_frac:.4f}"])
    return new_rows


def refresh_predictions(ws: Workspace, model_path: Path, device: str = "auto") -> int:
    """Re-run the (new) model on queue items that are still 'todo'."""
    rows = ws.read_queue()
    todo = [r for r in rows if r["status"] == "todo"]
    if not todo:
        return 0
    predictor = Predictor(model_path, ws.config.get("imgsz", 1024), device)
    from tqdm import tqdm

    for row in tqdm(todo, desc="Re-predicting queue"):
        stem = row["stem"]
        rgb = read_rgb(ws.queue_dir / "images" / f"{stem}.png")
        pred = predictor(rgb)
        write_png_atomic(ws.queue_dir / "pred" / f"{stem}.png", pred.class_map)
        write_png_atomic(
            ws.queue_dir / "conf" / f"{stem}.png",
            (pred.conf_map * 255).clip(0, 255).astype(np.uint8),
        )
        row.update(
            score=f"{pred.score:.5f}",
            mean_conf=f"{pred.mean_conf:.4f}",
            low_conf_frac=f"{pred.low_conf_frac:.4f}",
        )
    with _queue_lock:
        ws.write_queue(rows)
    ws.log("repredict", model=str(rel_to_project(model_path)), n=len(todo))
    return len(todo)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------


def train_round(
    ws: Workspace,
    epochs: int,
    batch: int,
    device: str,
    init: str,
    patience: int,
    workers: int,
    extra: dict | None = None,
) -> Path:
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless
    from ultralytics import YOLO

    labeled = ws.labeled_stems()
    n_train = sum(1 for s in labeled.values() if s == "train")
    n_val = sum(1 for s in labeled.values() if s == "val")
    if n_train == 0 or n_val == 0:
        raise SystemExit(f"Need at least one train and one val image (have {n_train}/{n_val}).")

    if init == "current":
        if ws.current_model is None or not ws.current_model.exists():
            raise SystemExit("No current model in workspace; use --init with a weights path.")
        weights = ws.current_model
    else:
        # Local path if it exists, otherwise let ultralytics resolve the name
        # (e.g. "yolo26n-sem.pt" downloads, "yolo26n-sem.yaml" builds from scratch).
        weights = resolve_path(init) if resolve_path(init).exists() else Path(init)

    rnd = int(ws.config.get("round", 0)) + 1
    name = f"round_{rnd:02d}"
    print(f"Training {name}: {n_train} train / {n_val} val images, init={weights}")

    ws.write_data_yaml()
    # Remember exactly which labels this round sees, so `status` can tell
    # which annotations are newer than the current model.
    run_dir = ws.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    atomic_write_text(
        run_dir / "labels.json",
        json.dumps({"round": rnd, "started": started, "init": str(weights), "labels": labeled}, indent=1),
    )
    ws.log("train_started", round=rnd, n_train=n_train, n_val=n_val, init=str(weights), epochs=epochs)
    model = YOLO(str(weights))
    model.train(
        data=str(ws.data_yaml),
        task="semantic",
        epochs=epochs,
        imgsz=ws.config.get("imgsz", 1024),
        batch=batch,
        device=get_device(device),
        workers=workers,
        patience=patience,
        project=str(ws.runs_dir),
        name=name,
        exist_ok=True,
        plots=True,
        **(extra or {}),
    )

    best = ws.runs_dir / name / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but {best} not found.")

    miou = best_metric(ws.runs_dir / name / "results.csv", "metrics/mIoU")
    new_file = not ws.history_csv.exists()
    with ws.history_csv.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["round", "n_train", "n_val", "init", "best_mIoU", "weights"])
        w.writerow([rnd, n_train, n_val, str(weights), f"{miou:.4f}" if miou is not None else "", str(rel_to_project(best))])

    ws.config["round"] = rnd
    ws.config["current_model"] = str(rel_to_project(best))
    ws.save_config()
    ws.log("train_finished", round=rnd, best_mIoU=miou, weights=str(rel_to_project(best)))
    return best


def best_metric(results_csv: Path, column: str) -> float | None:
    if not results_csv.exists():
        return None
    with results_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    key = next((k for k in (rows[0].keys() if rows else []) if k.strip() == column), None)
    if key is None:
        return None
    vals = [float(r[key]) for r in rows if r[key].strip()]
    return max(vals) if vals else None
