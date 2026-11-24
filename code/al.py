#!/usr/bin/env python3
"""Active-learning loop for echogram semantic segmentation.

    init       create a workspace (and optionally import the Roboflow dataset)
    select     run the current model on unlabelled pool images, queue the most uncertain
    serve      start the browser annotator (use through an SSH tunnel)
    train      fine-tune on everything labelled so far -> new current model
    repredict  refresh predictions of not-yet-annotated queue images with the current model
    status     where am I in the loop, and what should I do next?
    log        print the full activity log
    pool       list / add directories of candidate (unlabelled) images

Typical cycle (on the remote, inside tmux):

    python code/al.py init --model runs/semantic/train/weights/best.pt \\
        --pool data/all --roboflow roboflow_download/echograms-2-3
    python code/al.py select --n 20
    python code/al.py serve                # annotate in the browser
    python code/al.py train --epochs 60
    python code/al.py select --n 20        # next round ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from al_core import (  # noqa: E402
    Workspace,
    import_roboflow,
    init_workspace,
    now_iso,
    refresh_predictions,
    resolve_path,
    select_images,
    train_round,
)

DEFAULT_WS = "data/al"


def cmd_init(args):
    import yaml

    if args.model:
        from ultralytics import YOLO

        from al_core import model_class_names

        model = YOLO(str(resolve_path(args.model)))
        if model.task != "semantic":
            raise SystemExit(f"{args.model} is a '{model.task}' model, expected 'semantic'.")
        classes = model_class_names(model)
    elif args.roboflow:
        cfg = yaml.safe_load((resolve_path(args.roboflow) / "data.yaml").read_text())
        names = cfg["names"]
        names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
        classes = {i: n for i, n in enumerate(names)}
        classes[len(names)] = "background"  # same convention as ultralytics polygon->semantic
    else:
        raise SystemExit("Need --model or --roboflow to determine class names.")

    ws = init_workspace(
        resolve_path(args.workspace),
        classes,
        [resolve_path(p) for p in args.pool],
        resolve_path(args.model) if args.model else None,
        args.imgsz,
        args.val_fraction,
    )
    print(f"Workspace: {ws.root}")
    print("Classes:", ws.classes)
    n = 0
    if args.roboflow:
        n = import_roboflow(ws, args.roboflow)
        print(f"Imported {n} labelled images from {args.roboflow}")
        # Masks changed after this moment are your edits (used by `status`).
        ws.config["labels_baseline"] = now_iso()
        ws.save_config()
    ws.log("init", model=ws.config.get("current_model"), pool=ws.config["pool"], imported=n)
    cmd_status(args)


def cmd_select(args):
    ws = Workspace.load(args.workspace)
    model = resolve_path(args.model) if args.model else ws.current_model
    if model is None or not model.exists():
        raise SystemExit(f"Model not found: {model}. Pass --model or train first.")
    rows = select_images(
        ws, model, args.n, args.strategy, args.sample, args.device, args.seed, args.random_fraction
    )
    print(f"\nQueued {len(rows)} images for annotation:")
    for r in rows:
        print(f"  {r['score']}  {r['stem']}")
    print("\nNext: python code/al.py serve")


def cmd_repredict(args):
    ws = Workspace.load(args.workspace)
    model = resolve_path(args.model) if args.model else ws.current_model
    n = refresh_predictions(ws, model, args.device)
    print(f"Re-predicted {n} queued images with {model}")


def cmd_serve(args):
    from al_server import serve

    ws = Workspace.load(args.workspace)
    serve(ws, args.host, args.port)


def cmd_train(args):
    ws = Workspace.load(args.workspace)
    extra = {}
    if args.freeze is not None:
        extra["freeze"] = args.freeze
    if args.lr0 is not None:
        extra["lr0"] = args.lr0
    best = train_round(ws, args.epochs, args.batch, args.device, args.init, args.patience, args.workers, extra)
    print(f"\nNew current model: {best}")
    if args.repredict:
        n = refresh_predictions(ws, best, args.device if "," not in args.device else args.device.split(",")[0])
        print(f"Re-predicted {n} queued images.")
    print("Next: python code/al.py select --n 20")


def cmd_status(args):
    from al_status import build_report, format_report

    ws = Workspace.load(args.workspace)
    report = build_report(ws, scan_pool=not getattr(args, "no_pool", False))
    print(format_report(ws, report, recent=getattr(args, "recent", 8)))
    if getattr(args, "verbose", False):
        print("New labels not yet used by the current model:", ", ".join(report.new_labels) or "-")
        print("Edited labels not yet used by the current model:", ", ".join(report.edited_labels) or "-")


def cmd_log(args):
    from al_status import describe_event

    ws = Workspace.load(args.workspace)
    events = ws.read_log()
    if not events:
        print(f"No activity logged yet ({ws.log_path}).")
    for e in events[-args.n:] if args.n else events:
        print(f"{e.get('time', '?').replace('T', ' ')}  {describe_event(e)}")


def cmd_pool(args):
    from al_core import find_images, rel_to_project

    ws = Workspace.load(args.workspace)
    if args.add:
        for d in args.add:
            path = resolve_path(d)
            if not path.exists():
                raise SystemExit(f"Not found: {path}")
            rel = str(rel_to_project(path))
            if rel not in ws.config["pool"]:
                ws.config["pool"].append(rel)
                ws.log("pool_add", dir=rel)
                print(f"Added {rel}")
        ws.save_config()
    for d in ws.config["pool"]:
        print(f"  {d:<40} {len(find_images([Path(d)]))} images")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workspace", "-w", default=DEFAULT_WS, help=f"Workspace dir (default {DEFAULT_WS})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="Create/update a workspace")
    s.add_argument("--model", help="Current semantic model weights (class names are read from it)")
    s.add_argument("--pool", nargs="+", default=["data/all"], help="Directories of unlabelled images")
    s.add_argument("--roboflow", help="Roboflow YOLO polygon dataset dir to import as initial labels")
    s.add_argument("--imgsz", type=int, default=1024)
    s.add_argument("--val-fraction", type=float, default=0.15)
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("select", help="Score the pool and queue images for annotation")
    s.add_argument("--n", type=int, default=20, help="Number of images to queue")
    s.add_argument("--strategy", choices=["uncertainty", "mixed", "random"], default="mixed",
                   help="mixed = mostly most-uncertain + some random (default)")
    s.add_argument("--random-fraction", type=float, default=0.25, help="Random share for --strategy mixed")
    s.add_argument("--sample", type=int, default=None, help="Only score a random subset of the pool")
    s.add_argument("--model", help="Override the workspace's current model")
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_select)

    s = sub.add_parser("serve", help="Run the browser annotator")
    s.add_argument("--host", default="127.0.0.1",
                   help="Bind address. Keep 127.0.0.1 and use an SSH tunnel (no auth built in).")
    s.add_argument("--port", type=int, default=8765)
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("train", help="Train a new round on all labelled data")
    s.add_argument("--epochs", type=int, default=60)
    s.add_argument("--batch", type=int, default=8)
    s.add_argument("--device", default="auto", help="auto, cpu, 0, or 0,1 for multi-GPU")
    s.add_argument("--init", default="current",
                   help="'current' = fine-tune the current model, or a weights path (e.g. models/yolo26n-sem.pt)")
    s.add_argument("--patience", type=int, default=30)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--freeze", type=int, default=None, help="Freeze first N layers")
    s.add_argument("--lr0", type=float, default=None)
    s.add_argument("--repredict", action="store_true", help="Refresh predictions for unannotated queue items")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("repredict", help="Refresh queue predictions with the current model")
    s.add_argument("--model")
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_repredict)

    s = sub.add_parser("status", help="Show progress per phase and advise the next step")
    s.add_argument("--no-pool", action="store_true", help="Skip scanning the pool dirs (faster)")
    s.add_argument("--recent", type=int, default=8, help="Number of recent activity lines")
    s.add_argument("--verbose", "-v", action="store_true", help="List the new/edited label names")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("log", help="Print the activity log")
    s.add_argument("--n", type=int, default=None, help="Only the last N events")
    s.set_defaults(func=cmd_log)

    s = sub.add_parser("pool", help="List pool dirs, or add new ones with --add")
    s.add_argument("--add", nargs="+", help="Directories of new candidate images")
    s.set_defaults(func=cmd_pool)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
