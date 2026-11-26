"""Status report and "what next?" advice for the active-learning loop.

Everything is derived from files in the workspace (masks, queue.csv, config.json,
manifests/round_XX.json, log.jsonl), so the report is correct even for work done
before the activity log existed.

The loop has five phases:

    1 Pool          candidate images that are neither labelled nor queued
    2 Select        score the pool with the current model, queue the most uncertain
    3 Pre-annotate  model predictions for queued images (made by `select`, refreshed by `repredict`)
    4 Annotate      check/correct predictions in the browser (`serve`)
    5 Retrain       fine-tune on all labels (`train`), giving a new current model
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from al_core import Workspace, find_images, rel_to_project, resolve_path

TRAIN_CMD = "python code/al.py train --epochs 60 --device 0,1 --repredict"
SUGGEST_TRAIN_AT = 10  # new labels before suggesting an early retrain


@dataclass
class Report:
    round: int = 0
    model: str | None = None
    model_exists: bool = False
    model_time: float | None = None
    model_basis: str = ""
    n_train: int = 0
    n_val: int = 0
    new_labels: list[str] = field(default_factory=list)  # labelled but unseen by current model
    edited_labels: list[str] = field(default_factory=list)  # seen, but mask changed since
    edits_approx: bool = False
    queue_counts: dict[str, int] = field(default_factory=dict)
    round_queue: dict[str, int] = field(default_factory=dict)  # queue items selected this round
    stale_preds: int = 0
    missing_preds: int = 0
    pool_dirs: list[str] = field(default_factory=list)
    pool_total: int | None = None
    pool_remaining: int | None = None
    pending_train: dict | None = None  # a round that started but has no history entry
    last_annotation: float | None = None
    events: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Building the report
# ----------------------------------------------------------------------------


def _mtime(p: Path) -> float | None:
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def _parse_time(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _import_baseline(mtimes: list[float], gap: float = 600.0) -> float | None:
    """Workspaces created before `labels_baseline` existed: the Roboflow import wrote
    all masks in one burst, so the end of the first burst (no >10 min gap) marks it."""
    if not mtimes:
        return None
    mtimes = sorted(mtimes)
    end = mtimes[0]
    for t in mtimes[1:]:
        if t - end > gap:
            break
        end = t
    return end + 1.0


def _manifest_for(ws: Workspace, rnd: int, model: Path | None) -> dict | None:
    """Label list written by train_round when round `rnd` started (manifests/round_XX.json;
    older code wrote runs/round_XX/labels.json, which ultralytics could delete)."""
    candidates = [ws.manifest_path(rnd)] if rnd > 0 else []
    if model is not None:
        candidates.append(model.parent.parent / "labels.json")
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
    return None


def _train_started(ws: Workspace, rnd: int) -> dict | None:
    """Last 'train_started' log event for round `rnd`."""
    hits = [e for e in ws.read_log() if e.get("event") == "train_started" and str(e.get("round")) == str(rnd)]
    return hits[-1] if hits else None


def build_report(ws: Workspace, scan_pool: bool = True) -> Report:
    r = Report()
    r.round = int(ws.config.get("round", 0))
    model = ws.current_model
    r.model = ws.config.get("current_model")
    r.model_exists = bool(model and model.exists())
    r.model_time = _mtime(model) if r.model_exists else None

    labeled = ws.labeled_stems()
    r.n_train = sum(s == "train" for s in labeled.values())
    r.n_val = sum(s == "val" for s in labeled.values())
    mask_time = {stem: _mtime(ws.mask_dir(split) / f"{stem}.png") or 0.0 for stem, split in labeled.items()}

    queue = ws.read_queue()
    queue_stems = {q["stem"] for q in queue}

    # --- which labels has the current model seen? -----------------------------
    manifest = _manifest_for(ws, r.round, model)
    started_ev = _train_started(ws, r.round) if manifest is None and r.round > 0 else None
    if manifest is None and started_ev is not None:
        # Round trained by this loop, but its label list is missing: the model saw
        # every mask that existed when training started (new and edited masks after
        # that can't be told apart, so they are all reported as new).
        started = (_parse_time(started_ev.get("time")) or 0.0) + 1.0
        seen = {s for s in labeled if mask_time[s] <= started}
        r.new_labels = sorted(set(labeled) - seen)
        r.edits_approx = True
        n_tr = sum(labeled[s] == "train" for s in seen)
        r.model_basis = (
            f"round {r.round}: {n_tr} train / {len(seen) - n_tr} val labels "
            "(label list reconstructed from file times)"
        )
    elif manifest is not None:
        seen = set(manifest.get("labels", {}))
        started = (_parse_time(manifest.get("started")) or 0.0) + 1.0  # ISO time has 1 s resolution
        r.new_labels = sorted(set(labeled) - seen)
        r.edited_labels = sorted(s for s in set(labeled) & seen if mask_time[s] > started)
        n_tr = sum(v == "train" for v in manifest["labels"].values())
        r.model_basis = f"round {manifest.get('round')}: {n_tr} train / {len(seen) - n_tr} val labels"
    else:
        # Initial model (trained outside the loop, e.g. on the Roboflow export).
        imported = [s for s in labeled if s not in queue_stems]
        r.new_labels = sorted(s for s in labeled if s in queue_stems)
        baseline = _parse_time(ws.config.get("labels_baseline"))
        if baseline is None:
            baseline = _import_baseline([mask_time[s] for s in imported])
            r.edits_approx = True
        if baseline is not None:
            r.edited_labels = sorted(s for s in imported if mask_time[s] > baseline)
        r.model_basis = f"initial model (not trained by this loop); {len(imported)} imported labels"

    # --- queue -----------------------------------------------------------------
    for q in queue:
        r.queue_counts[q["status"]] = r.queue_counts.get(q["status"], 0) + 1
        if str(q.get("round", "")) == str(r.round):
            r.round_queue[q["status"]] = r.round_queue.get(q["status"], 0) + 1
        if q["status"] == "todo":
            t = _mtime(ws.queue_dir / "pred" / f"{q['stem']}.png")
            if t is None:
                r.missing_preds += 1
            elif r.model_time and t < r.model_time:
                r.stale_preds += 1
    done_times = [mask_time[s] for s in labeled if s in queue_stems or s in r.edited_labels]
    r.last_annotation = max(done_times) if done_times else None

    # --- pool -----------------------------------------------------------------
    r.pool_dirs = list(ws.config.get("pool", []))
    if scan_pool:
        sources = {resolve_path(q["source"]).resolve() for q in queue if q.get("source")}
        images = find_images([Path(d) for d in r.pool_dirs])
        r.pool_total = len(images)
        r.pool_remaining = sum(1 for p in images if p.resolve() not in sources and p.stem not in labeled)

    # --- a training round that never finished (or is running now) --------------
    nxt = r.round + 1
    run_dir = ws.runs_dir / f"round_{nxt:02d}"
    pending_manifest = _manifest_for(ws, nxt, None)
    if run_dir.exists() or pending_manifest is not None:
        info = {"round": nxt, "dir": str(rel_to_project(run_dir)), "epochs_done": 0}
        res = run_dir / "results.csv"
        if res.exists():
            info["epochs_done"] = max(0, len(res.read_text().strip().splitlines()) - 1)
            info["last_update"] = _mtime(res)
        if pending_manifest is not None:
            info["started"] = _parse_time(pending_manifest.get("started"))
            info.setdefault("last_update", _mtime(ws.manifest_path(nxt)))
        r.pending_train = info

    r.events = ws.read_log()
    return r


# ----------------------------------------------------------------------------
# Advice
# ----------------------------------------------------------------------------


def next_steps(r: Report) -> list[str]:
    todo = r.queue_counts.get("todo", 0)
    n_new = len(r.new_labels) + len(r.edited_labels)
    steps: list[str] = []

    if r.pending_train:
        p = r.pending_train
        recent = p.get("last_update") and time.time() - p["last_update"] < 15 * 60
        if recent:
            steps.append(
                f"Training round {p['round']} looks like it is RUNNING ({p['epochs_done']} epochs logged, "
                f"updated {ago(p['last_update'])}). Check its tmux window and wait for it to finish."
            )
            return steps
        steps.append(
            f"Training round {p['round']} was started but never finished ({p['epochs_done']} epochs logged). "
            f"Re-run it (it overwrites {p['dir']}):\n      {TRAIN_CMD}"
        )
        return steps

    if not r.model_exists:
        steps.append(f"The current model file is missing ({r.model}). Re-run init with --model <weights>.")
        return steps

    if todo:
        if r.stale_preds or r.missing_preds:
            steps.append(
                f"{r.stale_preds + r.missing_preds} queued predictions are older than the current model. "
                "Refresh them so pre-annotations are better:\n      python code/al.py repredict"
            )
        steps.append(f"Annotate the {todo} remaining queued image(s):\n      python code/al.py serve")
        if n_new >= SUGGEST_TRAIN_AT:
            steps.append(
                f"Optional: you already have {n_new} new/edited labels. You may retrain now and annotate the "
                f"rest with better pre-annotations afterwards:\n      {TRAIN_CMD}"
            )
        return steps

    if n_new:
        steps.append(f"Retrain on all labels ({n_new} new/edited since the current model):\n      {TRAIN_CMD}")
        return steps

    if r.pool_remaining == 0:
        steps.append(
            "The pool is exhausted. Add new candidate images (e.g. pick a diverse subset with\n"
            "      python code/select_diverse_images.py --input <dir> --output data/<new> --n 500 --method fps\n"
            "      and register it:  python code/al.py pool --add data/<new>)"
        )
        return steps

    steps.append("Queue a new batch of uncertain images for annotation:\n      python code/al.py select --n 20")
    return steps


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------


def ago(ts: float | None) -> str:
    if not ts:
        return "never"
    d = time.time() - ts
    when = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    for unit, sec in (("d", 86400), ("h", 3600), ("min", 60)):
        if d >= sec:
            return f"{when} ({int(d // sec)} {unit} ago)"
    return f"{when} (just now)"


def describe_event(e: dict) -> str:
    ev = e.get("event")
    if ev == "init":
        return f"init: {e.get('imported', 0)} imported labels, model {e.get('model')}"
    if ev == "select":
        return f"select (round {e.get('round')}): scored {e.get('n_scored')}, queued {e.get('n_queued')}"
    if ev == "annotated":
        return f"annotated {e.get('stem')} ({e.get('kind')}, {e.get('split')})"
    if ev == "status_changed":
        return f"marked {e.get('stem')} as {e.get('status')}"
    if ev == "train_started":
        return f"train round {e.get('round')} started: {e.get('n_train')} train / {e.get('n_val')} val"
    if ev == "train_finished":
        miou = e.get("best_mIoU")
        return f"train round {e.get('round')} finished: best mIoU {miou:.4f}" if miou else f"train round {e.get('round')} finished"
    if ev == "repredict":
        return f"re-predicted {e.get('n')} queued images"
    if ev == "pool_add":
        return f"added pool dir {e.get('dir')}"
    return json.dumps(e)


def _collapse(events: list[dict]) -> list[tuple[str, str]]:
    """Merge consecutive annotation saves into one line ('annotated 7 images')."""
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(events):
        e = events[i]
        if e.get("event") == "annotated":
            j = i
            stems = set()
            while j < len(events) and events[j].get("event") == "annotated":
                stems.add(events[j].get("stem"))
                j += 1
            if j - i > 1:
                out.append((events[j - 1]["time"], f"annotation session: saved {len(stems)} image(s) ({e['time'][11:16]}-{events[j - 1]['time'][11:16]})"))
                i = j
                continue
        out.append((e.get("time", "?"), describe_event(e)))
        i += 1
    return out


def format_report(ws: Workspace, r: Report, recent: int = 8) -> str:
    todo = r.queue_counts.get("todo", 0)
    done = r.queue_counts.get("done", 0)
    skipped = r.queue_counts.get("skipped", 0)
    n_new, n_edit = len(r.new_labels), len(r.edited_labels)
    rq_total = sum(r.round_queue.values())
    L: list[str] = []
    bar = "=" * 70
    L += [bar, f" Active-learning status   {ws.root}", bar]

    L.append("Model")
    L.append(f"  current     {r.model}{'' if r.model_exists else '   (FILE MISSING)'}")
    L.append(f"  round       {r.round}   (saved {ago(r.model_time)})")
    L.append(f"  trained on  {r.model_basis}")

    L.append("Labels")
    L.append(f"  total       {r.n_train + r.n_val}  ({r.n_train} train / {r.n_val} val)")
    edit_note = " (approx.)" if r.edits_approx and n_edit else ""
    L.append(f"  not yet used by current model: {n_new} new + {n_edit} edited{edit_note}")
    L.append(f"  last annotation saved: {ago(r.last_annotation)}")

    L.append("Pool")
    if r.pool_total is None:
        L.append(f"  dirs        {', '.join(r.pool_dirs)}   (not scanned, --no-pool)")
    else:
        L.append(f"  dirs        {', '.join(r.pool_dirs)}")
        L.append(f"  images      {r.pool_total} total, {r.pool_remaining} unlabelled and not queued")

    L.append("Queue")
    L.append(f"  all rounds  {done} done / {todo} todo / {skipped} skipped")
    if r.stale_preds or r.missing_preds:
        L.append(f"  {r.stale_preds + r.missing_preds} todo prediction(s) older than current model")

    # Phase checklist
    def mark(state):
        return {"done": "[x]", "partial": "[~]", "todo": "[ ]", "warn": "[!]"}[state]

    phases = []
    if r.pool_remaining is None:
        phases.append(("todo", "1 Pool", "not scanned"))
    elif r.pool_remaining:
        phases.append(("done", "1 Pool", f"{r.pool_remaining} candidates available"))
    else:
        phases.append(("warn", "1 Pool", "exhausted, add new images"))
    phases.append(
        ("done", "2 Select", f"{rq_total} image(s) queued in round {r.round}")
        if rq_total else ("todo", "2 Select", f"nothing queued yet in round {r.round}")
    )
    if todo == 0:
        phases.append(("done" if rq_total else "todo", "3 Pre-annotate", "-" if not rq_total else "done"))
    elif r.stale_preds or r.missing_preds:
        phases.append(("warn", "3 Pre-annotate", f"{r.stale_preds + r.missing_preds} stale, run repredict"))
    else:
        phases.append(("done", "3 Pre-annotate", "predictions up to date"))
    if todo:
        phases.append(("partial", "4 Annotate", f"{done} done, {todo} to go"))
    else:
        phases.append(("done" if done else "todo", "4 Annotate", "queue empty" if done else "-"))
    if r.pending_train:
        phases.append(("warn", "5 Retrain", f"round {r.pending_train['round']} started, not finished"))
    elif n_new + n_edit:
        phases.append(("todo", "5 Retrain", f"{n_new + n_edit} label(s) not yet used"))
    else:
        phases.append(("done", "5 Retrain", "model is up to date with labels"))
    L.append(f"Phases (round {r.round})")
    for state, name, msg in phases:
        L.append(f"  {mark(state)} {name:<16}{msg}")

    hist = ws.history_csv
    if hist.exists():
        L.append("Training history")
        for line in hist.read_text().strip().splitlines():
            L.append("  " + line.replace(",", "  "))

    L.append("Recent activity")
    if r.events:
        for t, msg in _collapse(r.events)[-recent:]:
            L.append(f"  {t.replace('T', ' ')[:16]}  {msg}")
    else:
        L.append("  (no activity log yet: it starts with the next command you run)")

    L.append("Next step")
    for i, s in enumerate(next_steps(r)):
        L.append(f"  {'->' if i == 0 else '  '} {s}")
    L.append(bar)
    return "\n".join(L)
