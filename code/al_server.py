"""Browser-based mask correction tool for the active-learning loop.

Runs a small HTTP server (Python stdlib only) on the remote machine. Reach it
from your laptop through an SSH tunnel:

    # on the laptop
    ssh -L 8765:localhost:8765 user@remote
    # on the remote (inside that ssh session, ideally in tmux)
    python code/al.py serve
    # then open http://localhost:8765 in the laptop's browser
"""

from __future__ import annotations

import gzip
import json
import mimetypes
import sys
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from al_core import IGNORE_ID, Workspace

STATIC_DIR = Path(__file__).resolve().parent / "al_static"

# RGB colours by class name (falls back to a palette by id).
NAMED_COLORS = {
    "near field": [0, 200, 0],
    "seafloor": [230, 120, 0],
    "surface turbidity": [0, 140, 255],
    "background": [0, 0, 0],
}
PALETTE = [[0, 200, 0], [230, 120, 0], [0, 140, 255], [255, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 0]]


class Annotator:
    """Resolves annotation items (queue images + already-labelled dataset images)."""

    def __init__(self, ws: Workspace):
        self.ws = ws

    def meta(self) -> dict:
        classes = self.ws.classes
        bg = next((k for k, v in classes.items() if v == "background"), None)
        return {
            "classes": [
                {"id": k, "name": v, "color": NAMED_COLORS.get(v, PALETTE[k % len(PALETTE)])}
                for k, v in sorted(classes.items())
            ],
            "background_id": bg,
            "ignore_id": IGNORE_ID,
            "workspace": str(self.ws.root),
            "model": self.ws.config.get("current_model"),
            "round": self.ws.config.get("round", 0),
        }

    def items(self) -> list[dict]:
        labeled = self.ws.labeled_stems()
        out = []
        seen = set()
        for row in self.ws.read_queue():
            stem = row["stem"]
            seen.add(stem)
            status = row["status"]
            out.append(
                {
                    "stem": stem,
                    "kind": "queue",
                    "status": status,
                    "score": float(row["score"]) if row.get("score") else None,
                    "round": row.get("round"),
                    "split": labeled.get(stem, row.get("split") or self.ws.split_for(stem)),
                    "source": row.get("source"),
                }
            )
        for stem, split in sorted(labeled.items()):
            if stem not in seen:
                out.append(
                    {"stem": stem, "kind": "dataset", "status": "done", "score": None, "split": split, "source": None}
                )
        return out

    def _queue_row(self, stem: str) -> dict | None:
        return next((r for r in self.ws.read_queue() if r["stem"] == stem), None)

    def image_path(self, stem: str) -> Path | None:
        p = self.ws.queue_dir / "images" / f"{stem}.png"
        if p.exists():
            return p
        found = self.ws.find_label(stem)
        return found[0] if found else None

    def mask(self, stem: str, prefer: str = "auto") -> tuple[np.ndarray, str] | None:
        """Return (mask, origin) where origin is 'label', 'prediction' or 'empty'."""
        label = self.ws.find_label(stem)
        pred = self.ws.queue_dir / "pred" / f"{stem}.png"
        if prefer != "prediction" and label is not None:
            m = cv2.imread(str(label[1]), cv2.IMREAD_UNCHANGED)
            return m, "label"
        if pred.exists():
            return cv2.imread(str(pred), cv2.IMREAD_UNCHANGED), "prediction"
        img = self.image_path(stem)
        if img is None:
            return None
        h, w = cv2.imread(str(img)).shape[:2]
        bg = self.meta()["background_id"] or 0
        return np.full((h, w), bg, np.uint8), "empty"

    def save(self, stem: str, mask: np.ndarray, split: str) -> None:
        img = self.image_path(stem)
        if img is None:
            raise KeyError(stem)
        h, w = cv2.imread(str(img)).shape[:2]
        if mask.shape != (h, w):
            raise ValueError(f"Mask shape {mask.shape} != image shape {(h, w)}")
        valid = set(self.ws.classes) | {IGNORE_ID}
        bad = set(np.unique(mask).tolist()) - valid
        if bad:
            raise ValueError(f"Invalid class ids in mask: {sorted(bad)}")
        self.ws.save_label(stem, img, mask, split)
        if self._queue_row(stem) is not None:
            self.ws.update_queue_item(stem, status="done", split=split)

    def set_status(self, stem: str, status: str) -> None:
        if status not in ("todo", "skipped"):
            raise ValueError(status)
        if self._queue_row(stem) is None:
            raise KeyError(stem)
        self.ws.update_queue_item(stem, status=status)


def make_handler(annotator: Annotator):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EchogramAnnotator/1.0"

        def log_message(self, fmt, *args):  # quieter logs
            if not self.path.startswith("/api/image") and not self.path.startswith("/api/conf"):
                sys.stderr.write("[serve] " + (fmt % args) + "\n")

        # --- helpers -------------------------------------------------------
        def _send(self, status, body: bytes, ctype: str, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status=HTTPStatus.OK):
            self._send(status, json.dumps(obj).encode(), "application/json")

        def _error(self, status, msg):
            self._json({"error": msg}, status)

        def _parse(self):
            url = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            stem = q.get("stem")
            if stem is not None and ("/" in stem or "\\" in stem or stem.startswith(".")):
                stem = None
            return url.path, q, stem

        # --- GET -----------------------------------------------------------
        def do_GET(self):
            path, q, stem = self._parse()
            try:
                if path in ("/", "/index.html"):
                    body = (STATIC_DIR / "annotator.html").read_bytes()
                    return self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                if path == "/api/meta":
                    return self._json(annotator.meta())
                if path == "/api/items":
                    return self._json(annotator.items())
                if path == "/api/image" and stem:
                    p = annotator.image_path(stem)
                    if p is None:
                        return self._error(HTTPStatus.NOT_FOUND, "no image")
                    if p.suffix.lower() in (".tif", ".tiff", ".bmp"):  # not browser-viewable
                        ok, buf = cv2.imencode(".png", cv2.imread(str(p)))
                        return self._send(HTTPStatus.OK, buf.tobytes(), "image/png")
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    return self._send(HTTPStatus.OK, p.read_bytes(), ctype)
                if path == "/api/conf" and stem:
                    p = annotator.ws.queue_dir / "conf" / f"{stem}.png"
                    if not p.exists():
                        return self._error(HTTPStatus.NOT_FOUND, "no confidence map")
                    return self._send(HTTPStatus.OK, p.read_bytes(), "image/png")
                if path == "/api/mask" and stem:
                    res = annotator.mask(stem, q.get("prefer", "auto"))
                    if res is None:
                        return self._error(HTTPStatus.NOT_FOUND, "no mask")
                    mask, origin = res
                    h, w = mask.shape[:2]
                    body = gzip.compress(np.ascontiguousarray(mask, dtype=np.uint8).tobytes(), 6)
                    return self._send(
                        HTTPStatus.OK,
                        body,
                        "application/octet-stream",
                        {
                            "Content-Encoding": "gzip",
                            "X-Width": str(w),
                            "X-Height": str(h),
                            "X-Mask-Origin": origin,
                            "Access-Control-Expose-Headers": "X-Width, X-Height, X-Mask-Origin",
                        },
                    )
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as e:  # noqa: BLE001
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(e))

        # --- POST ----------------------------------------------------------
        def do_POST(self):
            path, q, stem = self._parse()
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length) if length else b""
                if path == "/api/mask" and stem:
                    w, h = int(q["w"]), int(q["h"])
                    raw = gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body
                    if len(raw) != w * h:
                        return self._error(HTTPStatus.BAD_REQUEST, "mask size mismatch")
                    mask = np.frombuffer(raw, np.uint8).reshape(h, w)
                    split = q.get("split") or annotator.ws.split_for(stem)
                    annotator.save(stem, mask, split)
                    return self._json({"ok": True, "split": split})
                if path == "/api/status" and stem:
                    annotator.set_status(stem, q.get("status", ""))
                    return self._json({"ok": True})
                return self._error(HTTPStatus.NOT_FOUND, "not found")
            except (KeyError, ValueError) as e:
                return self._error(HTTPStatus.BAD_REQUEST, repr(e))
            except Exception as e:  # noqa: BLE001
                return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(e))

    return Handler


def serve(ws: Workspace, host: str = "127.0.0.1", port: int = 8765) -> None:
    annotator = Annotator(ws)
    httpd = ThreadingHTTPServer((host, port), make_handler(annotator))
    print("=" * 64)
    print(f"Annotator running on http://{host}:{port}")
    print(f"Workspace: {ws.root}")
    if host in ("127.0.0.1", "localhost"):
        print("From your laptop, open an SSH tunnel and browse to it:")
        print(f"    ssh -N -L {port}:localhost:{port} <user>@<this-host>")
        print(f"    open http://localhost:{port}")
    print("Ctrl+C to stop.")
    print("=" * 64)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
