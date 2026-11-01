#!/usr/bin/env python3
"""
build_gcp_dataset.py

Derive a "GCP number" classification dataset from an object-detection export.
Supported annotation formats: YOLO txt (boxes or polygons), COCO json, and
per-image "boxes" JSON ({"boxes":[{"label","points":[[x,y],..]}], "key": ...}).
If the GCP / number annotations are polygons they are used directly (exact GCP
corners, exact digit outlines); plain boxes fall back to contour detection.

For every image that contains at least one GCP box and at least one number box:
  * each GCP is paired with the number whose box centre is closest to the
    GCP's centre,
  * the GCP is located precisely (quad detection inside its bounding box),
    perspective-deskewed to a square, and rotated in 90-degree steps so the
    number is upright,
  * the deskewed GCP is saved to   <out>/gcps/<number>.png
  * the upright number alone is saved to <out>/numbers/<number>.png
    (cut from the original full-resolution image).

If the same number occurs more than once the files are named
<number>.png, <number>_1.png, <number>_2.png ... (use --class-folders to put
each number in its own sub-folder instead). A manifest.csv records where
every sample came from.

Requirements:  pip install opencv-python numpy pyyaml
"""
import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------
# Annotation loading
# --------------------------------------------------------------------------
@dataclass
class Box:
    cls: str
    x1: float
    y1: float
    x2: float
    y2: float
    poly: Optional[np.ndarray] = None  # Nx2 polygon in image px, if annotated

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    def contains(self, x, y):
        if self.poly is not None and len(self.poly) >= 3:
            return cv2.pointPolygonTest(
                self.poly.reshape(-1, 1, 2).astype(np.float32), (float(x), float(y)), False
            ) >= 0
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def box_from_points(name, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
    return Box(name, float(x1), float(y1), float(x2), float(y2), poly=pts)


def load_yolo_names(root: Path):
    import yaml

    yaml_path = next(iter(sorted(root.rglob("data.yaml"))), None)
    if yaml_path is None:
        sys.exit("No data.yaml found - cannot read YOLO class names.")
    names = yaml.safe_load(yaml_path.read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return [str(n) for n in names]


def iter_yolo(root: Path, names):
    """Yield (image_path, get_boxes(w, h)). Handles Roboflow's
    <split>/images + <split>/labels layout, or labels next to the images."""
    for img_path in sorted(root.rglob("*")):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            label_path = img_path.with_suffix(".txt")
        if not label_path.exists():
            continue

        def get_boxes(w, h, label_path=label_path):
            boxes = []
            for line in label_path.read_text().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cid, vals = int(float(p[0])), [float(v) for v in p[1:]]
                if cid >= len(names):
                    continue
                if len(vals) == 4:  # cx cy w h
                    cx, cy, bw, bh = vals
                    x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
                else:  # polygon -> take its bounding box
                    pts = np.array(vals, np.float32).reshape(-1, 2) * [w, h]
                    boxes.append(box_from_points(names[cid], pts))
                    continue
                boxes.append(Box(names[cid], x1 * w, y1 * h, x2 * w, y2 * h))
            return boxes

        yield img_path, get_boxes


def iter_coco(root: Path):
    for json_path in sorted(root.rglob("_annotations.coco.json")):
        data = json.loads(json_path.read_text())
        cats = {c["id"]: str(c["name"]) for c in data["categories"]}
        per_image = defaultdict(list)
        for a in data["annotations"]:
            x, y, w, h = a["bbox"]
            per_image[a["image_id"]].append(Box(cats[a["category_id"]], x, y, x + w, y + h))
        for im in data["images"]:
            img_path = json_path.parent / im["file_name"]
            boxes = per_image.get(im["id"], [])
            yield img_path, (lambda w, h, boxes=boxes: boxes)


def iter_boxes_json(root: Path):
    """Per-image JSON: {"boxes":[{"label","type","x","y","width","height",
    "points"}], "key": "<image file name>", "width", "height"}.
    NB: x/y are box *centres* in this format. Polygon points are used when
    present. The image is looked up next to the JSON, then by file name
    anywhere under the dataset root."""
    index = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            index.setdefault(p.name, p)
    for jp in sorted(root.rglob("*.json")):
        try:
            data = json.loads(jp.read_text())
        except Exception:
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict) or "boxes" not in item:
                continue
            key = item.get("key") or item.get("filename") or item.get("image") or ""
            img_path = jp.parent / key
            if not img_path.is_file():
                img_path = index.get(Path(key).name)
            if img_path is None:
                print(f"  ! no image found for annotation key {key!r} ({jp.name})")
                continue

            def get_boxes(w, h, item=item):
                sx = w / item["width"] if w and item.get("width") else 1.0
                sy = h / item["height"] if h and item.get("height") else 1.0
                out = []
                for b in item["boxes"]:
                    name, pts = str(b["label"]), b.get("points")
                    if pts and len(pts) >= 3:
                        out.append(box_from_points(name, np.array(pts, np.float32) * [sx, sy]))
                    else:
                        cx, cy, bw, bh = b["x"], b["y"], b["width"], b["height"]
                        out.append(Box(name, (cx - bw / 2) * sx, (cy - bh / 2) * sy,
                                       (cx + bw / 2) * sx, (cy + bh / 2) * sy))
                return out

            yield img_path, get_boxes


def load_dataset(root: Path, fmt: str):
    if fmt == "auto":
        if any(root.rglob("_annotations.coco.json")):
            fmt = "coco"
        elif any(root.rglob("data.yaml")):
            fmt = "yolo"
        else:
            fmt = "boxes"
    if fmt in ("coco", "boxes"):
        samples = list(iter_coco(root) if fmt == "coco" else iter_boxes_json(root))
        names = sorted({b.cls for _, gb in samples for b in gb(0, 0)})
    else:
        names = load_yolo_names(root)
        samples = list(iter_yolo(root, names))
    return fmt, names, samples


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def crop_padded(img, x1, y1, x2, y2):
    """Crop [x1:x2, y1:y2]; anything outside the image is edge-replicated."""
    h, w = img.shape[:2]
    cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
    if cx2 <= cx1 or cy2 <= cy1:
        return np.zeros((max(y2 - y1, 1), max(x2 - x1, 1)) + img.shape[2:], img.dtype)
    roi = img[cy1:cy2, cx1:cx2]
    return cv2.copyMakeBorder(
        roi, cy1 - y1, y2 - cy2, cx1 - x1, x2 - cx2, cv2.BORDER_REPLICATE
    )


def order_quad(pts):
    """Order 4 points clockwise (image coordinates). The starting corner is
    arbitrary-but-consistent; the 4-way rotation search fixes the rest."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)


def quad_side_lengths(q):
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def hull_to_quad(hull):
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.10, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def find_gcp_quad(img, box, pad=0.10, min_area=0.20):
    """Locate the GCP's four corners inside its (axis-aligned) bounding box.

    Several foreground masks (Otsu, inverted Otsu, closed Canny edges) are
    tried; each external contour that doesn't touch the padded ROI border is
    reduced to a quad and scored by size * how well it fits a quad. Falls back
    to the plain bounding box if nothing convincing is found.
    Returns (quad[4,2] in full-image coordinates, method_name).
    """
    bw, bh = box.x2 - box.x1, box.y2 - box.y1
    x1, y1 = int(math.floor(box.x1 - bw * pad)), int(math.floor(box.y1 - bh * pad))
    x2, y2 = int(math.ceil(box.x2 + bw * pad)), int(math.ceil(box.y2 + bh * pad))
    fallback = np.array(
        [[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]], np.float32
    )
    roi = crop_padded(img, x1, y1, x2, y2)
    rh, rw = roi.shape[:2]
    if rh < 16 or rw < 16:
        return fallback, "bbox"

    gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8), iterations=2)
    k = max(3, int(0.03 * min(rh, rw))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    best = None
    for mask in (otsu, 255 - otsu, edges):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + w >= rw - 1 or y + h >= rh - 1:
                continue  # touches ROI border -> it's background, not the GCP
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area < min_area * rh * rw:
                continue
            quad = hull_to_quad(hull)
            quad_area = cv2.contourArea(quad.reshape(-1, 1, 2))
            fit = min(hull_area, quad_area) / max(hull_area, quad_area, 1e-6)
            score = (hull_area / (rh * rw)) * fit ** 2
            if best is None or score > best[0]:
                best = (score, quad)
    if best is None:
        return fallback, "bbox"
    return best[1] + np.array([x1, y1], np.float32), "contour-quad"


# --------------------------------------------------------------------------
# Orientation: which of the 4 rotations makes the number upright?
# --------------------------------------------------------------------------
_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
]
_template_cache = {}


def _normalise_glyph(binary, size=32):
    """Tight-crop a 0/255 mask, fit into size x size keeping aspect ratio,
    blur, and return a zero-mean unit-norm vector (or None if empty)."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
    h, w = crop.shape
    scale = (size - 4) / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.float32)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = crop / 255.0
    canvas = cv2.GaussianBlur(canvas, (5, 5), 1.2)
    canvas -= canvas.mean()
    n = np.linalg.norm(canvas)
    return canvas / n if n > 0 else None


def _templates(label):
    if label not in _template_cache:
        out = []
        for font in _FONTS:
            for thick in (2, 4):
                (tw, th), base = cv2.getTextSize(label, font, 2, thick)
                canvas = np.zeros((th + base + 40, tw + 40), np.uint8)
                cv2.putText(canvas, label, (20, 20 + th), font, 2, 255, thick, cv2.LINE_AA)
                g = _normalise_glyph(((canvas > 100) * 255).astype(np.uint8))
                if g is not None:
                    out.append(g)
        _template_cache[label] = out
    return _template_cache[label]


def glyph_mask(gray):
    """Foreground mask of the digit(s) in a padded grey crop. Foreground is
    assumed to be the minority colour; blobs touching the crop edge (panel
    borders etc.) are discarded."""
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if m.mean() > 127:
        m = 255 - m
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    H, W = m.shape
    keep = np.zeros_like(m)
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if x <= 0 or y <= 0 or x + w >= W or y + h >= H or a < 0.002 * H * W:
            continue
        keep[labels == i] = 255
    return keep if keep.any() else m


def choose_rotation(mask, label):
    """Return (k, scores): rotate by k*90deg CCW to get the digit upright;
    scores[k] is the template similarity of each of the 4 rotations."""
    templates = _templates(label)
    scores = []
    for k in range(4):
        g = _normalise_glyph(np.rot90(mask, k))
        scores.append(
            max((float(np.dot(g.ravel(), t.ravel())) for t in templates), default=0.0)
            if g is not None else 0.0
        )
    return int(np.argmax(scores)), scores


# --------------------------------------------------------------------------
# Pairing GCPs with numbers
# --------------------------------------------------------------------------
def pair_gcps_with_numbers(gcps, numbers, require_inside=True):
    """Each GCP gets the number whose centre is nearest to the GCP centre.
    If two GCPs claim the same number, the nearer GCP keeps it and the other
    is dropped (ambiguous). Returns (pairs, n_unmatched)."""
    claims = []
    for gi, g in enumerate(gcps):
        best = None
        for ni, n in enumerate(numbers):
            if require_inside and not g.contains(n.cx, n.cy):
                continue
            d = math.hypot(n.cx - g.cx, n.cy - g.cy)
            if best is None or d < best[0]:
                best = (d, ni)
        if best:
            claims.append((best[0], gi, best[1]))
    claims.sort()
    used, pairs = set(), []
    for _, gi, ni in claims:
        if ni not in used:
            used.add(ni)
            pairs.append((gcps[gi], numbers[ni]))
    return pairs, len(gcps) - len(pairs)


# --------------------------------------------------------------------------
# Main processing
# --------------------------------------------------------------------------
def warp_quad(img, quad, w, h):
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    ), M


SIDES = {"top": 0, "left": 1, "bottom": 2, "right": 3}  # CCW order, like np.rot90


def process_pair(img, gcp, num, label, args):
    """Returns (gcp_img, number_img, info) or (None, None, reason)."""
    # 1) GCP corners: annotated polygon if available, else contour detection
    if gcp.poly is not None and len(gcp.poly) >= 4:
        quad = (gcp.poly if len(gcp.poly) == 4
                else hull_to_quad(cv2.convexHull(gcp.poly.reshape(-1, 1, 2)))).astype(np.float32)
        method = "polygon"
    else:
        quad, method = find_gcp_quad(img, gcp)
    quad = order_quad(quad)
    S = max(args.gcp_size or int(round(np.mean(quad_side_lengths(quad)))), 16)
    warped, M = warp_quad(img, quad, S, S)
    Minv = np.linalg.inv(M)

    # 2) number outline (polygon, or box corners) -> deskewed GCP frame
    pts = num.poly if num.poly is not None else np.array(
        [[num.x1, num.y1], [num.x2, num.y1], [num.x2, num.y2], [num.x1, num.y2]], np.float32)
    npts = cv2.perspectiveTransform(pts.astype(np.float32).reshape(-1, 1, 2), M).reshape(-1, 2)
    rx1, ry1 = np.floor(npts.min(axis=0)).astype(int)
    rx2, ry2 = np.ceil(npts.max(axis=0)).astype(int)
    rx1, ry1, rx2, ry2 = max(rx1, 0), max(ry1, 0), min(rx2, S), min(ry2, S)
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        return None, None, "number outside detected GCP"

    # 3) glyph mask in a padded window around the number
    p = int(0.2 * max(rx2 - rx1, ry2 - ry1))
    ox, oy = rx1 - p, ry1 - p
    if num.poly is not None:  # exact outline from the annotation
        mask = np.zeros((ry2 + p - oy, rx2 + p - ox), np.uint8)
        cv2.fillPoly(mask, [np.round(npts - [ox, oy]).astype(np.int32)], 255)
    else:                     # estimate by thresholding
        gray = cv2.cvtColor(crop_padded(warped, ox, oy, rx2 + p, ry2 + p), cv2.COLOR_BGR2GRAY)
        mask = glyph_mask(gray)

    # 4) where does the number sit relative to the GCP centre? (0=top 1=left 2=bottom 3=right)
    ncx, ncy = (npts.min(axis=0) + npts.max(axis=0)) / 2
    dx, dy = (ncx - S / 2) / S, (ncy - S / 2) / S
    side = None
    if max(abs(dx), abs(dy)) >= args.min_offset:
        side = (0 if dy < 0 else 2) if abs(dy) >= abs(dx) else (1 if dx < 0 else 3)

    # 5) choose rotation k (np.rot90 steps, CCW)
    k, score, o_method, k_template = 0, None, "none", None
    if args.orientation != "none":
        k_template, scores = choose_rotation(mask, label)
        if args.orientation == "position" and side is not None:
            k = (SIDES[args.number_position] - side) % 4
            o_method = "position"
        else:
            k, o_method = k_template, "template"
        score = scores[k]
        if o_method == "template" and score < args.min_orient_score:
            return None, None, f"orientation uncertain (score {score:.2f})"

    gcp_out = np.ascontiguousarray(np.rot90(warped, k))

    # 6) number crop rectangle in the deskewed (un-rotated) frame
    x0, y0, x1, y1 = rx1, ry1, rx2, ry2
    if args.refine_number_crop or num.poly is not None:
        ys, xs = np.nonzero(mask)
        if len(xs):
            m = 0.08 * max(xs.max() - xs.min(), ys.max() - ys.min())
            x0, x1 = ox + xs.min() - m, ox + xs.max() + 1 + m
            y0, y1 = oy + ys.min() - m, oy + ys.max() + 1 + m

    # cut the upright number from the ORIGINAL image at native resolution
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
    corners = np.roll(corners, -k, axis=0)  # same rotation as np.rot90(.., k)
    oc = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), Minv).reshape(4, 2)
    w = (np.linalg.norm(oc[1] - oc[0]) + np.linalg.norm(oc[2] - oc[3])) / 2
    h = (np.linalg.norm(oc[3] - oc[0]) + np.linalg.norm(oc[2] - oc[1])) / 2
    num_out, _ = warp_quad(img, oc, max(int(round(w)), 2), max(int(round(h)), 2))

    info = dict(method=method, rotation_deg=k * 90, orient_score=score, quad=quad,
                orient_method=o_method,
                template_rotation_deg=None if k_template is None else k_template * 90,
                side_after=None if side is None else (side + k) % 4)
    return gcp_out, num_out, info


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path, help="root folder of the Roboflow export")
    ap.add_argument("-o", "--output", type=Path, default=Path("gcp_dataset"))
    ap.add_argument("--format", choices=["auto", "yolo", "coco", "boxes"], default="auto")
    ap.add_argument("--gcp-regex", default=r"(?i)gcp|^p$",
                    help="regex matching GCP class names (default: contains 'gcp', or exactly 'p')")
    ap.add_argument("--number-regex", default=r"^(\d+)$",
                    help="regex matching number class names; group 1 (or the whole "
                         "name) becomes the label (default: digits only)")
    ap.add_argument("--gcp-size", type=int, default=256,
                    help="output GCP size in px (square); 0 = native size")
    ap.add_argument("--orientation", choices=["digit", "position", "none"], default="digit",
                    help="'digit': pick the rotation whose digit best matches rendered "
                         "templates; 'position': rotate so the (off-centre) number sits at "
                         "--number-position; 'none': only deskew, no rotation")
    ap.add_argument("--number-position", choices=list(SIDES), default=None,
                    help="where the number sits on an upright GCP (for --orientation position)")
    ap.add_argument("--min-offset", type=float, default=0.08,
                    help="number must be at least this fraction of the GCP size away from "
                         "the GCP centre for the position cue to be used (default 0.08)")
    ap.add_argument("--min-orient-score", type=float, default=0.0,
                    help="skip samples whose best orientation score is below this "
                         "(scores are recorded in manifest.csv; ~0.7+ is confident)")
    ap.add_argument("--refine-number-crop", action="store_true",
                    help="tighten number crops to the detected glyph pixels")
    ap.add_argument("--require-inside", action=argparse.BooleanOptionalAction, default=True,
                    help="only pair a number whose centre lies inside the GCP box")
    ap.add_argument("--class-folders", action="store_true",
                    help="save into gcps/<number>/ and numbers/<number>/ sub-folders")
    ap.add_argument("--debug-dir", type=Path, default=None,
                    help="save annotated source images to check detection/pairing")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.orientation == "position" and not args.number_position:
        sys.exit("--orientation position requires --number-position top|left|bottom|right")
    fmt, names, samples = load_dataset(args.dataset, args.format)
    gcp_re, num_re = re.compile(args.gcp_regex), re.compile(args.number_regex)

    num_label = {}
    for n in names:
        m = num_re.search(n)
        if m and not gcp_re.search(n):
            num_label[n] = m.group(1) if m.groups() else m.group(0)
    gcp_names = {n for n in names if gcp_re.search(n)}
    print(f"Format: {fmt}   images with annotation files: {len(samples)}")
    print(f"All classes : {names}")
    print(f"GCP classes : {sorted(gcp_names)}")
    print(f"Number classes: {sorted(num_label)}")
    if not gcp_names or not num_label:
        sys.exit("Could not identify GCP and/or number classes - adjust "
                 "--gcp-regex / --number-regex using the class list above.")

    gcp_dir, num_dir = args.output / "gcps", args.output / "numbers"
    gcp_dir.mkdir(parents=True, exist_ok=True)
    num_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_dir:
        args.debug_dir.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    side_counts = defaultdict(int)
    stats = defaultdict(int)
    rows = []

    for img_path, get_boxes in samples:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ! cannot read {img_path}")
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        boxes = get_boxes(w, h)
        gcps = [b for b in boxes if b.cls in gcp_names]
        numbers = [b for b in boxes if b.cls in num_label]
        if not gcps or not numbers:
            stats["skipped: no GCP" if not gcps else "skipped: no number"] += 1
            continue

        pairs, unmatched = pair_gcps_with_numbers(gcps, numbers, args.require_inside)
        stats["GCPs without a usable number"] += unmatched
        dbg = img.copy() if args.debug_dir else None
        t = max(2, w // 600)

        for gcp, num in pairs:
            label = num_label[num.cls]
            gcp_img, num_img, info = process_pair(img, gcp, num, label, args)
            if gcp_img is None:
                stats[f"dropped: {info.split(' (')[0]}"] += 1
                continue

            n = counts[label]
            counts[label] += 1
            stem = label if n == 0 else f"{label}_{n}"
            sub = Path(label) if args.class_folders else Path()
            for d, im in ((gcp_dir, gcp_img), (num_dir, num_img)):
                (d / sub).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / sub / f"{stem}.png"), im)
            stats["saved"] += 1
            if (info["orient_method"] == "template" and info["side_after"] is not None
                    and info["orient_score"] >= 0.6):
                side_counts[info["side_after"]] += 1
            rows.append(dict(
                file=f"{stem}.png", number=label, source=str(img_path),
                gcp_box=[round(v, 1) for v in (gcp.x1, gcp.y1, gcp.x2, gcp.y2)],
                number_box=[round(v, 1) for v in (num.x1, num.y1, num.x2, num.y2)],
                deskew=info["method"], rotation_deg=info["rotation_deg"],
                template_rotation_deg=info["template_rotation_deg"],
                orient_method=info["orient_method"], orient_score="" if info["orient_score"] is None else round(info["orient_score"], 3),
            ))
            if dbg is not None:
                cv2.polylines(dbg, [info["quad"].astype(np.int32)], True, (0, 0, 255), t)
                cv2.rectangle(dbg, (int(gcp.x1), int(gcp.y1)), (int(gcp.x2), int(gcp.y2)), (0, 255, 0), t)
                if num.poly is not None:
                    cv2.polylines(dbg, [num.poly.astype(np.int32)], True, (255, 0, 0), t)
                else:
                    cv2.rectangle(dbg, (int(num.x1), int(num.y1)), (int(num.x2), int(num.y2)), (255, 0, 0), t)
                cv2.putText(dbg, f"{label} rot{info['rotation_deg']}", (int(gcp.x1), max(int(gcp.y1) - 8, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, t / 3, (0, 0, 255), t)
        if dbg is not None:
            cv2.imwrite(str(args.debug_dir / f"{img_path.stem}.jpg"), dbg)

    with open(args.output / "manifest.csv", "w", newline="") as f:
        cols = ["file", "number", "source", "gcp_box", "number_box", "deskew",
                "rotation_deg", "template_rotation_deg", "orient_method", "orient_score"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    print("\nSummary")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"  distinct numbers: {len(counts)}")
    if side_counts:
        tot = sum(side_counts.values())
        pos = {name: side_counts.get(i, 0) for name, i in SIDES.items()}
        print("\nWhere the (off-centre) number ends up after template-based rotation "
              f"(confident samples only): {pos}")
        best = max(pos, key=pos.get)
        if pos[best] >= 0.8 * tot:
            print(f"  -> consistent. For deterministic orientation re-run with "
                  f"--orientation position --number-position {best}")
    print(f"Output written to {args.output.resolve()}")


if __name__ == "__main__":
    main()