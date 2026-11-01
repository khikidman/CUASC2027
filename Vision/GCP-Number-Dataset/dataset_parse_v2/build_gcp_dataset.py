#!/usr/bin/env python3
"""
Build a digit-classification dataset from the Roboflow "number-symbols" COCO export.

For every image in the export this script:
  1. finds each GCP polygon (class "gcp") and fits a 4-corner quadrilateral to it,
  2. assigns the GCP a number: the digit annotation lying inside the GCP that is closest to
     the GCP's center,
  3. deskews the GCP to a size x size square (perspective warp, same idea as recognize.py),
  4. rotates it into the canonical orientation: number on the RIGHT, upright
     (this is the orientation recognize.py's retrieve_numbers() works from),
  5. saves the rectified GCP and a crop of the number.

Images are dropped if they have no GCP annotations or no digit annotations. Individual
GCPs are dropped if they touch the image border, are too small/distorted, or have no digit
inside them (so a GCP is never labelled with another GCP's number).

Output layout (the split folder name is taken from the folder holding the annotation file,
e.g. train / valid / test, so splits are never mixed):

    <output>/<split>/gcps/<digit>/<digit>_img<image_id>_a<gcp_annotation_id>.png
    <output>/<split>/numbers/<digit>/<same file name>.png            (tight crop from labels)
    <output>/<split>/numbers_pipeline/<digit>/<same file name>.png   (only with --pipeline-crops)
    <output>/manifest.csv

Usage:
    python build_gcp_dataset.py path/to/dataset_root -o gcp_dataset
    python build_gcp_dataset.py path/to/train/_annotations.coco.json -o gcp_dataset
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2 as cv
import numpy as np

# --------------------------------------------------------------------------------------
# Tunable values (same spirit as the globals in recognize.py)
# --------------------------------------------------------------------------------------
SIZE = 256                  # rectified GCP is SIZE x SIZE
BORDER_THRESHOLD = 0.01     # skip GCPs whose polygon is within this fraction of the image edge
MIN_SIDE_PX = 24            # skip GCPs with a side shorter than this in the source image
MAX_SIDE_RATIO = 3.0        # skip quads whose longest/shortest side exceeds this (bad fits)
NUMBER_PAD = 4             # padding (in rectified px) around the tight number crop
MIN_OFFSET = 0.15           # number centre must be at least this fraction of SIZE from the GCP centre...
MIN_DOMINANCE = 1.5         # ...and clearly on one axis (|dx|/|dy| or |dy|/|dx| >= this)
EPSILON_FACTORS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]   # approxPolyDP tolerances (x perimeter)

# The fixed crop used by retrieve_numbers() in recognize.py, expressed as fractions of the
# half-image that contains the number (rows, cols). Only used with --pipeline-crops.
PIPE_ROWS = (74 / 256, 168 / 256)
PIPE_COLS = (40 / 256, 120 / 256)

DIGITS = {str(i) for i in range(10)}


# --------------------------------------------------------------------------------------
# Annotation helpers
# --------------------------------------------------------------------------------------
def annotation_points(ann, allow_bbox_fallback=False):
    """Polygon of a COCO annotation as an (N, 2) float32 array (closing duplicate removed)."""
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg:
        poly = max(seg, key=len)  # if a mask has several parts, use the biggest
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) >= 3:
            return pts
    if allow_bbox_fallback and ann.get("bbox"):
        x, y, w, h = ann["bbox"]
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)
    return None


def bbox_center(pts):
    return (pts.min(axis=0) + pts.max(axis=0)) / 2.0


# --------------------------------------------------------------------------------------
# Geometry (adapted from recognize.py)
# --------------------------------------------------------------------------------------
def extract_corners(polygon, shape):
    """Fit 4 corners to a GCP polygon. Returns (corners, None) or (None, reason)."""
    h, w = shape[:2]
    x, y = polygon[:, 0], polygon[:, 1]

    # Same border rule as recognize.py: partially captured GCPs are not usable
    t = BORDER_THRESHOLD
    if not (t * w <= x.min() and x.max() < (1 - t) * w
            and t * h <= y.min() and y.max() < (1 - t) * h):
        return None, "GCP touches image border"

    # Hand-drawn polygons can be slightly non-convex; the hull of a projected square is convex
    contour = cv.convexHull(polygon.astype(np.float32).reshape(-1, 1, 2))
    perimeter = cv.arcLength(contour, True)
    if len(contour) < 4 or perimeter <= 0:
        return None, "GCP polygon has fewer than 4 corners"

    if len(contour) == 4:
        return contour.reshape(4, 2), None

    for eps in EPSILON_FACTORS:
        approx = cv.approxPolyDP(contour, eps * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2), None
    return None, "GCP polygon could not be reduced to 4 corners"


def order_clockwise(corners):
    """Sort corners clockwise (image coordinates) around their centroid."""
    c = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - c[1], corners[:, 0] - c[0])
    return corners[np.argsort(angles)].astype(np.float32)


def quad_is_reasonable(corners):
    sides = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    if sides.min() < MIN_SIDE_PX:
        return "GCP too small"
    if sides.max() / sides.min() > MAX_SIDE_RATIO:
        return "GCP quad too distorted"
    return None


def apply_h(H, pts):
    return cv.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)


def rotation_to_canonical(side, size):
    """
    3x3 matrix rotating the rectified image so the number ends up on the right, upright.
    Mirrors recognize.py: left -> 180 deg, bottom -> 90 CCW, top -> 90 CW.
    """
    s = size - 1
    if side == "right":
        m = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    elif side == "left":      # (x, y) -> (s-x, s-y)
        m = [[-1, 0, s], [0, -1, s], [0, 0, 1]]
    elif side == "bottom":    # (x, y) -> (y, s-x)   == cv.ROTATE_90_COUNTERCLOCKWISE
        m = [[0, 1, 0], [-1, 0, s], [0, 0, 1]]
    elif side == "top":       # (x, y) -> (s-y, x)   == cv.ROTATE_90_CLOCKWISE
        m = [[0, -1, s], [1, 0, 0], [0, 0, 1]]
    else:
        raise ValueError(side)
    return np.array(m, dtype=np.float64)


def number_side(center_xy, size):
    """Which side of the rectified GCP holds the number, or None if it is ambiguous."""
    dx = center_xy[0] - (size - 1) / 2
    dy = center_xy[1] - (size - 1) / 2
    ax, ay = abs(dx), abs(dy)
    if max(ax, ay) < MIN_OFFSET * size:
        return None
    if max(ax, ay) < MIN_DOMINANCE * min(ax, ay):
        return None
    if ax >= ay:
        return "right" if dx > 0 else "left"
    return "bottom" if dy > 0 else "top"


def pipeline_crop(canonical, size):
    """The same fixed number crop that retrieve_numbers() takes (number on the right)."""
    half = canonical[:, size // 2:]
    r0, r1 = round(size * PIPE_ROWS[0]), round(size * PIPE_ROWS[1])
    c0, c1 = round(size * PIPE_COLS[0]), round(size * PIPE_COLS[1])
    return half[r0:r1, c0:c1]


# --------------------------------------------------------------------------------------
# Dataset processing
# --------------------------------------------------------------------------------------
def write_png(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(path), img):
        raise IOError(f"Could not write {path}")


def process_split(json_path, images_dir, out_root, args, writer, stats):
    split = json_path.parent.name
    with open(json_path, "r") as f:
        data = json.load(f)

    # Map by NAME (category ids are offset: id 1 is digit "0", id 11 is "gcp")
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    gcp_ids = {i for i, n in cat_name.items() if n == args.gcp_class}
    digit_of = {i: n for i, n in cat_name.items() if n in DIGITS}
    if not gcp_ids:
        sys.exit(f"{json_path}: no category named '{args.gcp_class}'. Found: {sorted(cat_name.values())}")
    if not digit_of:
        sys.exit(f"{json_path}: no digit categories ('0'..'9') found.")

    anns_by_image = defaultdict(list)
    for a in data["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    print(f"[{split}] {len(data['images'])} images, {len(data['annotations'])} annotations")

    for im in data["images"]:
        stats["images seen"] += 1
        anns = anns_by_image.get(im["id"], [])
        gcp_anns = [a for a in anns if a["category_id"] in gcp_ids]
        num_anns = [a for a in anns if a["category_id"] in digit_of]

        # Filter out images lacking GCPs or numbers
        if not gcp_anns:
            stats["image skipped: no GCP annotation"] += 1
            continue
        if not num_anns:
            stats["image skipped: no number annotation"] += 1
            continue

        img_path = images_dir / im["file_name"]
        img = cv.imread(str(img_path), cv.IMREAD_COLOR)
        if img is None:
            stats["image skipped: file missing/unreadable"] += 1
            continue
        if (img.shape[1], img.shape[0]) != (im["width"], im["height"]):
            stats["image skipped: size differs from annotation file"] += 1
            continue
        stats["images used"] += 1

        numbers = []
        for a in num_anns:
            pts = annotation_points(a, allow_bbox_fallback=True)
            if pts is not None:
                numbers.append({"ann": a, "pts": pts, "center": bbox_center(pts),
                                "digit": digit_of[a["category_id"]]})

        for g in gcp_anns:
            stats["GCPs seen"] += 1
            poly = annotation_points(g)
            if poly is None:
                stats["GCP skipped: no polygon"] += 1
                continue

            corners, reason = extract_corners(poly, img.shape)
            if corners is None:
                stats[f"GCP skipped: {reason}"] += 1
                continue
            corners = order_clockwise(corners)
            reason = quad_is_reasonable(corners)
            if reason:
                stats[f"GCP skipped: {reason}"] += 1
                continue

            # Perspective transform to an (unrotated) square
            dst = np.array([[0, 0], [args.size - 1, 0],
                            [args.size - 1, args.size - 1], [0, args.size - 1]], dtype=np.float32)
            H0 = cv.getPerspectiveTransform(corners, dst)
            # True centre of the GCP in the source image (survives perspective distortion)
            center_src = apply_h(np.linalg.inv(H0), np.array([[(args.size - 1) / 2] * 2]))[0]

            # Numbers lying inside this GCP; the closest to the GCP centre is its number
            quad = corners.reshape(-1, 1, 2).astype(np.float32)
            inside = [n for n in numbers
                      if cv.pointPolygonTest(quad, (float(n["center"][0]), float(n["center"][1])), False) >= 0]
            if not inside:
                stats["GCP skipped: no number inside GCP"] += 1
                continue
            if len(inside) > 1:
                stats["note: GCPs with >1 number inside (closest used)"] += 1
            num = min(inside, key=lambda n: np.linalg.norm(n["center"] - center_src))

            # Work out which side the number is on, from the label itself
            num_rect = apply_h(H0, num["pts"])
            side = number_side(bbox_center(num_rect), args.size)
            if side is None:
                stats["GCP skipped: number position ambiguous"] += 1
                continue

            # Single warp: perspective + rotation into canonical orientation
            H = rotation_to_canonical(side, args.size) @ H0
            canonical = cv.warpPerspective(img, H, (args.size, args.size))

            # Tight crop of the number from the canonical GCP, using its own polygon
            n_rect = apply_h(H, num["pts"])
            x0, y0 = np.floor(n_rect.min(axis=0)).astype(int) - NUMBER_PAD
            x1, y1 = np.ceil(n_rect.max(axis=0)).astype(int) + NUMBER_PAD
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, args.size), min(y1, args.size)
            if x1 - x0 < 4 or y1 - y0 < 4:
                stats["GCP skipped: degenerate number crop"] += 1
                continue
            number_img = canonical[y0:y1, x0:x1]

            gcp_out, num_out = canonical, number_img
            if args.green_only:  # matches the pipeline, which only uses the green channel
                gcp_out, num_out = canonical[:, :, 1], number_img[:, :, 1]

            digit = num["digit"]
            fname = f"{digit}_img{im['id']}_a{g['id']}.png"
            rel_gcp = Path(split) / "gcps" / digit / fname
            rel_num = Path(split) / "numbers" / digit / fname
            write_png(out_root / rel_gcp, gcp_out)
            write_png(out_root / rel_num, num_out)

            if args.pipeline_crops:
                crop = pipeline_crop(canonical, args.size)
                crop = crop[:, :, 1] if args.green_only else crop
                write_png(out_root / split / "numbers_pipeline" / digit / fname, crop)

            writer.writerow({
                "split": split, "digit": digit,
                "gcp_file": rel_gcp.as_posix(), "number_file": rel_num.as_posix(),
                "image_file": im["file_name"],
                "source_name": (im.get("extra") or {}).get("name", ""),
                "image_id": im["id"], "gcp_annotation_id": g["id"],
                "number_annotation_id": num["ann"]["id"],
                "number_side_in_source": side,
            })
            stats["GCPs saved"] += 1
            stats[f"saved digit {digit}"] += 1


def find_annotation_files(inputs):
    files = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            files += sorted(p.rglob("_annotations*.json"))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"Not found: {p}")
    if not files:
        sys.exit("No annotation files found (looking for _annotations*.json).")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="Dataset folder(s) (searched recursively for _annotations*.json) or COCO json file(s)")
    ap.add_argument("-o", "--output", default="gcp_dataset", help="Output folder (default: gcp_dataset)")
    ap.add_argument("--images-dir", help="Folder with the images, if not next to the json (single json only)")
    ap.add_argument("--size", type=int, default=SIZE, help=f"Rectified GCP side length (default {SIZE})")
    ap.add_argument("--gcp-class", default="gcp", help="Name of the GCP class in the json (default: gcp)")
    ap.add_argument("--green-only", action="store_true",
                    help="Save only the green channel, as recognize.py processes it")
    ap.add_argument("--pipeline-crops", action="store_true",
                    help="Also save the fixed-ratio number crop that retrieve_numbers() uses")
    args = ap.parse_args()

    json_files = find_annotation_files(args.inputs)
    if args.images_dir and len(json_files) != 1:
        sys.exit("--images-dir can only be used with a single annotation file")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    stats = Counter()

    fields = ["split", "digit", "gcp_file", "number_file", "image_file", "source_name",
              "image_id", "gcp_annotation_id", "number_annotation_id", "number_side_in_source"]
    with open(out_root / "manifest.csv", "w", newline="") as mf:
        writer = csv.DictWriter(mf, fieldnames=fields)
        writer.writeheader()
        for jp in json_files:
            images_dir = Path(args.images_dir) if args.images_dir else jp.parent
            process_split(jp, images_dir, out_root, args, writer, stats)

    print("\n=== Summary ===")
    for k in sorted(stats, key=lambda s: (s.startswith("saved digit"), s)):
        print(f"{k:55s} {stats[k]}")
    print(f"\nOutput written to {out_root.resolve()}")


if __name__ == "__main__":
    main()