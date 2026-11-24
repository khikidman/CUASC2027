from pathlib import Path
import math
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO
import vpi

device = ("cuda")

_rectified = vpi.Image((256, 256), vpi.Format.U8, flags=vpi.Backend.CUDA)

"""Load template matching files on CPU"""
def _load_templates() -> list[np.ndarray]:
    directory = Path("Numbers")
    files = sorted(
        [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() == ".jpg"]
    )
    return [cv.imread(str(entry))[:, :, 1] for entry in files]
templates = _load_templates()

# Exact 3x3 affine matrices for 90-degree rotations of a 256x128 (W x H) image -> 128x256
ROT90_CW_XFORM = np.array(
    [[0.0, -1.0, 127.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
)
ROT90_CCW_XFORM = np.array(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 255.0], [0.0, 0.0, 1.0]], dtype=np.float32
)

"""Load CNN"""
def _load_cnn() -> torch.fx.GraphModule: 
    exported_program = torch.export.load("team_air_gcp_localization_protoype.pt2")
    return exported_program.module()
"""
TODO: Optimize CNN (and YOLO)
"""
cnn = _load_cnn()

"""Find Squares Algo"""
def _extract_corners(polygon: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    h, w = shape[:2]
    x = polygon[:, 0]
    y = polygon[:, 1]

    threshold = 0.01
    valid = (
        threshold * w <= x.min()
        and x.max() < (1 - threshold) * w
        and threshold * h <= y.min()
        and y.max() < (1 - threshold) * h
    )

    if not valid:
        print("Captured image to close to borders, skipping")
        return None

    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    perimeter = cv.arcLength(contour, True)

    for epsilon_factor in (0.01, 0.02, 0.03, 0.05):
        approx = cv.approxPolyDP(contour, epsilon_factor * perimeter, True)
        if len(approx) == 4:
            if epsilon_factor == 0.05:
                print("Received high noise")
            return approx.reshape(4, 2)

    print("Captured non-square; likely partially captured")
    return None


def _rectify_target(vpi_image: vpi.Image, corners: np.ndarray) -> vpi.Image:
    # Sort corners cyclically around their centroid
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]

    destination = np.array(
        [[0, 0], [255, 0], [255, 255], [0, 255]],
        dtype=np.float32,
    )

    # 3x3 homography computed on CPU; perspective warp executed on VPI CUDA backend
    H = cv.getPerspectiveTransform(corners.astype(np.float32), destination)
    with vpi.Backend.CUDA:
        vpi_image.perspwarp(
            H,
            out=_rectified,
            interp=vpi.Interp.LINEAR,
            border=vpi.Border.ZERO,
        )
        return _rectified


"""Display Numbers from Square Algo"""
def _analyze_squares(vpi_rectified: vpi.Image, padding: int = 10) -> tuple[str, str]:
    N = 256
    size_sq = 48
    center = (N - size_sq) // 2

    # rlock_cpu() exposes a zero-copy unified-memory read view on Jetson Orin Nano
    with vpi_rectified.rlock_cpu() as image:
        regions = {
            "top": image[padding : padding + size_sq, center : center + size_sq],
            "bottom": image[
                N - padding - size_sq : N - padding, center : center + size_sq
            ],
            "left": image[center : center + size_sq, padding : padding + size_sq],
            "right": image[
                center : center + size_sq, N - padding - size_sq : N - padding
            ],
        }
        means = {name: float(region.mean()) for name, region in regions.items()}

    horizontal = means["left"] + means["right"]
    vertical = means["top"] + means["bottom"]

    if horizontal > vertical:
        axis = "horizontal"
        candidates = ("left", "right")
    else:
        axis = "vertical"
        candidates = ("top", "bottom")

    number_side = min(candidates, key=lambda x: means[x])
    return axis, number_side


def _retrieve_numbers(vpi_rectified: vpi.Image) -> np.ndarray:
    axis, number_side = _analyze_squares(vpi_rectified)

    with vpi.Backend.CUDA:
        if axis == "horizontal":
            # Zero-copy 128x256 ROI view via vpi.RectangleI(x, y, width, height)
            rect = (
                vpi.RectangleI(128, 0, 128, 256)
                if number_side == "right"
                else vpi.RectangleI(0, 0, 128, 256)
            )
            vpi_half = vpi_rectified.view(rect)
            if number_side == "left":
                vpi_half = vpi_half.image_flip(vpi.Flip.BOTH)
        else:
            # Zero-copy 256x128 ROI view via vpi.RectangleI(x, y, width, height)
            rect = (
                vpi.RectangleI(0, 128, 256, 128)
                if number_side == "bottom"
                else vpi.RectangleI(0, 0, 256, 128)
            )
            vpi_half = vpi_rectified.view(rect)
            xform = ROT90_CW_XFORM if number_side == "top" else ROT90_CCW_XFORM
            vpi_half = vpi_half.perspwarp(
                xform,
                size=(128, 256),
                interp=vpi.Interp.NEAREST,
                border=vpi.Border.ZERO,
            )

        # Zero-copy final crop: x in [40, 120) -> w=80, y in [74, 168) -> h=94
        vpi_num = vpi_half.view(vpi.RectangleI(40, 74, 80, 94))

    # Synchronize and copy the 94x80 patch to a contiguous CPU NumPy array
    return np.ascontiguousarray(vpi_num.cpu())


"""Classification with CPU template matching"""
def _conf_func(diff: float, val: float) -> float:
    return ((2 / (1 + math.exp(-20 * diff))) - 1) * (
        (2 / (1 + math.exp(-4.7 * val))) - 1
    )


def _template_match(img: np.ndarray) -> tuple[int, float]:
    scores = []
    for template in templates:
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        scores.append(cv.minMaxLoc(res)[1])

    arr = sorted(scores)
    return int(np.argmax(scores)), _conf_func(arr[-1] - arr[-2], arr[-1])


"""ALL THAT NEEDS TO BE IMPORTED"""
def return_points(results, vpi_image: vpi.Image, scale_factor: int) -> str:
    """
    results: Results from YOLO Model
    vpi_image: We are going to have the images as 640\*scale_factor x 640\*scale_factor px VPI Image
    scale_factor: We rescale to (640\*scale_factor, 640\*scale_factor), so we need to rescale back
    """

    for i, result in enumerate(results):
        if result.masks is None:
            print(f"No results found in image {i}")
            continue

        for polygon in result.masks.xy:
            corners = _extract_corners(polygon*scale_factor, (640*scale_factor, 640*scale_factor))
            if corners is None:
                continue

            centroid = np.mean(corners, axis=0)  # (x, y) in pixels
            vpi_rectified = _rectify_target(vpi_image, corners)
            num_img = _retrieve_numbers(vpi_rectified)
            num, conf = _template_match(num_img)
            mod = "Tem"

            if conf < 0.85:
                num_tensor = TF.to_tensor(num_img).unsqueeze(0).to(device)
                with torch.inference_mode():
                    cnn_output = cnn(num_tensor)
                    probs = torch.softmax(cnn_output, dim=1)
                    cnn_conf, cnn_num = torch.max(probs, dim=1)
                    if cnn_conf.item() > conf:
                        mod = "CNN"
                        num = int(cnn_num.item())
                        conf = float(cnn_conf.item())

            if conf < 0.7:
                print("Cannot reliably detect image; skipping")
                continue

            res = f"Number: {num}\nPixel coordinates: {centroid}\nModel: {mod}\nConfidence: {conf:.4f}\n\n"
            print(f"Number: {num}\nPixel coordinates: {centroid}\nModel: {mod}\nConfidence: {conf:.4f}\n\n")
            return res
